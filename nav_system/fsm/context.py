#!/usr/bin/env python
"""状态机共享上下文 (NavContext)：管理跨状态共享的位姿平滑、路线规划、VLM 事务及控制平滑器。"""
from __future__ import annotations

import time
import numpy as np
from typing import Any, Optional, Type, TYPE_CHECKING

from nav_page import YawSmoother, OutputSmoother
from nav_constants import (
    FOLLOW_YAW_EMA,
    FOLLOW_OUT_EMA,
    FOLLOW_DEADBAND,
    FOLLOW_SLEW,
    NAV_SMOOTH_POS,
    NAV_SMOOTH_YAW,
    ODOM_YAW_OFFSET,
    POST_ESCAPE_HOLD,
    POST_ESCAPE_CONVERGE_DIST,
    POST_ESCAPE_CONVERGE_YAW,
)
from nav_control import _angle_diff
from nav_helpers import _odom_nav_pose, extract_target_3d_from_snapshot, _get_accumulated_map
from nav_runtime import NavigationRuntime
from nav_vlm import VlmJob

if TYPE_CHECKING:
    from fsm.base_state import BaseNavState
    from policies.base_policy import BaseNavigationPolicy
    from nav_services import NavServices


class NavContext:
    """导航状态机全生命周期共享运行环境。"""

    def __init__(self, services: NavServices, policy: BaseNavigationPolicy):
        self.services = services
        self.policy = policy
        # 复用 legacy 的剩余路径校验缓存，避免 FOLLOW 每帧重复扫描同一
        # 条路径；缓存会按路径对象、waypoint 索引和障碍快照版本失效。
        self.runtime = NavigationRuntime()

        # ---- 路线与目标管理 ----
        self.patrol_target: Optional[tuple] = None          # (x, z) 巡逻语义目标
        self.patrol_goal_nav: Optional[tuple] = None        # (x, z) 巡逻安全站位
        self.patrol_goal_nav_target: Optional[tuple] = None # 站位对应的目标快照
        self.patrol_source: Optional[str] = None           # "frontier" | "vlm_dir"

        self.final_target: Optional[tuple] = None           # (x, z) 最终语义目标
        self.final_goal_nav: Optional[tuple] = None         # (x, z) 终点安全站位
        self.goal_source: Optional[str] = None              # "vlm_det"

        self.path: Optional[list] = None                    # 当前活跃路径 [(x, z), ...]
        self.path_idx: int = 0                              # 活跃路径前进游标

        # ---- 异常中断与脱困恢复上下文 ----
        self.resume_state_cls: Optional[Type[BaseNavState]] = None
        self.escape_target: Optional[tuple] = None
        self.escape_semantic_target: Optional[tuple] = None
        self.escape_replan_from_semantic: bool = False

        # ---- 位姿平滑与 Odom 推算持久状态 ----
        self.anchor_kf: Optional[tuple] = None
        self.anchor_cur: Optional[tuple] = None
        self.kf_prev_pos: Optional[tuple] = None
        self.nav_smooth: Optional[tuple] = None
        self.odom_anchor: Optional[tuple] = None
        self.slam_anchor: Optional[tuple] = None
        self.plan_slam_anchor: Optional[tuple] = None
        self.plan_odom_anchor: Optional[tuple] = None
        self.plan_pose: Optional[tuple] = None
        self.nav_source: str = "slam"
        self.use_odom: bool = False
        self.post_escape: bool = False
        self.post_escape_t: float = 0.0
        self.prev_escaping: bool = False
        self.escaping_active: bool = False
        self.yaw_sign: float = -1.0

        # ---- 控制指令与平滑器 ----
        self.yaw_smoother = YawSmoother(alpha=FOLLOW_YAW_EMA)
        self.out_smoother = OutputSmoother(
            alpha=FOLLOW_OUT_EMA, deadband=FOLLOW_DEADBAND, slew=FOLLOW_SLEW
        )
        self.last_cmd: str = "idle"
        self.last_action_key: Optional[tuple] = None
        self.d_tgt: Optional[float] = None
        self.d_robot: Optional[float] = None
        self.lookahead_target: Optional[tuple] = None

        # ---- VLM 异步事务管理 ----
        self.vlm_epoch: int = 0
        self.vlm_request_id: int = 0
        self.vlm_latest: tuple = (None, None)  # (dets, masks)
        self.presence_confirmed: bool = False
        self.auto_vlm_asked: bool = False
        self.auto_vlm_ask_t: float = 0.0
        self.auto_det_t: float = 0.0
        self.vlm_reask_pending: bool = False
        self.final_bbox_anchor: Optional[dict] = None

        # ---- 扫描与 RELOC 全局状态 ----
        self.boot_scan_done: bool = self.policy.default_boot_scan_done
        self.reloc_giveup: bool = False
        self.reloc_prev_state_cls: Optional[Type[BaseNavState]] = None
        self.final_from_reloc: bool = False

    def invalidate_vlm(self, reason: str = ""):
        """使所有未完成的异步 VLM HTTP 任务失效。"""
        self.vlm_epoch += 1

    def clear_active_path(self):
        """清空当前活跃路径。"""
        self.path = None
        self.path_idx = 0
        self.runtime.invalidate_path_validation()

    def clear_patrol(self, clear_target: bool = True):
        """清理巡逻目标与站位。"""
        self.patrol_goal_nav = None
        self.patrol_goal_nav_target = None
        if clear_target:
            self.patrol_target = None
            self.patrol_source = None

    def clear_final(self, clear_target: bool = True):
        """清理最终目标与站位。"""
        self.final_goal_nav = None
        if clear_target:
            self.final_target = None
            self.goal_source = None

    def reset_smoothers(self):
        """重置航向与速度输出平滑器。"""
        self.yaw_smoother.reset()
        self.out_smoother.reset()

    def log_action(self, tag: str, method: str, detail: str = ""):
        """动作名 + 移动方式 仅在切换时打印一次。"""
        key = (tag, method)
        if key != self.last_action_key:
            line = f"[nav_2d] >>> ACTION={tag}  METHOD={method}"
            if detail:
                line += f"  {detail}"
            print(line)
            self.last_action_key = key

    # ---- 统一的位姿融合更新算法 ----
    def update_pose_estimation(
        self,
        raw_pose: Optional[tuple],
        kf_pose: Optional[tuple],
        mode: Any,
        odom_now: Optional[tuple],
        preturn_active: bool,
        preturn_slam_anchor: Optional[tuple],
        preturn_odom_anchor: Optional[tuple],
    ) -> tuple[Optional[tuple], Optional[tuple], bool, str]:
        """计算融合后的当前导航位姿 (cur_pose) 与规划位姿 (_plan_pose)。"""
        cur_pose = raw_pose
        kf_updated = False

        if raw_pose is not None and kf_pose is not None:
            kf_shift = (
                (kf_pose[0] - (self.kf_prev_pos[0] if self.kf_prev_pos else kf_pose[0])) ** 2
                + (kf_pose[1] - (self.kf_prev_pos[1] if self.kf_prev_pos else kf_pose[1])) ** 2
            ) ** 0.5
            if self.kf_prev_pos is None or kf_shift > 0.02:
                self.anchor_kf = kf_pose
                self.anchor_cur = raw_pose
                self.kf_prev_pos = (kf_pose[0], kf_pose[1])
                kf_updated = True
            elif self.anchor_kf is None:
                self.anchor_kf = kf_pose
                self.anchor_cur = raw_pose
                kf_updated = True

            fx = self.anchor_kf[0] + (raw_pose[0] - self.anchor_cur[0])
            fz = self.anchor_kf[1] + (raw_pose[1] - self.anchor_cur[1])
            fyaw = self.anchor_kf[2] + _angle_diff(raw_pose[2], self.anchor_cur[2])

            if self.nav_smooth is None:
                self.nav_smooth = (fx, fz, fyaw)
            else:
                sx = self.nav_smooth[0] + NAV_SMOOTH_POS * (fx - self.nav_smooth[0])
                sz = self.nav_smooth[1] + NAV_SMOOTH_POS * (fz - self.nav_smooth[1])
                syaw = self.nav_smooth[2] + NAV_SMOOTH_YAW * _angle_diff(fyaw, self.nav_smooth[2])
                self.nav_smooth = (sx, sz, syaw)
            cur_pose = self.nav_smooth

        nav_source = "slam"
        use_odom = False

        # 脱困结束瞬间 -> 进入 odom 保持窗口
        if self.prev_escaping and not self.escaping_active:
            self.post_escape = True
            self.post_escape_t = time.time()
        if self.escaping_active:
            self.post_escape = False
        self.prev_escaping = self.escaping_active

        if cur_pose is not None:
            use_odom = (
                (mode is not None and getattr(mode, "name", "") == "RELOC")
                or self.escaping_active
                or self.post_escape
                or preturn_active
            )
            if use_odom and odom_now is not None:
                slam_pose_fresh = cur_pose
                if preturn_active and preturn_slam_anchor is not None and preturn_odom_anchor is not None:
                    odom_pose = _odom_nav_pose(
                        preturn_slam_anchor, preturn_odom_anchor, odom_now, yaw_sign=self.yaw_sign
                    )
                else:
                    if self.odom_anchor is None or self.slam_anchor is None:
                        self.slam_anchor = slam_pose_fresh
                        self.odom_anchor = odom_now
                    odom_pose = _odom_nav_pose(
                        self.slam_anchor, self.odom_anchor, odom_now, yaw_sign=self.yaw_sign
                    )

                if self.post_escape:
                    dx = slam_pose_fresh[0] - odom_pose[0]
                    dz = slam_pose_fresh[1] - odom_pose[1]
                    dpos = (dx * dx + dz * dz) ** 0.5
                    dyaw = abs(_angle_diff(slam_pose_fresh[2], odom_pose[2]))
                    if dpos < POST_ESCAPE_CONVERGE_DIST and dyaw < POST_ESCAPE_CONVERGE_YAW:
                        self.post_escape = False
                    elif time.time() - self.post_escape_t > POST_ESCAPE_HOLD:
                        self.post_escape = False

                cur_pose = odom_pose
                nav_source = "odom"
                self.nav_smooth = None
            else:
                self.slam_anchor = cur_pose
                self.odom_anchor = odom_now

        if use_odom:
            plan_pose = cur_pose
        elif cur_pose is not None and odom_now is not None:
            plan_yaw_diff = (
                abs(_angle_diff(cur_pose[2], self.plan_slam_anchor[2]))
                if self.plan_slam_anchor is not None
                else float("inf")
            )
            if kf_updated or plan_yaw_diff > 0.17:
                self.plan_slam_anchor = cur_pose
                self.plan_odom_anchor = odom_now
            if self.plan_slam_anchor is not None and self.plan_odom_anchor is not None:
                plan_pose = _odom_nav_pose(
                    self.plan_slam_anchor, self.plan_odom_anchor, odom_now, yaw_sign=self.yaw_sign
                )
            else:
                plan_pose = cur_pose
        else:
            plan_pose = cur_pose

        self.use_odom = use_odom
        self.nav_source = nav_source
        self.plan_pose = plan_pose
        return cur_pose, plan_pose, use_odom, nav_source

    # ---- 统一的 VLM 异步任务提交包装 ----
    def submit_vlm_job(
        self,
        kind: str,
        img: np.ndarray,
        target: Optional[str],
        prompt: Optional[str],
        cur_pose: tuple,
        nav_y: float,
        step: int,
        current_state_name: str,
        allow_map_fallback: bool = False,
        **extra_ctx,
    ) -> bool:
        """从当前帧冻结快照并安全异步提交给 VLM Worker。"""
        if self.services.vlm_worker.busy() or img is None or cur_pose is None:
            return False

        pointcloud_2d = self.services.slam.get_pointcloud_2d()
        if pointcloud_2d is None:
            return False

        if self.use_odom:
            half_yaw = 0.5 * float(cur_pose[2])
            camera_pose = (
                float(cur_pose[0]),
                float(nav_y),
                float(cur_pose[1]),
                0.0,
                float(np.sin(half_yaw)),
                0.0,
                float(np.cos(half_yaw)),
            )
        else:
            pose_full = self.services.slam.get_pose_full()
            if pose_full is None:
                return False
            camera_pose = tuple(float(v) for v in pose_full[:7])

        context = {
            "pointcloud_2d": np.array(pointcloud_2d, copy=True),
            "image_shape": tuple(int(v) for v in img.shape[:2]),
            "camera_pos_3d": camera_pose[:3],
            "camera_pose_3d": camera_pose,
            "nav_y": float(nav_y),
            "allow_map_fallback": bool(allow_map_fallback),
            "use_odom": bool(self.use_odom),
            "submit_pose": tuple(float(v) for v in cur_pose),
        }
        context.update(extra_ctx)

        req_id = self.vlm_request_id + 1
        job = VlmJob(
            request_id=req_id,
            kind=kind,
            image=np.array(img, copy=True),
            target=target,
            prompt=prompt,
            submit_step=step,
            submit_state=current_state_name,
            epoch=self.vlm_epoch,
            context=context,
        )
        if self.services.vlm_worker.submit(job):
            self.vlm_request_id = req_id
            return True
        return False
