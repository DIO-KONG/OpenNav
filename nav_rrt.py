"""2D RRT* 避障路径规划 (X-Z 平面)。

复用项目内 RRT.py 的 RRTStar2D 算法 (Open3D KDTreeFlann 碰撞查询),
主要增强:
  1. 障碍点压到 fixed_y 平面, 用 Open3D 3D KDTree 做本质上的 X-Z 2D 距离查询
     (与 RRT.py 同源, 不依赖 scipy)。
  2. 采样边界自动围绕 start/goal, 不再写死 (-5,5)。
  3. goal 必须由上层解析为安全导航站位，RRT 不再自行移动目标。
  4. 规划成功后做贪心 line-of-sight 捷径化简, 减少 waypoint 数量, 利于
     Pure Pursuit 跟随。
  5. 起点侵入障碍时不再硬失败: 播种"后退候选点"让树能找到脱困出口,
     由主循环的"过近即退"负责最终脱困。

机器人"体积"通过 robot_radius 体现: 碰撞判定 = 障碍点到路径点的最近距离
是否 >= robot_radius, 即把障碍向外膨胀了一圈机器人半径。
"""

import numpy as np
import open3d as o3d
import time


# 起点侵入障碍时的 path[0] 候选根参数。
# 候选点必须满足完整 robot_radius 净空；距离采样采用 0.1m 细分，
# 并将后退/侧向最大漂移限制在 0.8m，避免把过远的点作为 path[0]。
RETREAT_CANDIDATE_MIN_DIST = 0.40
RETREAT_CANDIDATE_MAX_DIST = 0.80
RETREAT_CANDIDATE_STEP = 0.10



def build_obstacle_tree(obstacle_pts, fixed_y=0.0):
    """用障碍点 (N,3) 构建 Open3D KDTreeFlann。

    所有障碍点被压到 fixed_y 高度平面, 因此随后的 3D 查询
    search_knn_vector_3d([x, fixed_y, z]) 返回的距离即为 X-Z 平面距离。
    返回 KDTreeFlann 对象, 或 None (无障碍)。
    """
    if o3d is None:
        return None
    if obstacle_pts is None or len(obstacle_pts) == 0:
        return None
    obs = np.asarray(obstacle_pts, dtype=np.float64)
    n = len(obs)
    pts = np.zeros((n, 3), dtype=np.float64)
    pts[:, 0] = obs[:, 0]
    pts[:, 1] = float(fixed_y)
    pts[:, 2] = obs[:, 2] if obs.shape[1] >= 3 else obs[:, 1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return o3d.geometry.KDTreeFlann(pcd)


def tree_nearest_dist(tree, fixed_y, x, z):
    """查询 (x,z) 到障碍点的最近距离 (X-Z 平面)。tree 为 None 时返回 inf。"""
    return tree_nearest_dists(tree, fixed_y, x, z, k=1)[0]


def tree_nearest_dists(tree, fixed_y, x, z, k=2):
    """查询前 ``k`` 个障碍点距离；不足 ``k`` 个时用 ``inf`` 补齐。"""
    count = max(1, int(k))
    if tree is None:
        return tuple(float("inf") for _ in range(count))
    q = np.array([float(x), float(fixed_y), float(z)], dtype=np.float64)
    _, _, dists_sq = tree.search_knn_vector_3d(q, count)
    distances = [float(np.sqrt(value)) for value in dists_sq
                 if np.isfinite(value) and value >= 0.0]
    distances.sort()
    distances.extend([float("inf")] * (count - len(distances)))
    return tuple(distances[:count])


def tree_supported_clearance(tree, fixed_y, x, z):
    """返回第二近障碍距离。

    导航规则要求两个最近障碍点都位于净空阈值内才否决，因此第二近距离
    是权威净空；单个孤立点不会独立触发碰撞。
    """
    return tree_nearest_dists(tree, fixed_y, x, z, k=2)[1]


def nearest_obstacle_point(tree, fixed_y, x, z):
    """查询 (x,z) 最近的障碍点坐标 (X-Z 平面), 返回 (ox, oz, dist) 或 (None, None, inf)。"""
    if tree is None:
        return None, None, np.inf
    q = np.array([float(x), float(fixed_y), float(z)], dtype=np.float64)
    _, idx, dists_sq = tree.search_knn_vector_3d(q, 1)
    if len(idx) == 0:
        return None, None, np.inf
    pts = np.asarray(tree.geometry.point_cloud.points)
    pt = pts[idx[0]]
    return float(pt[0]), float(pt[2]), float(np.sqrt(dists_sq[0]))


class RRTStar2D:
    """2D RRT* 规划器 (X-Z 平面, 带 robot_radius 碰撞膨胀)。"""

    def __init__(self, obstacles_3d, robot_radius, fixed_y=0.0,
                 bounds=None, max_iter=6000, step_size=0.2,
                 goal_tol=0.2, goal_bias=0.1,
                 max_time=1.5, validation_tree=None):
        self.radius = float(robot_radius)
        self.fixed_y = float(fixed_y)
        self.step = float(step_size)
        self.goal_tol = float(goal_tol)
        self.goal_bias = float(goal_bias)
        self.max_iter = int(max_iter)
        self.max_time = float(max_time)
        if obstacles_3d is not None and len(obstacles_3d) > 0:
            obs = np.asarray(obstacles_3d, dtype=np.float64)
            n = len(obs)
            pts = np.zeros((n, 3), dtype=np.float64)
            pts[:, 0] = obs[:, 0]
            pts[:, 1] = self.fixed_y
            pts[:, 2] = obs[:, 2] if obs.shape[1] >= 3 else obs[:, 1]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            self.tree = o3d.geometry.KDTreeFlann(pcd)
        else:
            self.tree = None
        # 降采样树只负责搜索；root/goal 使用完整障碍树作权威检查。
        self.validation_tree = validation_tree if validation_tree is not None \
            else self.tree
        self.bounds = bounds  # (low_2d, high_2d) 或 None
        self.nodes = []       # list of (point2d, parent_idx, cost)

    # ---- 碰撞 ----
    def _dist(self, p):
        """(x,z) 的权威支持净空：第二近障碍的 X-Z 距离。"""
        return tree_supported_clearance(
            self.tree, self.fixed_y, float(p[0]), float(p[1]))

    def is_collision_free(self, p):
        """点 (x,z) 是否远离所有障碍 >= robot_radius。"""
        return self._dist(p) >= self.radius

    def _validation_dist(self, p):
        return tree_supported_clearance(
            self.validation_tree, self.fixed_y, float(p[0]), float(p[1]))

    def is_authoritatively_free(self, p):
        return self._validation_dist(p) >= self.radius

    # ---- RRT 基本操作 ----
    def _sample(self):
        lo, hi = self.bounds
        return np.random.uniform(lo, hi, size=2)

    def _nearest(self, p):
        arr = np.array([n[0] for n in self.nodes], dtype=np.float64)
        return int(np.argmin(np.linalg.norm(arr - p, axis=1)))

    def _steer(self, frm, to):
        d = to - frm
        dist = np.linalg.norm(d)
        if dist <= self.step:
            return to.copy()
        return frm + d / dist * self.step

    # ---- 起点侵入: 播种后退候选点 ----
    def _retreat_candidates(self, start, goal):
        """起点在障碍内时, 在机器人"后方/侧方"找若干空闲候选, 作为树的额外根,
        让 RRT 能找到"先退一步再绕出去"的出口。"""
        direction = start - goal
        dl = np.linalg.norm(direction)
        unit = direction / dl if dl > 1e-6 else np.array([1.0, 0.0])
        cands = []
        distances = np.arange(
            RETREAT_CANDIDATE_MIN_DIST,
            RETREAT_CANDIDATE_MAX_DIST + RETREAT_CANDIDATE_STEP * 0.5,
            RETREAT_CANDIDATE_STEP,
            dtype=np.float64,
        )
        for d in distances:
            c = start + unit * d
            if self.is_authoritatively_free(c):
                cands.append(c)
        if dl > 1e-6:
            perp = np.array([-unit[1], unit[0]])
            for s in (1.0, -1.0):
                for d in distances:
                    c = start + perp * s * d
                    if self.is_authoritatively_free(c):
                        cands.append(c)
        return cands

    # ---- 规划 ----
    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        final_goal = goal
        if not self.is_authoritatively_free(final_goal):
            return None

        # 直线快速检查 (方案 B): start->final_goal 完全自由(考虑 robot_radius) 时
        # 直接返回直线, 不走 RRT。避免开放区域因随机采样出现绕路/折返弯路。
        # 前提: start 自身自由(否则是侵入脱困场景, 应交由下方 RRT 后退候选点处理),
        # final_goal 也通过权威净空检查，且整条线段自由。
        if (self.is_authoritatively_free(start) and
                self._segment_free(start, final_goal)):
            return [start.copy(), final_goal.copy()]

        # 起点侵入障碍时不能把侵入中的 start 作为可返回路径根；只使用满足
        # 完整 robot_radius 净空的后方/侧方候选点。没有安全根则本次规划失败，
        # 交由上层 ESCAPE fallback_retreat 后重新规划。
        if self.is_authoritatively_free(start):
            roots = [start.copy()]
        else:
            roots = [c.copy() for c in self._retreat_candidates(start, goal)]
            if not roots:
                return None

        # 自动采样边界 (围绕 start/goal, 留 1.5m 余量)
        if self.bounds is None:
            pts = np.vstack([start, final_goal])
            c = pts.mean(axis=0)
            half = np.max(np.abs(pts - c), axis=0) + 1.5
            self.bounds = (c - half, c + half)

        # 多 root 脱困时将真实 start -> root 位移计入总代价，避免较远的
        # path[0] 与最近候选同为零成本而仅由 RRT 随机扩展决定胜出者。
        # 正常自由起点只有 root=start，因此初始代价仍严格为 0。
        self.nodes = [
            (r, -1, float(np.linalg.norm(r - start))) for r in roots
        ]
        goal_idx = None
        best_cost = None   # 到 goal 的最优总代价 (含最后一段到 final_goal)
        t_start = time.time()
        for _ in range(self.max_iter):
            if time.time() - t_start > self.max_time:
                # 方案 A: 超时不再直接判无解, 而是停止迭代、保留已找到的最优解
                # (若从未找到 goal, 下方 goal_idx is None 仍返回 None)
                break
            if np.random.rand() < self.goal_bias:
                rp = final_goal
            else:
                rp = self._sample()
            ni = self._nearest(rp)
            np_ = self.nodes[ni][0]
            new = self._steer(np_, rp)
            if not self.is_collision_free(new):
                continue
            cost = self.nodes[ni][2] + np.linalg.norm(new - np_)
            self.nodes.append((new, ni, cost))
            if (np.linalg.norm(new - final_goal) < self.goal_tol and
                    self._segment_free(new, final_goal)):
                # 方案 A: RRT* 不立即返回, 而是用代价更小的 goal 端点替换并继续迭代,
                # 让后续更密集/更直接的采样逐步优化出更短更直的路径。
                _total = cost + np.linalg.norm(new - final_goal)
                if best_cost is None or _total < best_cost:
                    goal_idx = len(self.nodes) - 1
                    best_cost = _total

        if goal_idx is None:
            return None

        # 回溯路径 (goal -> start)
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(self.nodes[idx][0])
            idx = self.nodes[idx][1]
        path = path[::-1]
        # 路径必须以真实导航终点结束；最后连接段已在 goal 候选选择时检查。
        if np.linalg.norm(path[-1] - final_goal) > 1e-9:
            path.append(final_goal.copy())
        path = self._shortcut(path)
        return path

    # ---- 捷径化简: 贪心 line-of-sight ----
    def _shortcut(self, path):
        if len(path) <= 2:
            return path
        path = [np.asarray(p, dtype=np.float64) for p in path]
        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._segment_free(path[i], path[j]):
                    break
                j -= 1
            simplified.append(path[j])
            i = j
        return simplified

    def _segment_free(self, a, b, n=12):
        """线段 a->b 是否全程远离障碍 (用 radius*0.9 放宽, 避免砍掉贴边路)。"""
        if self.tree is None:
            return True
        for t in np.linspace(0.0, 1.0, n):
            p = a + (b - a) * t
            # if self._dist(p) < self.radius * 0.9:
            if self._dist(p) < self.radius:
                return False
        return True


def plan_with_rrt(start, goal, obstacle_pts, robot_radius,
                  fixed_y=0.0, max_iter=6000, max_obs=4000,
                  max_time=1.5, validation_tree=None):
    """对外接口: RRT* 避障规划。

    参数:
        start          (x, z) 起点 (机器人当前位置)
        goal           (x, z) 目标
        obstacle_pts   (N,3) 世界坐标点云 (x,y,z); 投影到 X-Z 平面
        robot_radius   机器人半径/体积 (m), 碰撞膨胀用
        fixed_y        碰撞查询高度平面 (机器人 nav_y)
        max_iter       RRT 最大迭代
        max_obs        障碍点降采样上限 (控时)
        max_time       单次规划最大耗时 (s), 防止无解时阻塞主循环
    返回:
        list[(x,z), ...] 或 None (无解/目标被完全包围)
    """
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    if obstacle_pts is None or len(obstacle_pts) == 0:
        return [start, goal]
    obs = np.asarray(obstacle_pts, dtype=np.float64)
    if obs.shape[1] < 3:
        # 只有 2D -> 补一个 y 列 (用 fixed_y)
        pad = np.full((len(obs), 1), float(fixed_y))
        obs = np.hstack([obs[:, :2], pad])
    if len(obs) > max_obs:
        idx = np.random.choice(len(obs), max_obs, replace=False)
        obs = obs[idx]
    planner = RRTStar2D(
        obs, robot_radius, fixed_y=fixed_y, max_iter=max_iter,
        max_time=max_time, validation_tree=validation_tree)
    path = planner.plan(start, goal)
    if path is None:
        return None
    return [(float(p[0]), float(p[1])) for p in path]


if __name__ == "__main__":
    # 自测: 中央一根柱子把 start/goal 隔开, 应能绕开
    np.random.seed(0)
    obs = []
    for th in np.linspace(0, 2 * np.pi, 80):
        obs.append([0.8 * np.cos(th), 0.0, 0.8 * np.sin(th)])  # 半径 0.8 的柱子
    obs = np.array(obs)
    p = plan_with_rrt((-2.0, 0.0), (2.0, 0.0), obs, robot_radius=0.35)
    if p is None:
        print("self-test: NO PATH (unexpected)")
    else:
        print(f"self-test: path with {len(p)} waypoints")
        print(p)
