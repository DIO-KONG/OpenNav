document.addEventListener("DOMContentLoaded", () => {
    let currentEpisode = null;
    let episodeData = null;
    let isPlaying = false;
    let playRaf = null;
    let playWallStart = 0;
    let playEpisodeStartTs = 0;
    /**
     * 时间轴索引（timelineSteps 的下标），不是原始导航 step。
     * timelineSteps[i] 才是真实 step 编号（如 1, 3, 5, ..., 61）。
     */
    let currentIndex = 0;
    let maxIndex = 0;
    /** 合并 poses/frames/statuses/events/vlm 后的有序 step 列表 */
    let timelineSteps = [];
    let playSpeed = 1.0;

    /**
     * 加载后一次性建好的 O(1) 查找缓存。
     * 播放每帧只读 cache，不再扫 JSONL。
     * @type {null|{
     *   pose: Array, status: Array, frame: Array,
     *   ts: number[], firstTs: number, lastTs: number,
     *   posePrefix: Array[],  // posePrefix[i] = step<=timelineSteps[i] 的位姿序列
     *   activeEvent: number[], activeVlm: number[], activeMemory: number[],
     *   activeConsole: number[],
     * }}
     */
    let stepCache = null;
    /** 当前已成功绘入 canvas 的快照 URL */
    let displayedFrameUrl = "";
    /** 每次请求快照递增，丢弃过期 onload/draw */
    let snapRequestId = 0;
    /** 播放代际：stopPlay 时递增，使 in-flight 播放回调失效 */
    let playGeneration = 0;
    /** 播放中是否正在等待当前快照 ready */
    let playAdvancePending = false;

    /**
     * 预加载池: url -> { url, image, state, promise, queued }
     * state: idle | loading | ready | error
     */
    const preloadPool = new Map();
    const PRELOAD_LIMIT = 10;
    const PRELOAD_AHEAD = 4;
    const MAX_PRELOAD_IN_FLIGHT = 1;
    const preloadQueue = [];
    let preloadInFlight = 0;

    // UI Elements — 缺关键节点直接抛错，避免静默失败难排查
    function requireEl(id) {
        const el = document.getElementById(id);
        if (!el) {
            throw new Error(
                `[Episode Viewer] 缺少必需 DOM 节点 #${id}。` +
                `请检查 templates/index.html 与 static/js/app.js 是否一致。`
            );
        }
        return el;
    }

    const epSelect = requireEl("ep-select");
    const refreshEpBtn = requireEl("refresh-ep-btn");
    const epKindFilterEl = requireEl("ep-kind-filter");
    const epAllFiltersEl = requireEl("ep-all-filters");
    const epDateFilterEl = requireEl("ep-date-filter");
    const epHourFilterEl = requireEl("ep-hour-filter");
    const epModeBtn = requireEl("ep-mode-btn");
    const epMarkBtn = requireEl("ep-mark-btn");
    const epDeleteBtn = requireEl("ep-delete-btn");
    // 快照区用 canvas 渲染（支持裁剪模式）
    const snapshotCanvas = requireEl("snapshot-canvas");
    const snapshotCtx = snapshotCanvas.getContext("2d");
    if (!snapshotCtx) {
        throw new Error("[Episode Viewer] snapshot-canvas 无法取得 2d context");
    }
    const noSnapshot = requireEl("no-snapshot");
    const snapshotModeBtns = requireEl("snapshot-mode-btns");
    const canvas = requireEl("trajectory-canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("[Episode Viewer] trajectory-canvas 无法取得 2d context");
    }

    /** 快照显示模式: "full" | "map" */
    let snapMode = "full";
    const SNAPSHOT_LAYOUT = {
        containerWidth: 1200,
        gap: 12,
        border: 2,
        imageTop: 60,
        titleBaseline: 35,
        separator: 4,
    };

    const slider = requireEl("timeline-slider");
    const btnPlay = requireEl("btn-play");
    const btnFirst = requireEl("btn-first");
    const btnPrev = requireEl("btn-prev");
    const btnNext = requireEl("btn-next");
    const btnLast = requireEl("btn-last");
    const speedSelect = requireEl("speed-select");

    const currentStepEl = requireEl("current-step");
    const maxStepEl = requireEl("max-step");
    const currentTimeEl = requireEl("current-time");
    const totalTimeEl = requireEl("total-time");

    const valState = requireEl("val-state");
    const valPose = requireEl("val-pose");
    const valDtgt = requireEl("val-dtgt");
    const valDobs = requireEl("val-dobs");
    const hudPose = requireEl("hud-pose");
    const hudPath = requireEl("hud-path");

    const eventsUl = requireEl("event-items-ul");
    const vlmUl = requireEl("vlm-items-ul");
    const memoryUl = requireEl("memory-items-ul");
    const logPre = requireEl("log-text-pre");
    const consoleUl = requireEl("console-items-ul");
    const stepJumpInput = requireEl("step-jump-input");
    const stepJumpBtn = requireEl("step-jump-btn");

    // ---- Episode 列表筛选 / success 标记 ----
    /** "recent" = 最新 5 条; "all" = 全部 + 日期小时筛选 */
    let listMode = "recent";
    let epKindFilter = "all"; // all | ep | eps
    let epDateFilter = ""; // YYYY-MM-DD or ""
    let epHourFilter = ""; // "" or "0".."23"
    /** @type {Map<string, object>} */
    let episodeMetaById = new Map();

    // 填充小时下拉 00-23（一次性）
    (function initHourOptions() {
        for (let h = 0; h < 24; h++) {
            const opt = document.createElement("option");
            opt.value = String(h);
            opt.textContent = `${String(h).padStart(2, "0")}:00`;
            epHourFilterEl.appendChild(opt);
        }
    })();

    function syncFilterControlsUi() {
        const isAll = listMode === "all";
        epAllFiltersEl.style.display = isAll ? "flex" : "none";
        epModeBtn.textContent = isAll ? "返回最近" : "查看全部";
        epModeBtn.title = isAll ? "回到最新 5 条" : "查看全部 Episode，可按日期/小时筛选";
    }

    /**
     * 根据当前选中 episode 更新标记按钮。
     * @returns {void}
     */
    function updateMarkButton() {
        const meta = currentEpisode ? episodeMetaById.get(currentEpisode) : null;
        epMarkBtn.classList.remove("is-success", "is-unmark");
        if (!meta) {
            epMarkBtn.disabled = true;
            epMarkBtn.textContent = "标记成功";
            epMarkBtn.title = "请先选择 Episode";
            updateDeleteButton();
            return;
        }
        if (!meta.ended) {
            epMarkBtn.disabled = true;
            epMarkBtn.textContent = meta.success ? "取消标记" : "标记成功";
            epMarkBtn.title = "仅已结束的 Episode（存在 summary.json）可标记";
            updateDeleteButton();
            return;
        }
        if (meta.success) {
            epMarkBtn.disabled = false;
            epMarkBtn.textContent = "取消标记";
            epMarkBtn.title = "取消 success：eps_ → ep_";
            epMarkBtn.classList.add("is-unmark");
        } else {
            epMarkBtn.disabled = false;
            epMarkBtn.textContent = "标记成功";
            epMarkBtn.title = "标记 success：ep_ → eps_";
            epMarkBtn.classList.add("is-success");
        }
        updateDeleteButton();
    }

    /**
     * 删除按钮：仅已结束 episode 可删。
     * @returns {void}
     */
    function updateDeleteButton() {
        const meta = currentEpisode ? episodeMetaById.get(currentEpisode) : null;
        if (!meta) {
            epDeleteBtn.disabled = true;
            epDeleteBtn.title = "请先选择 Episode";
            return;
        }
        if (!meta.ended) {
            epDeleteBtn.disabled = true;
            epDeleteBtn.title = "仅已结束的 Episode（存在 summary.json）可删除";
            return;
        }
        epDeleteBtn.disabled = false;
        epDeleteBtn.title = `删除 ${meta.id}（不可恢复）`;
    }

    /**
     * 删除当前已结束 episode。
     * @returns {Promise<void>}
     */
    async function deleteCurrentEpisode() {
        if (!currentEpisode) {
            throw new Error("[Episode Viewer] 无当前 Episode，无法删除");
        }
        const id = currentEpisode;
        const meta = episodeMetaById.get(id);
        if (!meta || !meta.ended) {
            throw new Error("仅已结束的 Episode 可删除");
        }
        const ok = window.confirm(
            `确认删除 episode\n${id}\n\n此操作将永久删除目录，不可恢复。`
        );
        if (!ok) return;

        const res = await fetch(`/api/episodes/${encodeURIComponent(id)}`, {
            method: "DELETE",
        });
        let body = null;
        try {
            body = await res.json();
        } catch (err) {
            body = null;
        }
        if (!res.ok) {
            const code = (body && body.error) || res.statusText || String(res.status);
            const detail = body && body.detail ? ` (${body.detail})` : "";
            const msg = `删除失败: ${code}${detail}`;
            console.error(msg, body);
            throw new Error(msg);
        }
        currentEpisode = null;
        episodeData = null;
        await loadEpisodeList({ preferId: null });
    }

    /**
     * 标记 / 取消 success。
     * @param {boolean} success
     * @returns {Promise<void>}
     */
    async function markSuccess(success) {
        if (!currentEpisode) {
            throw new Error("[Episode Viewer] 无当前 Episode，无法标记");
        }
        const id = currentEpisode;
        const res = await fetch(`/api/episodes/${encodeURIComponent(id)}/mark_success`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ success: !!success }),
        });
        let body = null;
        try {
            body = await res.json();
        } catch (err) {
            body = null;
        }
        if (!res.ok) {
            const code = (body && body.error) || res.statusText || String(res.status);
            const detail = body && body.detail ? ` (${body.detail})` : "";
            const msg = `标记失败: ${code}${detail}`;
            console.error(msg, body);
            throw new Error(msg);
        }
        const newId = body && body.id;
        if (!newId) {
            throw new Error("标记失败: 响应缺少新 id");
        }
        currentEpisode = newId;
        await loadEpisodeList({ preferId: newId });
    }

    // ============================================================
    //  索引构建: 加载时一次 O(n)，播放时每帧 O(1)
    // ============================================================

    /**
     * 从多路 JSONL 流中收集全部 step，去重排序，作为时间轴主轴。
     * @returns {number[]} 升序 step 数组
     */
    function buildTimelineSteps() {
        const stepSet = new Set();
        const streams = [
            episodeData.poses,
            episodeData.frames,
            episodeData.statuses,
            episodeData.events,
            episodeData.vlm_events,
            episodeData.coord_events,
            episodeData.memory_events,
            episodeData.performance,
            episodeData.console_events,
        ];
        streams.forEach((arr) => {
            (arr || []).forEach((r) => {
                if (r && r.step !== undefined && r.step !== null && !isNaN(r.step)) {
                    stepSet.add(Number(r.step));
                }
            });
        });
        if (stepSet.size === 0 && (episodeData.poses || []).length > 0) {
            for (let i = 0; i < episodeData.poses.length; i++) stepSet.add(i);
        }
        return Array.from(stepSet).sort((a, b) => a - b);
    }

    /**
     * 按 step 升序排序的记录副本（仅含有效 step）。
     * @param {Array} records
     * @returns {Array}
     */
    function sortedByStep(records) {
        return (records || [])
            .filter((r) => r && r.step !== undefined && r.step !== null && !isNaN(r.step))
            .map((r) => ({ r, s: Number(r.step) }))
            .sort((a, b) => a.s - b.s);
    }

    /**
     * 在已按 step 升序的数组上，找 step <= target 的最后一条（floor）。
     * @param {Array<{r:object,s:number}>} sorted
     * @param {number} target
     * @returns {object|null}
     */
    function floorRecord(sorted, target) {
        if (!sorted.length) return null;
        let lo = 0;
        let hi = sorted.length - 1;
        let ans = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (sorted[mid].s <= target) {
                ans = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return ans >= 0 ? sorted[ans].r : sorted[0].r;
    }

    /**
     * 在已按 step 升序的数组上，找 |step-target| 最小的一条。
     * @param {Array<{r:object,s:number}>} sorted
     * @param {number} target
     * @returns {object|null}
     */
    function nearestRecord(sorted, target) {
        if (!sorted.length) return null;
        let lo = 0;
        let hi = sorted.length - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (sorted[mid].s < target) lo = mid + 1;
            else hi = mid;
        }
        let best = sorted[lo];
        if (lo > 0) {
            const prev = sorted[lo - 1];
            if (Math.abs(prev.s - target) <= Math.abs(best.s - target)) best = prev;
        }
        return best.r;
    }

    /**
     * 记录在列表中的“有效 step”（无 step 时用 ts 映射；加载时已尽量带 step）。
     * @param {object} r
     * @returns {number}
     */
    function recordStep(r) {
        if (r.step !== undefined && r.step !== null && !isNaN(r.step)) return Number(r.step);
        return 0;
    }

    /**
     * 对事件类列表，预计算每个 timeline 下标应对应的 active 行下标
     * （step <= cur 的最后一条）。
     * @param {Array} records
     * @param {number[]} steps timelineSteps
     * @returns {number[]} 与 timeline 等长，-1 表示尚无
     */
    function buildActiveIndexSeries(records, steps) {
        const out = new Array(steps.length).fill(-1);
        if (!records || !records.length || !steps.length) return out;
        const recSteps = records.map(recordStep);
        let j = -1;
        for (let i = 0; i < steps.length; i++) {
            const cur = steps[i];
            while (j + 1 < recSteps.length && recSteps[j + 1] <= cur) j++;
            out[i] = j;
        }
        return out;
    }

    /**
     * 加载完成后构建全量 step 缓存。
     * 播放路径只读此缓存，杜绝每帧扫全表。
     * @returns {void}
     */
    function buildStepCache() {
        const steps = timelineSteps;
        const n = steps.length;
        const posesSorted = sortedByStep(episodeData.poses);
        const statusSorted = sortedByStep(episodeData.statuses);
        const framesSorted = sortedByStep(episodeData.frames);

        const pose = new Array(n);
        const status = new Array(n);
        const frame = new Array(n);
        const ts = new Array(n);
        const posePrefix = new Array(n);

        // step -> 最早出现的 ts。旧 episode 同一 step 的各数据流时间不同，
        // 取最早值最接近该轮主循环开始；新 episode 会共享同一个 step_ts。
        const tsByStep = new Map();
        let firstTs = Infinity;
        let lastTs = -Infinity;
        [
            episodeData.frames,
            episodeData.poses,
            episodeData.statuses,
            episodeData.events,
            episodeData.vlm_events,
            episodeData.coord_events,
            episodeData.memory_events,
            episodeData.performance,
            episodeData.console_events,
        ].forEach((arr) => {
            (arr || []).forEach((r) => {
                if (!r || r.ts === undefined || r.ts === null) return;
                if (r.ts < firstTs) firstTs = r.ts;
                if (r.ts > lastTs) lastTs = r.ts;
                if (r.step !== undefined && r.step !== null && !isNaN(r.step)) {
                    const s = Number(r.step);
                    const recordTs = Number(r.ts);
                    const oldTs = tsByStep.get(s);
                    if (Number.isFinite(recordTs) &&
                            (oldTs === undefined || recordTs < oldTs)) {
                        tsByStep.set(s, recordTs);
                    }
                }
            });
        });
        if (!isFinite(firstTs)) firstTs = 0;
        if (!isFinite(lastTs)) lastTs = 0;

        // 双指针推进: pose/status/frame/ts 一次扫完, 播放时 O(1) 读
        let poseEnd = 0;
        let lastKnownTs = firstTs;
        for (let i = 0; i < n; i++) {
            const cur = steps[i];
            while (poseEnd < posesSorted.length && posesSorted[poseEnd].s <= cur) {
                poseEnd++;
            }
            // 共享底层引用: 每帧 slice 一次数组头, 避免每帧 filter
            posePrefix[i] = posesSorted.slice(0, poseEnd).map((x) => x.r);
            pose[i] = poseEnd > 0 ? posesSorted[poseEnd - 1].r : null;
            status[i] = floorRecord(statusSorted, cur);
            frame[i] = nearestRecord(framesSorted, cur);
            if (tsByStep.has(cur)) lastKnownTs = tsByStep.get(cur);
            ts[i] = lastKnownTs;
        }

        // 防御旧日志中的系统时钟回拨或乱序记录，保证播放时间单调。
        for (let i = 1; i < ts.length; i++) {
            if (!Number.isFinite(ts[i]) || ts[i] < ts[i - 1]) {
                ts[i] = ts[i - 1];
            }
        }
        if (ts.length) {
            firstTs = ts[0];
            lastTs = ts[ts.length - 1];
        }

        stepCache = {
            pose,
            status,
            frame,
            ts,
            firstTs,
            lastTs,
            posePrefix,
            activeEvent: buildActiveIndexSeries(episodeData.events || [], steps),
            activeVlm: buildActiveIndexSeries(episodeData.vlm_coord_events || [], steps),
            activeMemory: buildActiveIndexSeries(episodeData.memory_events || [], steps),
            activeConsole: buildActiveIndexSeries(episodeData.console_events || [], steps),
        };
    }

    /**
     * 将真实 step 映射为 timelineSteps 下标（二分最近邻）。
     * @param {number} step
     * @returns {number}
     */
    function stepToIndex(step) {
        if (!timelineSteps.length) return 0;
        const target = Number(step);
        if (!Number.isFinite(target)) return 0;
        let lo = 0;
        let hi = timelineSteps.length - 1;
        let bestIdx = 0;
        let bestDiff = Infinity;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            const v = timelineSteps[mid];
            const diff = Math.abs(v - target);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestIdx = mid;
            }
            if (v === target) return mid;
            if (v < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return bestIdx;
    }

    /**
     * 跳转到最接近 targetStep 的时间轴位置（同步 HUD + 快照 + 列表跟随）。
     * @param {number|string} rawStep
     * @returns {Promise<void>}
     */
    async function jumpToNearestStep(rawStep) {
        if (!timelineSteps.length) {
            throw new Error("当前无时间轴数据");
        }
        // 空串 Number("")===0 会误跳到 step0；必须先 trim 判空
        const raw = String(rawStep ?? "").trim();
        if (!raw) {
            throw new Error("请输入 step 数字");
        }
        const n = Number(raw);
        if (!Number.isFinite(n)) {
            throw new Error("请输入有效的 step 数字");
        }
        // 播放中跳转时先暂停，避免与 play 门闩抢 currentIndex
        if (isPlaying) stopPlay();
        const idx = stepToIndex(n);
        currentIndex = idx;
        // 输入框回填实际落到的 step，方便确认“最近”
        const landed = timelineSteps[idx];
        stepJumpInput.value = String(landed);
        // forceListFollow: 即使 active 行未变（同事件区间内跳转）也滚到可见
        await renderAndDisplay({ forceListFollow: true });
    }

    // ============================================================
    //  数据加载
    // ============================================================

    /**
     * 拉取 episode 列表并重建下拉。
     * @param {{preferId?: string|null}} [opts]
     * @returns {Promise<void>}
     */
    async function loadEpisodeList(opts) {
        const preferId =
            opts && Object.prototype.hasOwnProperty.call(opts, "preferId")
                ? opts.preferId
                : currentEpisode;
        try {
            const params = new URLSearchParams();
            params.set("kind", epKindFilter || "all");
            params.set("all", listMode === "all" ? "1" : "0");
            if (listMode === "all") {
                if (epDateFilter) {
                    params.set("date", epDateFilter);
                    // hour 仅在有 date 时生效
                    if (epHourFilter !== "") params.set("hour", epHourFilter);
                }
            }
            params.set("t", String(Date.now()));

            const res = await fetch(`/api/episodes?${params.toString()}`);
            if (!res.ok) {
                const text = await res.text();
                throw new Error(`列表 API ${res.status}: ${text}`);
            }
            const episodes = await res.json();
            if (!Array.isArray(episodes)) {
                throw new Error("列表 API 返回非数组");
            }

            episodeMetaById = new Map();
            epSelect.innerHTML = "";
            if (episodes.length === 0) {
                epSelect.innerHTML = `<option value="">未找到 Episode</option>`;
                currentEpisode = null;
                updateMarkButton();
                return;
            }
            episodes.forEach((ep) => {
                episodeMetaById.set(ep.id, ep);
                const opt = document.createElement("option");
                opt.value = ep.id;
                const mark = ep.success ? "✓ " : "";
                opt.textContent = `${mark}${ep.id} (${ep.started_at}) [${ep.frame_count} 帧]`;
                epSelect.appendChild(opt);
            });

            // 优先保留 preferId；否则选列表第一条（已倒序=最新）
            const stillExists = preferId && episodes.some((e) => e.id === preferId);
            const targetId = stillExists ? preferId : episodes[0].id;

            epSelect.value = targetId;
            if (targetId !== currentEpisode) {
                await loadEpisodeData(targetId);
            } else {
                currentEpisode = targetId;
                updateMarkButton();
            }
        } catch (err) {
            console.error("加载 Episode 列表失败:", err);
            throw err;
        }
    }

    async function loadEpisodeData(epId) {
        if (!epId) {
            currentEpisode = null;
            updateMarkButton();
            return;
        }
        currentEpisode = epId;
        updateMarkButton();
        stopPlay();
        displayedFrameUrl = "";
        snapRequestId++;
        playAdvancePending = false;
        preloadPool.clear();
        preloadQueue.length = 0;
        preloadInFlight = 0;
        stepCache = null;
        try {
            const res = await fetch(`/api/episodes/${encodeURIComponent(epId)}/data?t=${Date.now()}`);
            if (!res.ok) {
                const text = await res.text();
                throw new Error(`数据 API ${res.status}: ${text}`);
            }
            episodeData = await res.json();
            if (episodeData && episodeData.error) {
                throw new Error(`数据 API error: ${episodeData.error}`);
            }
            episodeData.vlm_coord_events = mergeVlmCoordEvents(
                episodeData.vlm_events,
                episodeData.coord_events
            );
            setupTimeline();
            loadLogs(epId);
            // 从当前点（起点）加载，不从虚构的“全片第0批”灌
            await renderAndDisplay();
            updateMarkButton();
        } catch (err) {
            console.error("加载 Episode 数据失败:", err);
            throw err;
        }
    }

    async function loadLogs(epId) {
        const cons = (episodeData && episodeData.console_events) || [];
        if (cons.length > 0 && consoleUl) {
            consoleUl.style.display = "block";
            logPre.style.display = "none";
            renderConsoleList();
            return;
        }
        if (consoleUl) consoleUl.style.display = "none";
        logPre.style.display = "block";
        try {
            const res = await fetch(`/episodes/${epId}/log?t=${Date.now()}`);
            if (res.ok) {
                const text = await res.text();
                renderPlainLogWithSeverity(text || "");
            } else {
                logPre.textContent = "无控制台日志文件";
            }
        } catch (err) {
            logPre.textContent = "日志加载失败";
        }
    }

    function formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return "00:00.00";
        const m = Math.floor(seconds / 60);
        const s = (seconds % 60).toFixed(2);
        return `${m.toString().padStart(2, "0")}:${s.padStart(5, "0")}`;
    }

    /**
     * 构建统一时间轴 + O(1) 查找缓存。
     */
    function setupTimeline() {
        if (!episodeData) return;
        timelineSteps = buildTimelineSteps();
        maxIndex = Math.max(0, timelineSteps.length - 1);
        currentIndex = 0;
        buildStepCache();

        slider.min = 0;
        slider.max = maxIndex;
        slider.value = 0;
        slider.step = 1;

        const realMaxStep = timelineSteps.length ? timelineSteps[timelineSteps.length - 1] : 0;
        maxStepEl.textContent = realMaxStep;

        const first = stepCache ? stepCache.firstTs : 0;
        const last = stepCache ? stepCache.lastTs : 0;
        totalTimeEl.textContent = formatTime(last - first);

        renderEventsList();
        renderVlmList();
        renderMemoryList();
        if ((episodeData.console_events || []).length > 0) {
            renderConsoleList();
        }
    }

    /**
     * 格式化 VLM data 为可读摘要。
     * @param {object} data
     * @returns {string}
     */
    function formatVlmDetail(data) {
        if (!data || typeof data !== "object") return String(data || "");
        const stage = data.stage || data.status || "?";
        const parts = [`[${stage}]`];
        if (data.target) parts.push(`tgt=${data.target}`);
        if (data.present !== undefined) parts.push(`present=${data.present}`);
        if (data.answer !== undefined && data.answer !== null) {
            const a = String(data.answer);
            parts.push(`ans=${a.length > 40 ? a.slice(0, 40) + "…" : a}`);
        }
        if (data.dir) parts.push(`dir=${data.dir}`);
        if (data.label) parts.push(`label=${data.label}`);
        if (data.target_xz) {
            const xz = data.target_xz;
            parts.push(`xz=(${Number(xz[0]).toFixed(2)},${Number(xz[1]).toFixed(2)})`);
        }
        if (data.bbox_2d_px) parts.push(`bbox_px=${JSON.stringify(data.bbox_2d_px)}`);
        else if (data.bbox_2d) parts.push(`bbox=${JSON.stringify(data.bbox_2d)}`);
        if (data.detail) parts.push(`detail=${data.detail}`);
        if (data.trigger) parts.push(`trig=${data.trigger}`);
        return parts.join(" ");
    }

    /**
     * VLM 与目标坐标投影属于同一条感知链路；合并后按 step/seq/ts 排序展示。
     * @param {Array} vlmEvents
     * @param {Array} coordEvents
     * @returns {Array}
     */
    function mergeVlmCoordEvents(vlmEvents, coordEvents) {
        const merged = [];
        (vlmEvents || []).forEach((r) => merged.push({ ...r, _detailKind: "vlm" }));
        (coordEvents || []).forEach((r) => merged.push({ ...r, _detailKind: "coord" }));
        merged.sort((a, b) => {
            const stepDiff = recordStep(a) - recordStep(b);
            if (stepDiff !== 0) return stepDiff;
            const seqA = a.seq !== undefined && a.seq !== null ? Number(a.seq) : Infinity;
            const seqB = b.seq !== undefined && b.seq !== null ? Number(b.seq) : Infinity;
            if (seqA !== seqB) return seqA - seqB;
            return Number(a.ts || 0) - Number(b.ts || 0);
        });
        return merged;
    }

    function formatCoordVector(values, digits = 2) {
        if (!Array.isArray(values)) return "-";
        return `(${values.map((v) => Number(v).toFixed(digits)).join(",")})`;
    }

    /**
     * 将 coord_debug 的 project / pc_region 事件压缩为一行可读摘要。
     * @param {object} data
     * @returns {string}
     */
    function formatCoordDetail(data) {
        if (!data || typeof data !== "object") return String(data || "");
        const stage = String(data.stage || "coord");
        const parts = [`[coord:${stage}]`];

        if (stage === "pc_region") {
            if (data.bbox) parts.push(`bbox=${formatCoordVector(data.bbox, 1)}`);
            if (data.n_pts !== undefined) parts.push(`pts=${data.n_pts}`);
            if (Array.isArray(data.X) && Array.isArray(data.Y) && Array.isArray(data.Z)) {
                const target = [data.X[2], data.Y[2], data.Z[2]];
                parts.push(`target_xyz=${formatCoordVector(target)}`);
                parts.push(
                    `range=X${formatCoordVector(data.X.slice(0, 2))}` +
                    `/Y${formatCoordVector(data.Y.slice(0, 2))}` +
                    `/Z${formatCoordVector(data.Z.slice(0, 2))}`
                );
            }
            if (data.cam_pos) parts.push(`cam=${formatCoordVector(data.cam_pos)}`);
            return parts.join(" ");
        }

        if (data.state) parts.push(`state=${data.state}`);
        if (data.bbox_px) parts.push(`bbox_px=${formatCoordVector(data.bbox_px, 0)}`);
        else if (data.bbox) parts.push(`bbox=${formatCoordVector(data.bbox, 1)}`);
        if (data.img_size) parts.push(`img=${formatCoordVector(data.img_size, 0)}`);
        if (data.target_3d) parts.push(`target_xyz=${formatCoordVector(data.target_3d)}`);
        else if (Object.prototype.hasOwnProperty.call(data, "target_3d")) parts.push("target_xyz=null");
        if (data.cam_pos_3d) parts.push(`cam=${formatCoordVector(data.cam_pos_3d)}`);
        if (data.cur_pose) parts.push(`pose_xzyaw=${formatCoordVector(data.cur_pose)}`);
        if (data.use_odom !== undefined) parts.push(`odom=${data.use_odom}`);
        if (data.allow_map_fallback !== undefined) parts.push(`fallback=${data.allow_map_fallback}`);
        return parts.join(" ");
    }

    // ============================================================
    //  严重级别: 只改字体颜色 (不改行背景)
    //  error 红 / warn 黄 / ok 绿
    //  对照 nav_debug.py stage 与 nav_auto.py 文案
    // ============================================================

    /**
     * 文本关键字分级。优先 error > warn > ok。
     * @param {string} text
     * @returns {"error"|"warn"|"ok"|""}
     */
    function classifyTextSeverity(text) {
        if (!text) return "";
        const s = String(text);

        // --- error (红) ---
        if (
            /\[ERROR\]/i.test(s) ||
            /\bERROR\b/.test(s) ||
            /\bError\b/.test(s) ||
            /\bException\b/i.test(s) ||
            /\bTraceback\b/i.test(s) ||
            /VLM 错误/.test(s) ||
            /规划不成功/.test(s) ||
            /异常/.test(s) ||
            /\bfailed\b/i.test(s) ||
            /\bfailure\b/i.test(s)
        ) {
            return "error";
        }

        // --- warn (黄) ---
        if (
            /\[WARN(?:ING)?\]/i.test(s) ||
            /\bWARN(?:ING)?\b/.test(s) ||
            /\bWarning\b/i.test(s) ||
            /警告/.test(s) ||
            /失败/.test(s) ||
            /bbox->3D 失败/.test(s) ||
            /路径规划失败/.test(s) ||
            /最终路径规划失败/.test(s) ||
            /位姿侵入障碍/.test(s) ||
            /未返回有效 bbox/.test(s) ||
            /未检测到目标/.test(s) ||
            /未找到可用 frontier/.test(s) ||
            /RRT 无解/.test(s) ||
            /vlm ask failed/i.test(s) ||
            /\bfail\b/i.test(s)
        ) {
            return "warn";
        }

        // --- ok (绿): 成功/到达/恢复 ---
        if (
            /恢复 TRACKING/.test(s) ||
            /VLM 目标\s*\(/.test(s) ||
            /-> FINAL_ADJUST/.test(s) ||
            /已到达/.test(s) ||
            /到达 sub-opt-goal/.test(s) ||
            /FOLLOW 到达/.test(s) ||
            /导航结束/.test(s) ||
            /Model loaded/.test(s) ||
            /Initialized/.test(s) ||
            /已启用/.test(s) ||
            /SLAM 已追上/.test(s) ||
            /脱困结束/.test(s) ||
            /\bsuccess(?:ful(?:ly)?)?\b/i.test(s) ||
            /\bok\b/i.test(s) && /\[.*\]/.test(s) // 避免普通英文 ok 误伤
        ) {
            return "ok";
        }
        return "";
    }

    /**
     * 状态/动作事件分级。
     * @param {object} ev
     * @returns {"error"|"warn"|"ok"|""}
     */
    function classifyEventSeverity(ev) {
        if (!ev) return "";
        const kind = String(ev.kind || "");
        const data = ev.data;
        const stateName =
            data && typeof data === "object"
                ? String(data.state || data.action || "")
                : "";
        const detailStr =
            typeof data === "object" && data !== null
                ? JSON.stringify(data)
                : String(data || "");
        const blob = `${kind} ${stateName} ${detailStr}`;

        // NavState 成功终态
        if (kind === "state" && stateName === "DONE") return "ok";
        // 终调/锁定目标: 视为正向进展
        if (kind === "state" && stateName === "FINAL_ADJUST") return "ok";
        if (kind === "state" && (stateName === "FINAL_PLAN" || stateName === "FINAL_FOLLOW")) {
            return "ok";
        }
        // ESCAPE / RELOC 相关偏警告
        if (kind === "state" && stateName === "ESCAPE") return "warn";

        if (/error|exception/i.test(kind)) return "error";
        if (/fail|warn/i.test(kind)) {
            const t = classifyTextSeverity(blob);
            return t || "warn";
        }
        return classifyTextSeverity(blob);
    }

    /**
     * VLM 记录分级（优先 stage，再看 present/dir/detail）。
     * 对照 nav_debug.record_vlm:
     *   presence|detect|detect_error|no_detection|bbox_3d_fail|
     *   direction|h_ask|locked_skip
     * @param {object} v
     * @returns {"error"|"warn"|"ok"|""}
     */
    function classifyVlmSeverity(v) {
        if (!v) return "";
        const data = v.data || {};
        const stage = String(data.stage || data.status || "").toLowerCase();

        // 错误
        if (stage === "detect_error" || stage === "error") return "error";
        // 软失败 / 未检出
        if (stage === "bbox_3d_fail" || stage === "no_detection") return "warn";
        // 成功检出目标
        if (stage === "detect" || stage === "detected") return "ok";
        // 存在性问询: present=true 绿; present=false 不标色
        if (stage === "presence" || stage === "inquiry") {
            if (data.present === true) return "ok";
            return "";
        }
        // 方向问询: 有效方向绿; none 黄
        if (stage === "direction") {
            const d = String(data.dir || "").toLowerCase();
            if (!d || d === "none" || d === "null") return "warn";
            return "ok";
        }
        // h_ask / locked_skip: 中性，再扫 detail
        const blob = formatVlmDetail(data);
        return classifyTextSeverity(blob);
    }

    function classifyCoordSeverity(v) {
        if (!v) return "";
        const data = v.data || {};
        const stage = String(data.stage || "").toLowerCase();
        if (stage === "pc_region") {
            return Number(data.n_pts || 0) > 0 ? "ok" : "warn";
        }
        if (stage === "project") {
            return data.target_3d ? "ok" : "warn";
        }
        return "";
    }

    /**
     * Console 行分级。
     * @param {object} c
     * @returns {"error"|"warn"|"ok"|""}
     */
    function classifyConsoleSeverity(c) {
        if (!c) return "";
        const stream = (c.data && c.data.stream) || "";
        const text = (c.data && (c.data.text || c.data.line)) || "";
        if (stream === "stderr") {
            const t = classifyTextSeverity(text);
            return t === "error" ? "error" : t || "warn";
        }
        return classifyTextSeverity(text);
    }

    /**
     * 严重级别 → 字体 class（仅 color）。
     * @param {"error"|"warn"|"ok"|""} sev
     * @returns {string}
     */
    function severityClass(sev) {
        if (sev === "error") return "sev-error";
        if (sev === "warn") return "sev-warn";
        if (sev === "ok") return "sev-ok";
        return "";
    }

    /**
     * 纯文本 nav_auto.log fallback：按行给字体上色。
     * @param {string} text
     * @returns {void}
     */
    function renderPlainLogWithSeverity(text) {
        if (!logPre) return;
        if (!text) {
            logPre.textContent = "日志为空";
            return;
        }
        const lines = String(text).split("\n");
        const frag = document.createDocumentFragment();
        logPre.textContent = "";
        lines.forEach((line, i) => {
            const span = document.createElement("span");
            const sev = classifyTextSeverity(line);
            const cls = severityClass(sev);
            span.className = cls ? `log-line ${cls}` : "log-line";
            span.textContent = line + (i < lines.length - 1 ? "\n" : "");
            frag.appendChild(span);
        });
        logPre.appendChild(frag);
    }

    // ============================================================
    //  事件列表渲染 + 点击 Seek
    // ============================================================

    function renderEventsList() {
        eventsUl.innerHTML = "";
        const events = episodeData.events || [];
        if (events.length === 0) {
            eventsUl.innerHTML = `<li class="empty-tip">暂无状态事件</li>`;
            return;
        }
        events.forEach((ev, idx) => {
            const li = document.createElement("li");
            li.dataset.evIndex = idx;
            const realStep = ev.step !== undefined ? Number(ev.step) : 0;
            li.dataset.step = realStep;
            const tsStr = new Date(ev.ts * 1000).toLocaleTimeString();

            // 信息栏去掉重复 source，并把 patrol_source 提到独立列
            let patrolSource = "";
            let detailStr = "";
            if (ev.data && typeof ev.data === "object") {
                const data = { ...ev.data };
                if (data.patrol_source !== undefined && data.patrol_source !== null) {
                    patrolSource = String(data.patrol_source);
                }
                delete data.source;
                delete data.patrol_source;
                const keys = Object.keys(data);
                detailStr = keys.length ? JSON.stringify(data) : "";
            } else {
                detailStr = String(ev.data || "");
            }

            const sev = severityClass(classifyEventSeverity(ev));
            const bodyCls = sev ? `ev-body ${sev}` : "ev-body";
            const patrolCls = patrolSource ? "ev-patrol" : "ev-patrol muted";
            const patrolText = patrolSource || "—";
            const infoCore = detailStr
                ? `[${escapeHtml(String(ev.kind || ""))}] ${escapeHtml(detailStr)}`
                : `[${escapeHtml(String(ev.kind || ""))}]`;
            li.innerHTML =
                `<span class="ev-time">${tsStr}</span>` +
                `<span class="ev-step">s${realStep}</span>` +
                `<span class="${patrolCls}" title="patrol_source">${escapeHtml(patrolText)}</span>` +
                `<span class="${bodyCls}">${infoCore}</span>`;
            li.addEventListener("click", () => {
                const targetStep = parseInt(li.dataset.step, 10);
                if (!isNaN(targetStep)) {
                    currentIndex = stepToIndex(targetStep);
                    renderCurrentFrame();
                }
            });
            eventsUl.appendChild(li);
        });
    }

    function renderVlmList() {
        vlmUl.innerHTML = "";
        const vlms = episodeData.vlm_coord_events || [];
        if (vlms.length === 0) {
            vlmUl.innerHTML = `<li class="empty-tip">暂无 VLM 或坐标记录</li>`;
            return;
        }
        vlms.forEach((v, idx) => {
            const li = document.createElement("li");
            li.dataset.vlmIndex = idx;
            const realStep = v.step !== undefined ? Number(v.step) : 0;
            li.dataset.step = realStep;
            const tsStr = new Date(v.ts * 1000).toLocaleTimeString();
            const isCoord = v._detailKind === "coord" || v.kind === "coord";
            const detailStr = isCoord ? formatCoordDetail(v.data) : formatVlmDetail(v.data);
            const rawStage = (v.data && (v.data.stage || v.data.status)) || "";
            const stage = isCoord ? `coord/${rawStage || "?"}` : rawStage;
            const sev = severityClass(
                isCoord ? classifyCoordSeverity(v) : classifyVlmSeverity(v)
            );
            // stage / 正文共用同一字体色；无分级时 stage 用 muted
            const stageCls = sev ? `ev-stage ${sev}` : "ev-stage muted";
            const bodyCls = sev ? `ev-body ${sev}` : "ev-body";
            li.innerHTML =
                `<span class="ev-time">${tsStr}</span>` +
                `<span class="ev-step">s${realStep}</span>` +
                `<span class="${stageCls}">${escapeHtml(String(stage))}</span>` +
                `<span class="${bodyCls}">${escapeHtml(detailStr)}</span>`;
            li.addEventListener("click", () => {
                const targetStep = parseInt(li.dataset.step, 10);
                if (!isNaN(targetStep)) {
                    currentIndex = stepToIndex(targetStep);
                    renderCurrentFrame();
                }
            });
            vlmUl.appendChild(li);
        });
    }

    function formatMemoryNumber(value, digits = 3) {
        if (value === null || value === undefined || value === "") return "n/a";
        const n = Number(value);
        return Number.isFinite(n) ? n.toFixed(digits) : "n/a";
    }

    function formatMemoryRange(lo, hi) {
        if (lo === null || lo === undefined || hi === null || hi === undefined) {
            return "n/a";
        }
        if (!Number.isFinite(Number(lo)) || !Number.isFinite(Number(hi))) {
            return "n/a";
        }
        return `${Number(lo).toFixed(3)}..${Number(hi).toFixed(3)}`;
    }

    function formatMemoryDetail(data) {
        if (!data || typeof data !== "object") return String(data || "");
        const stage = String(data.stage || "?");
        if (stage === "frontier_snapshot") {
            const maxText = data.max_radius_mem_id === null || data.max_radius_mem_id === undefined
                ? "n/a"
                : `id${data.max_radius_mem_id}/kf${data.max_radius_kf_id}/${formatMemoryNumber(data.radius_W_max)}`;
            const removedKfs = Array.isArray(data.removed_kf_ids) && data.removed_kf_ids.length
                ? ` kfs=${data.removed_kf_ids.join(",")}`
                : "";
            const frontier = data.frontier_found ? "found" : "not-found";
            return `total=${data.n_total || 0} walked=${data.n_walked || 0} cand=${data.n_candidates || 0} ` +
                `anchors=${data.n_anchors || 0} scale=${formatMemoryRange(data.anchor_scale_min, data.anchor_scale_max)} ` +
                `ratio=${formatMemoryRange(data.scale_ratio_min, data.scale_ratio_max)} ` +
                `radiusW=${formatMemoryRange(data.radius_W_min, data.radius_W_max)} max=${maxText} ` +
                `removed=${data.removed_last || 0}/${data.removed_total || 0}${removedKfs} ` +
                `oob=scale:${data.scale_oob_count || 0},radius:${data.radius_oob_count || 0},invalid:${data.invalid_scale_count || 0} ` +
                `frontier=${frontier}${data.frontier_message ? ` msg=${data.frontier_message}` : ""}`;
        }
        if (stage === "bounds_warning") {
            const id = data.mem_id === null || data.mem_id === undefined ? "new" : data.mem_id;
            const bounds = `[${data.lower ?? "-∞"},${data.upper ?? "+∞"}]`;
            return `mem=${id} kf=${data.kf_id ?? "?"} field=${data.field || "?"} ` +
                `value=${String(data.value)} bounds=${bounds} action=${data.action || "?"}`;
        }
        return JSON.stringify(data);
    }

    function classifyMemorySeverity(record) {
        const data = (record && record.data) || {};
        const stage = String(data.stage || "");
        if (stage === "bounds_warning") {
            const action = String(data.action || "");
            return action.includes("skip") || action.includes("reject") ? "error" : "warn";
        }
        if (stage === "frontier_snapshot") {
            if (Number(data.invalid_scale_count || 0) > 0 || Number(data.radius_oob_count || 0) > 0) {
                return "error";
            }
            if (Number(data.scale_oob_count || 0) > 0 || Number(data.removed_last || 0) > 0) {
                return "warn";
            }
            return "ok";
        }
        return "";
    }

    function renderMemoryList() {
        memoryUl.innerHTML = "";
        const records = episodeData.memory_events || [];
        if (records.length === 0) {
            memoryUl.innerHTML = `<li class="empty-tip">暂无 Memory 记录</li>`;
            return;
        }
        records.forEach((record, idx) => {
            const li = document.createElement("li");
            li.dataset.memoryIndex = idx;
            const realStep = record.step !== undefined && record.step !== null
                ? Number(record.step)
                : 0;
            li.dataset.step = realStep;
            const tsStr = record.ts ? new Date(record.ts * 1000).toLocaleTimeString() : "--";
            const stage = (record.data && record.data.stage) || "memory";
            const sev = severityClass(classifyMemorySeverity(record));
            const stageCls = sev ? `ev-stage ${sev}` : "ev-stage muted";
            const bodyCls = sev ? `ev-body ${sev}` : "ev-body";
            li.innerHTML =
                `<span class="ev-time">${tsStr}</span>` +
                `<span class="ev-step">s${realStep}</span>` +
                `<span class="${stageCls}">${escapeHtml(String(stage))}</span>` +
                `<span class="${bodyCls}">${escapeHtml(formatMemoryDetail(record.data))}</span>`;
            li.addEventListener("click", () => {
                const targetStep = parseInt(li.dataset.step, 10);
                if (!isNaN(targetStep)) {
                    currentIndex = stepToIndex(targetStep);
                    renderCurrentFrame();
                }
            });
            memoryUl.appendChild(li);
        });
    }

    function renderConsoleList() {
        if (!consoleUl) return;
        consoleUl.innerHTML = "";
        const cons = episodeData.console_events || [];
        if (cons.length === 0) {
            consoleUl.innerHTML = `<li class="empty-tip">暂无结构化 console 记录</li>`;
            return;
        }
        cons.forEach((c, idx) => {
            const li = document.createElement("li");
            li.dataset.consoleIndex = idx;
            const realStep =
                c.step !== undefined && c.step !== null ? Number(c.step) : 0;
            li.dataset.step = realStep;
            const tsStr = c.ts ? new Date(c.ts * 1000).toLocaleTimeString() : "--";
            const text = (c.data && (c.data.text || c.data.line)) || "";
            const seqStr = c.seq !== undefined ? `q${c.seq}` : "";
            const sev = severityClass(classifyConsoleSeverity(c));
            const bodyCls = sev ? `ev-console ${sev}` : "ev-console";
            li.innerHTML =
                `<span class="ev-time">${tsStr}</span>` +
                `<span class="ev-step">s${realStep}</span>` +
                `<span class="ev-seq">${seqStr}</span>` +
                `<span class="${bodyCls}">${escapeHtml(text)}</span>`;
            li.addEventListener("click", () => {
                const targetStep = parseInt(li.dataset.step, 10);
                if (!isNaN(targetStep)) {
                    currentIndex = stepToIndex(targetStep);
                    renderCurrentFrame();
                }
            });
            consoleUl.appendChild(li);
        });
    }

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    /** 播放中列表跟随滚动的最小间隔 (ms)，避免每帧强制 layout */
    const LIST_FOLLOW_MIN_INTERVAL_MS = 100;
    /** @type {WeakMap<HTMLElement, number>} ul -> 上次 follow 滚动时刻 */
    const listFollowLastTs = new WeakMap();
    /** @type {WeakMap<HTMLElement, number>} ul -> 上次 active 下标 */
    const listFollowLastIdx = new WeakMap();

    /**
     * 元素是否在可见 pane 中（display:none 的 dtab 不滚动）。
     * @param {HTMLElement} el
     * @returns {boolean}
     */
    function isElementVisible(el) {
        if (!el) return false;
        // offsetParent 在 fixed 下可能为 null；用 client rects 更稳
        return el.getClientRects().length > 0;
    }

    /**
     * 在最近的可滚动祖先容器内滚动 el，使其可见。
     * 不用 Element.scrollIntoView：避免拖动整页/快照视口。
     * @param {HTMLElement} el
     * @returns {void}
     */
    function scrollListItemIntoContainer(el) {
        if (!el || !isElementVisible(el)) return;
        let scroller = el.parentElement;
        while (scroller && scroller !== document.body) {
            const style = window.getComputedStyle(scroller);
            const oy = style.overflowY;
            if (
                (oy === "auto" || oy === "scroll" || oy === "overlay") &&
                scroller.scrollHeight > scroller.clientHeight + 1
            ) {
                break;
            }
            scroller = scroller.parentElement;
        }
        if (!scroller || scroller === document.body || scroller === document.documentElement) {
            return;
        }
        const cRect = scroller.getBoundingClientRect();
        const eRect = el.getBoundingClientRect();
        // 已在可视区内则不动，减少无谓 layout 写入
        if (eRect.top >= cRect.top + 2 && eRect.bottom <= cRect.bottom - 2) {
            return;
        }
        if (eRect.top < cRect.top) {
            scroller.scrollTop += eRect.top - cRect.top - 4;
        } else if (eRect.bottom > cRect.bottom) {
            scroller.scrollTop += eRect.bottom - cRect.bottom + 4;
        }
    }

    /**
     * 用预计算 active 下标高亮列表，并在需要时滚动到可见区域。
     * - 暂停/逐帧：active 变化立即跟随
     * - 播放中：仍跟随，但节流到 LIST_FOLLOW_MIN_INTERVAL_MS，且仅可见 pane
     * - forceScroll：step 查找等场景，即使 active 未变也滚到可见
     * @param {HTMLElement} ul
     * @param {number} activeIdx
     * @param {{forceScroll?: boolean}} [opts]
     */
    function applyListHighlight(ul, activeIdx, opts) {
        if (!ul) return;
        const forceScroll = !!(opts && opts.forceScroll);
        const prevIdx = listFollowLastIdx.get(ul);
        const lis = ul.children;
        let newlyActive = null;
        let activeLi = null;

        // 仅改 prev/next 两个 class，避免每帧扫全表
        if (prevIdx !== undefined && prevIdx >= 0 && lis[prevIdx] && prevIdx !== activeIdx) {
            lis[prevIdx].classList.remove("active-ev");
        }
        if (activeIdx >= 0 && lis[activeIdx] && !lis[activeIdx].classList.contains("empty-tip")) {
            lis[activeIdx].classList.add("active-ev");
            activeLi = lis[activeIdx];
            if (prevIdx !== activeIdx) newlyActive = activeLi;
        }
        listFollowLastIdx.set(ul, activeIdx);

        if (!activeLi) return;
        if (!isElementVisible(activeLi)) return; // 隐藏的 VLM/Console tab 不滚
        // 无 force 且 active 未变：不重复滚（播放步进靠 newlyActive）
        if (!forceScroll && !newlyActive) return;

        if (!isPlaying || forceScroll) {
            scrollListItemIntoContainer(activeLi);
            listFollowLastTs.set(ul, performance.now());
            return;
        }
        // 播放中节流跟随
        const now = performance.now();
        const last = listFollowLastTs.get(ul) || 0;
        if (now - last >= LIST_FOLLOW_MIN_INTERVAL_MS) {
            scrollListItemIntoContainer(activeLi);
            listFollowLastTs.set(ul, now);
        }
    }

    // ============================================================
    //  快照 URL / 限流预取 / 显示门闩
    // ============================================================

    /**
     * 由 frame 记录拼出可请求 URL。
     * @param {object|null} matchedFrame
     * @returns {string} 空串表示无图
     */
    function frameUrl(matchedFrame) {
        if (!matchedFrame || !matchedFrame.data || !matchedFrame.data.path) return "";
        const filename = matchedFrame.data.path.replace("frames/", "");
        return `/episodes/${currentEpisode}/frames/${filename}?s=${matchedFrame.step}`;
    }

    function setSnapshotCanvasSize(width, height) {
        if (snapshotCanvas.width !== width || snapshotCanvas.height !== height) {
            snapshotCanvas.width = width;
            snapshotCanvas.height = height;
        }
    }

    function rawSnapshotRegions(img, frameData) {
        if (!img || !img.naturalWidth) return;
        const iw = img.naturalWidth;
        const ih = img.naturalHeight;
        const rgbShape = frameData.rgb_shape;
        const mapShape = frameData.map_shape;
        const rgbW = rgbShape ? Number(rgbShape[1]) : null;
        const rgbH = rgbShape ? Number(rgbShape[0]) : ih;
        const mapW = mapShape ? Number(mapShape[1]) : null;
        const mapH = mapShape ? Number(mapShape[0]) : null;

        let splitX = rgbW;
        if (!splitX) {
            const guessMapW = Math.min(ih, iw - SNAPSHOT_LAYOUT.separator);
            splitX = iw - SNAPSHOT_LAYOUT.separator - guessMapW;
        }
        const actualMapW = mapW || (iw - splitX - SNAPSHOT_LAYOUT.separator);
        const actualMapH = mapH || Math.min(actualMapW, ih);
        return {
            rgb: { x: 0, y: 0, width: splitX, height: Math.min(rgbH, ih) },
            map: {
                x: splitX + SNAPSHOT_LAYOUT.separator,
                y: 0,
                width: actualMapW,
                height: Math.min(actualMapH, ih),
            },
        };
    }

    function drawRegion(img, region) {
        const width = Math.max(1, Math.round(region.width));
        const height = Math.max(1, Math.round(region.height));
        setSnapshotCanvasSize(width, height);
        snapshotCtx.clearRect(0, 0, width, height);
        snapshotCtx.drawImage(
            img,
            region.x,
            region.y,
            region.width,
            region.height,
            0,
            0,
            width,
            height
        );
    }

    function drawLegacyStreams(img, frameData) {
        const regions = rawSnapshotRegions(img, frameData);
        if (!regions) return;
        const boxWidth = (SNAPSHOT_LAYOUT.containerWidth - SNAPSHOT_LAYOUT.gap) / 2;
        const imageWidth = boxWidth - 2 * SNAPSHOT_LAYOUT.border;
        const rgbHeight = Math.max(
            1,
            Math.round(regions.rgb.height * imageWidth / regions.rgb.width)
        );
        const mapHeight = Math.max(
            1,
            Math.round(regions.map.height * imageWidth / regions.map.width)
        );
        const canvasHeight =
            SNAPSHOT_LAYOUT.imageTop
            + Math.max(rgbHeight, mapHeight)
            + 2 * SNAPSHOT_LAYOUT.border;
        const rightX = boxWidth + SNAPSHOT_LAYOUT.gap;

        setSnapshotCanvasSize(SNAPSHOT_LAYOUT.containerWidth, canvasHeight);
        snapshotCtx.fillStyle = "#111";
        snapshotCtx.fillRect(0, 0, SNAPSHOT_LAYOUT.containerWidth, canvasHeight);
        snapshotCtx.fillStyle = "#0f0";
        snapshotCtx.font = "bold 19px monospace";
        snapshotCtx.textBaseline = "alphabetic";
        snapshotCtx.fillText("RGB View", 0, SNAPSHOT_LAYOUT.titleBaseline);
        snapshotCtx.fillText("Top-Down Map", rightX, SNAPSHOT_LAYOUT.titleBaseline);
        snapshotCtx.imageSmoothingEnabled = true;
        snapshotCtx.imageSmoothingQuality = "high";

        const drawStream = (region, x, height) => {
            snapshotCtx.fillStyle = "#333";
            snapshotCtx.fillRect(
                x,
                SNAPSHOT_LAYOUT.imageTop,
                boxWidth,
                height + 2 * SNAPSHOT_LAYOUT.border
            );
            snapshotCtx.drawImage(
                img,
                region.x,
                region.y,
                region.width,
                region.height,
                x + SNAPSHOT_LAYOUT.border,
                SNAPSHOT_LAYOUT.imageTop + SNAPSHOT_LAYOUT.border,
                imageWidth,
                height
            );
        };
        drawStream(regions.rgb, 0, rgbHeight);
        drawStream(regions.map, rightX, mapHeight);
    }

    /**
     * 按记录的 layout 和当前 snapMode 绘制快照。
     * @param {HTMLImageElement} img
     * @param {object} frameData
     */
    function drawSnapshotCropped(img, frameData) {
        if (!img || !img.naturalWidth) return;
        const layout = frameData.layout || "raw_pair_v1";

        if (snapMode === "map") {
            if (layout === "streams_v1") {
                const stored = frameData.map_region;
                const region = Array.isArray(stored) && stored.length === 4
                    ? { x: stored[0], y: stored[1], width: stored[2], height: stored[3] }
                    : { x: 608, y: 62, width: 590, height: 590 };
                drawRegion(img, region);
                return;
            }
            const regions = rawSnapshotRegions(img, frameData);
            if (regions) drawRegion(img, regions.map);
            return;
        }

        if (layout === "streams_v1") {
            setSnapshotCanvasSize(img.naturalWidth, img.naturalHeight);
            snapshotCtx.clearRect(0, 0, img.naturalWidth, img.naturalHeight);
            snapshotCtx.drawImage(img, 0, 0);
            return;
        }
        drawLegacyStreams(img, frameData);
    }

    function trimSnapshotPool() {
        while (preloadPool.size > PRELOAD_LIMIT) {
            let removed = false;
            for (const [url, entry] of preloadPool) {
                if (url === displayedFrameUrl || entry.state === "loading") continue;
                preloadPool.delete(url);
                try {
                    if (entry.image) entry.image.src = "";
                } catch (_) { /* ignore */ }
                removed = true;
                break;
            }
            if (!removed) break;
        }
    }

    function startSnapshotLoad(entry) {
        if (!entry || entry.state !== "idle") return;
        entry.state = "loading";
        entry.image.src = entry.url;
    }

    function pumpPreloadQueue() {
        while (preloadInFlight < MAX_PRELOAD_IN_FLIGHT && preloadQueue.length) {
            const entry = preloadQueue.shift();
            entry.queued = false;
            if (entry.state !== "idle" || preloadPool.get(entry.url) !== entry) continue;
            preloadInFlight++;
            startSnapshotLoad(entry);
            const release = () => {
                preloadInFlight = Math.max(0, preloadInFlight - 1);
                pumpPreloadQueue();
            };
            entry.promise.then(release, release);
        }
    }

    /**
     * 取得/创建池条目。immediate=true 时立即开始加载（当前显示帧）。
     * error 状态会丢弃并重建，避免一次失败后永远无法重试。
     * @param {string} url
     * @param {boolean} immediate
     */
    function getSnapshotEntry(url, immediate = false) {
        let entry = preloadPool.get(url);
        // 失败条目不可复用：promise 已 reject，后续 jump/play 会一直失败
        if (entry && entry.state === "error") {
            preloadPool.delete(url);
            entry = null;
        }
        if (entry) {
            preloadPool.delete(url);
            preloadPool.set(url, entry); // LRU: 移到末尾
            if (immediate && entry.state === "idle") startSnapshotLoad(entry);
            return entry;
        }
        const image = new Image();
        image.decoding = "async";
        entry = { url, image, state: "idle", queued: false, promise: null };
        entry.promise = new Promise((resolve, reject) => {
            image.onload = async () => {
                try {
                    if (image.decode) await image.decode();
                } catch (_) {
                    // decode 失败仍可 drawImage
                }
                entry.state = "ready";
                resolve(image);
            };
            image.onerror = () => {
                entry.state = "error";
                reject(new Error(`快照加载失败: ${url}`));
            };
        });
        entry.promise.catch(() => {});
        preloadPool.set(url, entry);
        trimSnapshotPool();
        if (immediate) startSnapshotLoad(entry);
        return entry;
    }

    /**
     * 从 centerIdx 起小窗口预取；in-flight 限 1。不从 0 灌全片。
     * @param {number} fromIdx
     */
    function schedulePrefetch(fromIdx) {
        if (!stepCache || !currentEpisode) return;
        const end = Math.min(maxIndex, fromIdx + PRELOAD_AHEAD - 1);
        for (let i = Math.max(0, fromIdx); i <= end; i++) {
            const url = frameUrl(stepCache.frame[i]);
            if (!url || url === displayedFrameUrl) continue;
            const entry = getSnapshotEntry(url, false);
            if (entry.state === "idle" && !entry.queued) {
                entry.queued = true;
                preloadQueue.push(entry);
            }
        }
        pumpPreloadQueue();
    }

    /**
     * 显示 matchedFrame 对应快照；等待加载+绘制完成。
     * 过期请求 (seek/快进) resolve false 且不覆盖画面。
     * @param {object|null} matchedFrame
     * @param {number} [bindIndex] 发起时的 timeline 下标；若与 currentIndex 不一致则丢弃
     * @returns {Promise<boolean>}
     */
    async function displaySnapshot(matchedFrame, bindIndex) {
        const requestId = ++snapRequestId;
        const expectedIndex = bindIndex !== undefined ? bindIndex : currentIndex;
        const url = frameUrl(matchedFrame);
        if (!url) {
            if (requestId !== snapRequestId || expectedIndex !== currentIndex) return false;
            displayedFrameUrl = "";
            snapshotCanvas.style.display = "none";
            snapshotModeBtns.style.display = "none";
            noSnapshot.style.display = "flex";
            return true;
        }
        try {
            const entry = getSnapshotEntry(url, true);
            const img = await entry.promise;
            // 过期 seek / 播放又推进了：不覆盖更新的画面
            if (requestId !== snapRequestId || expectedIndex !== currentIndex) return false;
            const fd = (matchedFrame && matchedFrame.data) || {};
            drawSnapshotCropped(img, fd);
            displayedFrameUrl = url;
            snapshotCanvas.style.display = "block";
            snapshotModeBtns.style.display = "flex";
            noSnapshot.style.display = "none";
            return true;
        } catch (err) {
            if (requestId !== snapRequestId || expectedIndex !== currentIndex) return false;
            console.error("[Episode Viewer] 快照加载失败:", url, err);
            // 允许下次重试
            if (preloadPool.get(url) && preloadPool.get(url).state === "error") {
                preloadPool.delete(url);
            }
            displayedFrameUrl = "";
            snapshotCanvas.style.display = "none";
            snapshotModeBtns.style.display = "none";
            noSnapshot.style.display = "flex";
            return true;
        }
    }

    /**
     * 模式切换时重绘当前已显示帧（不重新请求）。
     */
    function redrawCurrentSnapshot() {
        if (!displayedFrameUrl || !stepCache) return;
        const matchedFrame = stepCache.frame[currentIndex];
        if (!matchedFrame) return;
        const entry = preloadPool.get(displayedFrameUrl);
        if (!entry || !entry.image || !entry.image.naturalWidth) return;
        const fd = matchedFrame.data || {};
        drawSnapshotCropped(entry.image, fd);
    }

    // ============================================================
    //  主渲染: O(1) 读缓存
    // ============================================================

    /**
     * 同步更新 HUD/轨迹/列表；快照异步显示由 renderAndDisplay 等待。
     * @param {{forceListFollow?: boolean}} [opts]
     * @returns {{frame: object|null, index: number}}
     */
    function renderCurrentFrameSync(opts) {
        const forceListFollow = !!(opts && opts.forceListFollow);
        if (!episodeData || !timelineSteps.length || !stepCache) {
            drawCanvasTrajectory([]);
            snapshotCanvas.style.display = "none";
            snapshotModeBtns.style.display = "none";
            noSnapshot.style.display = "flex";
            return { frame: null, index: currentIndex };
        }

        currentIndex = Math.min(Math.max(0, currentIndex), maxIndex);
        const curStep = timelineSteps[currentIndex];
        const curTs = stepCache.ts[currentIndex] || 0;
        const firstTs = stepCache.firstTs;

        slider.value = currentIndex;
        currentStepEl.textContent = curStep;
        currentTimeEl.textContent = formatTime(curTs - firstTs);

        // --- Pose (O(1)) ---
        const matchedPose = stepCache.pose[currentIndex];
        if (matchedPose) {
            const pData = matchedPose.data || matchedPose;
            valPose.textContent = `${(pData.x ?? 0).toFixed(2)}, ${(pData.z ?? 0).toFixed(2)}, ${(((pData.yaw || 0) * 180) / Math.PI).toFixed(0)}°`;
            hudPose.textContent = `(${(pData.x ?? 0).toFixed(2)}, ${(pData.z ?? 0).toFixed(2)})`;
        } else {
            valPose.textContent = "N/A (该 step 无 pose)";
            hudPose.textContent = "N/A";
        }

        // --- Status (O(1)) ---
        const matchedSt = stepCache.status[currentIndex];
        if (matchedSt && matchedSt.data) {
            valState.textContent = matchedSt.data.state || "UNKNOWN";
            valDtgt.textContent =
                matchedSt.data.distance_to_target !== undefined &&
                matchedSt.data.distance_to_target !== null
                    ? `${matchedSt.data.distance_to_target.toFixed(2)}m`
                    : "N/A";
            valDobs.textContent =
                matchedSt.data.distance_to_obstacle !== undefined &&
                matchedSt.data.distance_to_obstacle !== null
                    ? `${matchedSt.data.distance_to_obstacle.toFixed(2)}m`
                    : "N/A";
            hudPath.textContent = `${matchedSt.data.path_index || 0}/${matchedSt.data.path_size || 0}`;
        } else {
            valState.textContent = "N/A";
            valDtgt.textContent = "N/A";
            valDobs.textContent = "N/A";
            hudPath.textContent = "N/A";
        }

        const hlOpts = forceListFollow ? { forceScroll: true } : undefined;
        applyListHighlight(eventsUl, stepCache.activeEvent[currentIndex], hlOpts);
        applyListHighlight(vlmUl, stepCache.activeVlm[currentIndex], hlOpts);
        applyListHighlight(memoryUl, stepCache.activeMemory[currentIndex], hlOpts);
        applyListHighlight(consoleUl, stepCache.activeConsole[currentIndex], hlOpts);
        drawCanvasTrajectory(stepCache.posePrefix[currentIndex] || []);
        return { frame: stepCache.frame[currentIndex], index: currentIndex };
    }

    /**
     * 同步面板 + 等待当前快照 ready 后绘制，再从当前点小窗口预取。
     * @param {{forceListFollow?: boolean}} [opts]
     * @returns {Promise<boolean>}
     */
    async function renderAndDisplay(opts) {
        const { frame, index } = renderCurrentFrameSync(opts);
        const ok = await displaySnapshot(frame, index);
        // 仅当仍停在同一 index 时预取，避免过期回调用旧位置灌队列
        if (ok && index === currentIndex) schedulePrefetch(currentIndex + 1);
        return ok;
    }

    /** 兼容旧调用名：异步显示当前帧 */
    function renderCurrentFrame() {
        return renderAndDisplay();
    }

    function drawCanvasTrajectory(historyPoses) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!historyPoses || historyPoses.length === 0) return;

        let minX = Infinity,
            maxX = -Infinity,
            minZ = Infinity,
            maxZ = -Infinity;
        for (let i = 0; i < historyPoses.length; i++) {
            const d = historyPoses[i].data || historyPoses[i];
            if (d.x < minX) minX = d.x;
            if (d.x > maxX) maxX = d.x;
            if (d.z < minZ) minZ = d.z;
            if (d.z > maxZ) maxZ = d.z;
        }

        const pad = 1.0;
        minX -= pad;
        maxX += pad;
        minZ -= pad;
        maxZ += pad;
        const rangeX = maxX - minX || 1;
        const rangeZ = maxZ - minZ || 1;

        function toPx(x, z) {
            const px = ((x - minX) / rangeX) * canvas.width;
            const py = canvas.height - ((z - minZ) / rangeZ) * canvas.height;
            return { x: px, y: py };
        }

        ctx.beginPath();
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        for (let i = 0; i < historyPoses.length; i++) {
            const d = historyPoses[i].data || historyPoses[i];
            const pt = toPx(d.x, d.z);
            if (i === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
        }
        ctx.stroke();

        const lastP =
            historyPoses[historyPoses.length - 1].data ||
            historyPoses[historyPoses.length - 1];
        const lastPt = toPx(lastP.x, lastP.z);

        ctx.fillStyle = "#00d26a";
        ctx.beginPath();
        ctx.arc(lastPt.x, lastPt.y, 6, 0, Math.PI * 2);
        ctx.fill();

        const yaw = lastP.yaw || 0;
        const dirX = lastPt.x + Math.sin(yaw) * 16;
        const dirY = lastPt.y - Math.cos(yaw) * 16;
        ctx.beginPath();
        ctx.strokeStyle = "#00f078";
        ctx.lineWidth = 2;
        ctx.moveTo(lastPt.x, lastPt.y);
        ctx.lineTo(dirX, dirY);
        ctx.stroke();
    }

    // ============================================================
    //  播放控制: rAF 驱动，主线程忙时不堆积 setInterval 回调
    // ============================================================

    function togglePlay() {
        if (isPlaying) stopPlay();
        else startPlay();
    }

    /** 返回 episode 时间不晚于 targetTs 的最后一个时间轴下标。 */
    function timeToIndex(targetTs) {
        if (!stepCache || !stepCache.ts.length) return 0;
        let lo = 0;
        let hi = stepCache.ts.length - 1;
        let ans = 0;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (stepCache.ts[mid] <= targetTs) {
                ans = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return ans;
    }

    /**
     * 按记录的真实 ts 播放。墙钟时间是唯一锚点，JPEG 加载不会累积到播放时长；
     * 浏览器若短暂卡顿，恢复后直接追赶到此刻应显示的 step。
     */
    function startPlay() {
        if (currentIndex >= maxIndex) currentIndex = 0;
        isPlaying = true;
        playAdvancePending = false;
        const generation = ++playGeneration;
        btnPlay.innerHTML = `<i class="fa-solid fa-pause"></i>`;
        schedulePrefetch(currentIndex + 1);
        playWallStart = performance.now();
        playEpisodeStartTs = (stepCache && stepCache.ts.length)
            ? stepCache.ts[currentIndex]
            : 0;
        const tick = (now) => {
            if (!isPlaying || generation !== playGeneration) return;
            const elapsedEpisodeSeconds =
                ((now - playWallStart) / 1000.0) * playSpeed;
            const targetTs = playEpisodeStartTs + elapsedEpisodeSeconds;
            const targetIndex = Math.min(maxIndex, timeToIndex(targetTs));
            if (!playAdvancePending && targetIndex > currentIndex) {
                playAdvancePending = true;
                currentIndex = targetIndex;
                renderAndDisplay().finally(() => {
                    if (!isPlaying || generation !== playGeneration) return;
                    playAdvancePending = false;
                });
            }
            if (!playAdvancePending && currentIndex >= maxIndex &&
                    (!stepCache || targetTs >= stepCache.lastTs)) {
                stopPlay();
                return;
            }
            playRaf = requestAnimationFrame(tick);
        };
        playRaf = requestAnimationFrame(tick);
    }

    function stopPlay() {
        isPlaying = false;
        playAdvancePending = false;
        playGeneration++;
        btnPlay.innerHTML = `<i class="fa-solid fa-play"></i>`;
        if (playRaf !== null) {
            cancelAnimationFrame(playRaf);
            playRaf = null;
        }
    }

    // Event Listeners
    epSelect.addEventListener("change", (e) => {
        loadEpisodeData(e.target.value).catch((err) => {
            console.error(err);
            alert(String(err.message || err));
        });
    });
    refreshEpBtn.addEventListener("click", () => {
        loadEpisodeList({ preferId: currentEpisode }).catch((err) => {
            console.error(err);
            alert(String(err.message || err));
        });
    });

    epKindFilterEl.addEventListener("change", () => {
        epKindFilter = epKindFilterEl.value || "all";
        loadEpisodeList({ preferId: currentEpisode }).catch((err) => {
            console.error(err);
            alert(String(err.message || err));
        });
    });
    epDateFilterEl.addEventListener("change", () => {
        epDateFilter = epDateFilterEl.value || "";
        // 换日期时若未显式选小时可保留；空日期则清空小时限制也合理
        loadEpisodeList({ preferId: currentEpisode }).catch((err) => {
            console.error(err);
            alert(String(err.message || err));
        });
    });
    epHourFilterEl.addEventListener("change", () => {
        epHourFilter = epHourFilterEl.value;
        loadEpisodeList({ preferId: currentEpisode }).catch((err) => {
            console.error(err);
            alert(String(err.message || err));
        });
    });
    epModeBtn.addEventListener("click", () => {
        listMode = listMode === "all" ? "recent" : "all";
        syncFilterControlsUi();
        loadEpisodeList({ preferId: currentEpisode }).catch((err) => {
            console.error(err);
            alert(String(err.message || err));
        });
    });
    epMarkBtn.addEventListener("click", () => {
        const meta = currentEpisode ? episodeMetaById.get(currentEpisode) : null;
        if (!meta) {
            alert("请先选择 Episode");
            return;
        }
        const wantSuccess = !meta.success;
        epMarkBtn.disabled = true;
        markSuccess(wantSuccess)
            .catch((err) => {
                console.error(err);
                alert(String(err.message || err));
            })
            .finally(() => {
                updateMarkButton();
            });
    });

    epDeleteBtn.addEventListener("click", () => {
        epDeleteBtn.disabled = true;
        deleteCurrentEpisode()
            .catch((err) => {
                console.error(err);
                alert(String(err.message || err));
            })
            .finally(() => {
                updateMarkButton();
            });
    });

    syncFilterControlsUi();
    updateMarkButton();
    btnPlay.addEventListener("click", togglePlay);

    slider.addEventListener("input", (e) => {
        currentIndex = parseInt(e.target.value, 10);
        renderCurrentFrame();
    });
    // 拖拽结束后释放焦点，避免空格被 range 控件吃掉
    slider.addEventListener("change", () => {
        slider.blur();
    });
    slider.addEventListener("pointerup", () => {
        slider.blur();
    });

    btnFirst.addEventListener("click", () => {
        currentIndex = 0;
        renderCurrentFrame();
        btnFirst.blur();
    });
    btnLast.addEventListener("click", () => {
        currentIndex = maxIndex;
        renderCurrentFrame();
        btnLast.blur();
    });

    // 快照模式切换
    document.querySelectorAll(".snap-mode-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".snap-mode-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            snapMode = btn.dataset.mode || "full";
            redrawCurrentSnapshot();
            btn.blur();
        });
    });
    btnPrev.addEventListener("click", () => {
        if (currentIndex > 0) {
            currentIndex--;
            renderCurrentFrame();
        }
        btnPrev.blur();
    });
    btnNext.addEventListener("click", () => {
        if (currentIndex < maxIndex) {
            currentIndex++;
            renderCurrentFrame();
        }
        btnNext.blur();
    });

    speedSelect.addEventListener("change", (e) => {
        playSpeed = parseFloat(e.target.value);
        if (isPlaying) {
            stopPlay();
            startPlay();
        }
    });

    document.addEventListener("keydown", (e) => {
        // 仅在真正可输入文本的控件内跳过快捷键。
        // range/checkbox/button 等 input 不跳过——否则拖完进度条焦点留在 slider 上时，
        // 空格会被吞掉无法播放/暂停。
        const el = e.target;
        if (el) {
            const tag = (el.tagName || "").toUpperCase();
            if (tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable) return;
            if (tag === "INPUT") {
                const t = (el.type || "text").toLowerCase();
                // 可键入文本的类型才跳过；range/checkbox/radio/button 等放行
                const typingTypes = new Set([
                    "text", "search", "email", "password", "url", "tel",
                    "number", "date", "time", "datetime-local", "month", "week",
                ]);
                if (typingTypes.has(t)) return;
            }
        }

        if (e.code === "Space") {
            e.preventDefault();
            togglePlay();
        } else if (e.code === "ArrowLeft") {
            e.preventDefault();
            if (currentIndex > 0) {
                currentIndex--;
                renderCurrentFrame();
            }
        } else if (e.code === "ArrowRight") {
            e.preventDefault();
            if (currentIndex < maxIndex) {
                currentIndex++;
                renderCurrentFrame();
            }
        } else if (e.code === "Home") {
            e.preventDefault();
            currentIndex = 0;
            renderCurrentFrame();
        } else if (e.code === "End") {
            e.preventDefault();
            currentIndex = maxIndex;
            renderCurrentFrame();
        }
    });

    function setViewTab(tab) {
        const activeTab = tab === "split" ? "split" : "image";
        document.querySelectorAll(".tab-btn").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.tab === activeTab);
        });
        const canvasBox = document.querySelector(".canvas-box");
        const imageBox = document.querySelector(".image-box");
        canvasBox.style.display = activeTab === "split" ? "flex" : "none";
        imageBox.style.display = "flex";
    }

    // 左侧视角切换: 双视角 / 快照帧
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            setViewTab(btn.dataset.tab);
            btn.blur();
        });
    });
    setViewTab("image");

    // 右侧详情 tab: 状态事件 / VLM / Console
    document.querySelectorAll(".dtab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".dtab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".dtab-pane").forEach((p) => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(btn.dataset.dtab).classList.add("active");
            btn.blur();
        });
    });

    // Step 查找：输入数字 -> 最近 timeline step
    function onStepJump() {
        jumpToNearestStep(stepJumpInput.value).catch((err) => {
            console.error(err);
            alert(String(err.message || err));
        });
    }
    stepJumpBtn.addEventListener("click", () => {
        onStepJump();
        stepJumpBtn.blur();
    });
    stepJumpInput.addEventListener("keydown", (e) => {
        if (e.code === "Enter" || e.key === "Enter") {
            e.preventDefault();
            e.stopPropagation();
            onStepJump();
        }
    });

    loadEpisodeList().catch((err) => {
        console.error(err);
        alert(String(err.message || err));
    });
});
