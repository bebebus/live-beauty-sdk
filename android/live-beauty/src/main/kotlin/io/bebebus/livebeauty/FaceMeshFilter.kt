package io.bebebus.livebeauty

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * 478 点人脸网格的时间平滑（仅 Android，纯逻辑，只在 GL 线程使用，每个样本零堆分配）。取代 FaceTimeline 旧的网格分支。
 *
 * 以前 468 点共用一个截止频率 `3 Hz + 18 × 锚点速度`：速度没做导数低通、取 6 个锚点里的最大值、死区按像素标定，
 * 静止时噪声自己就把截止频率打开（仿真滤波系数 0.87–0.99，等于没滤），美妆、皮肤遮罩、美瞳逐帧抖 3–7 px@1080。
 * 现在分两层：
 * - 快层（刚体姿态）：弱透视 `m = t + A·L + Z·w`（8 个未知数；L、Z 为规范 3D 坐标除以规范脸宽，以稳定点加权质心为原点）
 *   就是 `m = t + s·P·R·X` 的仿射写法，转头、点头、平移的视差整体由它承担。最小二乘只用 MediaPipe 的 Procrustes 稳定点
 *   （33 个去掉随下颌运动的 136/365），正规矩阵是常量，类加载时预计算 `K = (DᵀWD)⁻¹DᵀW`，每个样本只做一次 8×62 乘法；
 *   姿态再过 One-Euro（速度为稳定点上的刚体位移 RMS，脸宽 / 秒）。
 * - 慢层（规范残差）：`r = A⁻¹(m − t − Z·w) − L` 只剩表情、个体脸型与透视二阶项，按中上脸、双眼、下半脸三个区域
 *   各一个截止频率强平滑（区域速度先按 [RES_DCUT] 低通再求 RMS，相邻顶点同区同频，不会剪切三角形）。
 * - 死区一律是自适应噪声底的倍数：噪声底取扣除刚体运动后的残差原始速度的最小统计量，头动时不被抬高；
 *   采样越稀（降级、CPU），最小截止频率按 √(33 ms / 间隔) 降低、平滑越强。
 * - 虹膜 468..477 不进残差：圆心在滤波后的眼角坐标系里单独 One-Euro，半径取眼宽 × 跨帧低通的比例；
 *   另输出两眼眨眼门控（滤波后网格的睁眼比按个人基线归一化，经 One-Euro 与施密特迟滞）。
 *
 * 坐标：输入、输出都是 478 × 2 GL 纹理坐标（y 向上）；内部度量空间 `m = (x, y / aspect)`，单位为画面宽。
 * 仿真依据（σ = 0.5 采样像素，216×384 小图，720p 输出）：静止残留 30 Hz 0.63 px、15 Hz 0.74 px、10 Hz 0.82 px
 * （旧实现 1.47 / 1.55 / 1.60 px）；30°/s 转头无噪声滞后 RMS 0.29 px。
 */
internal class FaceMeshFilter {
    // 度量空间工作缓冲（预分配，零运行期 GC）。
    private val mX = FloatArray(FaceMeshFrame.LANDMARK_COUNT)
    private val mY = FloatArray(FaceMeshFrame.LANDMARK_COUNT)
    private val outX = FloatArray(FaceMeshFrame.LANDMARK_COUNT)
    private val outY = FloatArray(FaceMeshFrame.LANDMARK_COUNT)
    private val rRawX = FloatArray(FACE)
    private val rRawY = FloatArray(FACE)
    private val rPrevX = FloatArray(FACE)
    private val rPrevY = FloatArray(FACE)
    private val rX = FloatArray(FACE)
    private val rY = FloatArray(FACE)
    private val drX = FloatArray(FACE)
    private val drY = FloatArray(FACE)
    private val q = FloatArray(POSE_SIZE)
    private val qRaw = FloatArray(POSE_SIZE)
    private val qRawPrev = FloatArray(POSE_SIZE)
    private val regionRms = FloatArray(REGIONS)
    private val regionAlpha = FloatArray(REGIONS)

    private var started = false
    private var lastTs = Long.MIN_VALUE
    private var vp = 0f // 刚体速度（脸宽 / 秒，POSE_DCUT 低通）
    private var noisePower = -1f // 残差原始速度功率的 EMA；< 0 表示未初始化
    private var noiseMin = 0f // 最小统计量（功率）

    // 虹膜：每只眼一份（0 = 解剖右眼 468..472，1 = 解剖左眼 473..477）。
    private val irisQx = FloatArray(2)
    private val irisQy = FloatArray(2)
    private val irisRawX = FloatArray(2)
    private val irisRawY = FloatArray(2)
    private val irisDx = FloatArray(2)
    private val irisDy = FloatArray(2)
    private val irisK = FloatArray(2)
    private val irisPower = FloatArray(2)
    private val irisMin = FloatArray(2)

    // 眨眼门控：每只眼一份。
    private val gateBase = floatArrayOf(-1f, -1f) // 个人睁眼比基线；< 0 表示未初始化（硬重置才清）
    private val gateX = FloatArray(2)
    private val gatePrev = FloatArray(2)
    private val gateDx = FloatArray(2)
    private val gateClosed = BooleanArray(2)
    private val gateValue = floatArrayOf(1f, 1f)
    private val gateOpen = FloatArray(2)

    /** 上一个样本是否因身份跳变（中心或尺度突变）开始新一轮。 */
    var jumped = false
        private set
    /** 刚体速度（脸宽 / 秒，已低通，含噪声）。 */
    var poseSpeed = 0f
        private set
    /** 刚体速度的死区（脸宽 / 秒，噪声底的倍数）；[poseSpeed] 不超过它时视为静止，不外推。 */
    var poseDead = Float.MAX_VALUE
        private set
    /** 自适应噪声底：扣除刚体运动后的残差原始速度 RMS（脸宽 / 秒）。 */
    var noiseFloor = 0f
        private set
    /** 去抖动后的脸部运动速度（画面宽 / 秒）：刚体与表情残差超出各自死区部分的较大者，供时域降噪暂停。 */
    var motionSpeed = 0f
        private set
    /** 最新样本的脸宽（度量，画面宽），即姿态尺度 √|det A|。 */
    var poseScale = 0f
        private set
    /** 解剖右眼 / 左眼的眨眼门控（0 = 闭眼，1 = 睁眼），供美瞳。 */
    val rightGate: Float get() = gateValue[0]
    val leftGate: Float get() = gateValue[1]
    /** 滤波后网格的睁眼比（竖直睑裂 / 眼宽）。 */
    val rightOpen: Float get() = gateOpen[0]
    val leftOpen: Float get() = gateOpen[1]

    /** 重新开始；[keepBaseline] 为 true 时保留两眼的个人睁眼比基线（丢脸清理时用，换镜头等硬重置不保留）。 */
    fun reset(keepBaseline: Boolean = false) {
        started = false; lastTs = Long.MIN_VALUE; vp = 0f; noisePower = -1f; noiseMin = 0f
        jumped = false; poseSpeed = 0f; poseDead = Float.MAX_VALUE; noiseFloor = 0f; motionSpeed = 0f; poseScale = 0f
        if (!keepBaseline) gateBase.fill(-1f)
        gateValue.fill(1f)
    }

    /**
     * 滤一个样本：[raw] 为原始 478 × 2 纹理坐标（采样时间 [ts] 严格递增），结果写入 [out]（478 × 2）与 [poseOut]
     * （滤波后的姿态 `[tx, ty, a11, a12, a21, a22, w1, w2]`，度量空间）。
     * @return [CONTINUE]：同一轮继续；[NEW_RUN]：首个样本、相隔超过 [MAX_GAP_MS] 或身份跳变（见 [jumped]），状态已从头开始。
     */
    fun filter(ts: Long, raw: FloatArray, aspect: Float, out: FloatArray, poseOut: FloatArray): Int {
        require(raw.size >= FaceMeshFrame.COORD_COUNT && out.size >= FaceMeshFrame.COORD_COUNT && poseOut.size >= POSE_SIZE)
        for (i in 0 until FaceMeshFrame.LANDMARK_COUNT) {
            mX[i] = raw[i * 2]
            mY[i] = raw[i * 2 + 1] / aspect
        }
        fitPose(mX, mY, qRaw)
        jumped = false
        var restart = !started || ts <= lastTs || ts - lastTs > MAX_GAP_MS
        if (!restart) {
            // 身份跳变门：中心跳过半张脸宽或尺度突变（换人、镜子、海报），整体从头开始，不把两张脸混在一起滤。
            val s = scaleOf(q)
            val sRaw = scaleOf(qRaw)
            if (hypot(qRaw[0] - q[0], qRaw[1] - q[1]) > JUMP_T * s || !(sRaw > 0f) || abs(ln(sRaw / s)) > JUMP_LOGS) {
                restart = true; jumped = true
            }
        }
        if (abs(detOf(qRaw)) < 1e-10f) {
            // 点集退化（FaceMeshFrame 已拦下绝大多数情况）：原样输出，下一个样本从头开始。
            started = false
            raw.copyInto(out, 0, 0, FaceMeshFrame.COORD_COUNT)
            qRaw.copyInto(poseOut, 0, 0, POSE_SIZE)
            return NEW_RUN
        }
        val teMs = if (restart) 0f else (ts - lastTs).toFloat()
        if (restart) startRun() else step(teMs)
        lastTs = ts
        // 输出：t + A·(L + r) + Z·w；虹膜由眼角坐标系重建。
        val tx = q[0]; val ty = q[1]; val a11 = q[2]; val a12 = q[3]; val a21 = q[4]; val a22 = q[5]; val w1 = q[6]; val w2 = q[7]
        for (i in 0 until FACE) {
            val lx = L_X[i] + rX[i]; val ly = L_Y[i] + rY[i]; val z = Z[i]
            outX[i] = tx + a11 * lx + a12 * ly + z * w1
            outY[i] = ty + a21 * lx + a22 * ly + z * w2
        }
        updateIris(0, IRIS_RIGHT, 33, 133, teMs, restart)
        updateIris(1, IRIS_LEFT, 362, 263, teMs, restart)
        updateGate(0, RIGHT_LIDS, 33, 133, teMs, restart)
        updateGate(1, LEFT_LIDS, 362, 263, teMs, restart)
        for (i in 0 until FaceMeshFrame.LANDMARK_COUNT) {
            out[i * 2] = outX[i]
            out[i * 2 + 1] = outY[i] * aspect
        }
        q.copyInto(poseOut, 0, 0, POSE_SIZE)
        poseScale = scaleOf(q)
        return if (restart) NEW_RUN else CONTINUE
    }

    private fun startRun() {
        started = true
        qRaw.copyInto(q); qRaw.copyInto(qRawPrev)
        residual(qRaw, rRawX, rRawY)
        rRawX.copyInto(rX); rRawY.copyInto(rY)
        rRawX.copyInto(rPrevX); rRawY.copyInto(rPrevY)
        drX.fill(0f); drY.fill(0f)
        vp = 0f; noisePower = -1f; noiseMin = 0f
        poseSpeed = 0f; poseDead = Float.MAX_VALUE; noiseFloor = 0f; motionSpeed = 0f
    }

    private fun step(teMs: Float) {
        val teSec = teMs / 1000f
        val comp = min(1f, sqrt(TE_REF_MS / teMs))
        val s = scaleOf(q)
        // 刚体速度：稳定点上由姿态变化预测的位移（加权 RMS，按坐标平均），除以脸宽与间隔。
        var acc = 0f
        for (j in BASIS.indices) {
            val b = BASIS[j]
            val mx = (qRaw[0] - qRawPrev[0]) + (qRaw[2] - qRawPrev[2]) * L_X[b] + (qRaw[3] - qRawPrev[3]) * L_Y[b] + (qRaw[6] - qRawPrev[6]) * Z[b]
            val my = (qRaw[1] - qRawPrev[1]) + (qRaw[4] - qRawPrev[4]) * L_X[b] + (qRaw[5] - qRawPrev[5]) * L_Y[b] + (qRaw[7] - qRawPrev[7]) * Z[b]
            acc += BASIS_W[j] * (mx * mx + my * my)
        }
        val spRaw = sqrt(acc / 2f) / s / teSec
        vp += alpha(POSE_DCUT, teMs) * (spRaw - vp)
        // 姿态病态（极端侧脸时仿射退化）：只更新快层，残差保持。
        val illPosed = abs(detOf(qRaw)) < DET_MIN * s * s
        if (!illPosed) {
            residual(qRaw, rRawX, rRawY)
            var sum = 0f
            for (i in 0 until FACE) {
                val dx = rRawX[i] - rPrevX[i]; val dy = rRawY[i] - rPrevY[i]
                sum += dx * dx + dy * dy
            }
            val rmsAll = sqrt(sum / FACE) / teSec
            // 噪声底：残差原始速度功率的 EMA（τ = NF_TAU_MS），取最小统计量（随时可降，上升每秒最多 NF_RISE 倍）。
            val power = rmsAll * rmsAll
            if (noisePower < 0f) { noisePower = power; noiseMin = power }
            noisePower += (1f - exp(-teMs / NF_TAU_MS)) * (power - noisePower)
            noiseMin = max(min(noiseMin * (1f + NF_RISE * teSec), noisePower), 1e-8f)
        }
        val floor = if (noisePower < 0f) 0f else sqrt(noiseMin)
        noiseFloor = floor
        val dead = K_POSE * C_POSE * floor
        val a = alpha(POSE_FC_MIN * comp + POSE_BETA * max(0f, vp - dead), teMs)
        for (k in 0 until POSE_SIZE) q[k] += a * (qRaw[k] - q[k])
        qRaw.copyInto(qRawPrev)
        poseSpeed = vp
        poseDead = dead
        var resExcess = 0f
        if (!illPosed) {
            val ad = alpha(RES_DCUT, teMs)
            for (i in 0 until FACE) {
                drX[i] += ad * ((rRawX[i] - rPrevX[i]) / teSec - drX[i])
                drY[i] += ad * ((rRawY[i] - rPrevY[i]) / teSec - drY[i])
            }
            regionRms.fill(0f)
            for (i in 0 until FACE) {
                val p = drX[i] * drX[i] + drY[i] * drY[i]
                for (g in 0 until REGIONS) regionRms[g] += REG[i * REGIONS + g] * p
            }
            // 眨眼时只有眼睑环在动，按整个眼区平均会被周围皮肤稀释、眼睑滞后 1–2 个样本：眼区速度取眼区与眼睑环的较大者
            // （静止时两者都是同一噪声水平，截止频率不变；整个眼区一起提速，不剪切眼睑与上方皮肤之间的三角形）。
            var lidPower = 0f
            for (i in LID_RING) lidPower += drX[i] * drX[i] + drY[i] * drY[i]
            val resDead = K_RES * C_RES * (teMs / TE_REF_MS).pow(RES_TE_EXP) * floor
            for (g in 0 until REGIONS) {
                regionRms[g] = sqrt(regionRms[g] / REG_SUM[g])
                if (g == EYE_REGION) regionRms[g] = max(regionRms[g], sqrt(lidPower / LID_RING.size))
                val excess = max(0f, regionRms[g] - resDead)
                resExcess = max(resExcess, excess)
                regionAlpha[g] = alpha(RES_FC_MIN[g] * comp + RES_BETA[g] * excess, teMs)
            }
            for (i in 0 until FACE) {
                var ai = 0f
                for (g in 0 until REGIONS) ai += REG[i * REGIONS + g] * regionAlpha[g]
                rX[i] += ai * (rRawX[i] - rX[i])
                rY[i] += ai * (rRawY[i] - rY[i])
            }
            rRawX.copyInto(rPrevX); rRawY.copyInto(rPrevY)
        }
        motionSpeed = s * max(max(0f, vp - dead), resExcess)
    }

    /** r = A⁻¹(m − t − Z·w) − L（0..467）。 */
    private fun residual(pose: FloatArray, outRx: FloatArray, outRy: FloatArray) {
        val det = detOf(pose)
        val i11 = pose[5] / det; val i12 = -pose[3] / det
        val i21 = -pose[4] / det; val i22 = pose[2] / det
        for (i in 0 until FACE) {
            val dx = mX[i] - pose[0] - Z[i] * pose[6]
            val dy = mY[i] - pose[1] - Z[i] * pose[7]
            outRx[i] = i11 * dx + i12 * dy - L_X[i]
            outRy[i] = i21 * dx + i22 * dy - L_Y[i]
        }
    }

    /**
     * 虹膜：原始圆心取 5 点均值，换到滤波后眼角坐标系（原点为两眼角中点、x 轴指向 [b]、单位为眼宽）里 One-Euro；
     * 速度死区为自身噪声底的倍数，扫视（一个样本内 0.15 眼宽）时截止频率立刻升高。
     * 半径 = 刚体眼宽 × 低通后的比例：刚体眼宽由滤波后姿态重建两眼角得到（31 个稳定点拟合，几乎不抖，转头时照样按透视缩短），
     * 不用逐帧的眼角距离（它的抖动会原样传给半径）。
     */
    private fun updateIris(eye: Int, base: Int, a: Int, b: Int, teMs: Float, restart: Boolean) {
        var cx = 0f; var cy = 0f
        for (k in 0 until 5) { cx += mX[base + k]; cy += mY[base + k] }
        cx /= 5f; cy /= 5f
        var radius = 0f
        for (k in 1 until 5) radius += hypot(mX[base + k] - cx, mY[base + k] - cy)
        radius /= 4f
        val ox = 0.5f * (outX[a] + outX[b]); val oy = 0.5f * (outY[a] + outY[b])
        val ew = max(hypot(outX[b] - outX[a], outY[b] - outY[a]), 1e-4f)
        val ex = (outX[b] - outX[a]) / ew; val ey = (outY[b] - outY[a]) / ew
        val qx = ((cx - ox) * ex + (cy - oy) * ey) / ew
        val qy = (-(cx - ox) * ey + (cy - oy) * ex) / ew
        val dlx = L_X[b] - L_X[a]; val dly = L_Y[b] - L_Y[a]; val dz = Z[b] - Z[a]
        val rigidEw = max(hypot(q[2] * dlx + q[3] * dly + q[6] * dz, q[4] * dlx + q[5] * dly + q[7] * dz), 1e-4f)
        val kRaw = radius / rigidEw
        if (restart || !(teMs > 0f)) {
            irisQx[eye] = qx; irisQy[eye] = qy; irisRawX[eye] = qx; irisRawY[eye] = qy
            irisDx[eye] = 0f; irisDy[eye] = 0f; irisPower[eye] = -1f; irisMin[eye] = 0f
            irisK[eye] = kRaw.coerceIn(IRIS_K_MIN, IRIS_K_MAX)
        } else {
            val teSec = teMs / 1000f
            val vx = (qx - irisRawX[eye]) / teSec; val vy = (qy - irisRawY[eye]) / teSec
            irisRawX[eye] = qx; irisRawY[eye] = qy
            val power = vx * vx + vy * vy
            if (irisPower[eye] < 0f) { irisPower[eye] = power; irisMin[eye] = power }
            irisPower[eye] += (1f - exp(-teMs / NF_TAU_MS)) * (power - irisPower[eye])
            irisMin[eye] = max(min(irisMin[eye] * (1f + NF_RISE * teSec), irisPower[eye]), 1e-8f)
            val ad = alpha(IRIS_DCUT, teMs)
            irisDx[eye] += ad * (vx - irisDx[eye]); irisDy[eye] += ad * (vy - irisDy[eye])
            // 低通后噪声幅度约为原始噪声 × √(α / (2 − α))。
            val dead = IRIS_K_DEAD * sqrt(irisMin[eye]) * sqrt(ad / (2f - ad))
            val comp = min(1f, sqrt(TE_REF_MS / teMs))
            val fc = IRIS_FC * comp + IRIS_BETA * max(0f, hypot(irisDx[eye], irisDy[eye]) - dead)
            val aq = alpha(fc, teMs)
            irisQx[eye] += aq * (qx - irisQx[eye]); irisQy[eye] += aq * (qy - irisQy[eye])
            irisK[eye] += alpha(IRIS_K_FC, teMs) * (kRaw - irisK[eye])
        }
        val k = irisK[eye].coerceIn(IRIS_K_MIN, IRIS_K_MAX)
        val px = ox + ew * (irisQx[eye] * ex - irisQy[eye] * ey)
        val py = oy + ew * (irisQx[eye] * ey + irisQy[eye] * ex)
        val r = k * rigidEw
        // 与 FaceMeshFrame.synthesizeIris 同一约定：圆心、右、上、左、下（度量空间等距）。
        outX[base] = px; outY[base] = py
        outX[base + 1] = px + r; outY[base + 1] = py
        outX[base + 2] = px; outY[base + 2] = py + r
        outX[base + 3] = px - r; outY[base + 3] = py
        outX[base + 4] = px; outY[base + 4] = py - r
    }

    /**
     * 眨眼门控：滤波后网格的睁眼比 o（睑裂中部 3 对上下睑点距离的平均 / 眼宽，比单对点噪声小 √3）按个人基线归一化
     * （基线首样本初始化，向上 τ = 1 s、向下 τ = 15 s 跟随），经对称 One-Euro 后做施密特迟滞：
     * 低于 [GATE_CLOSE] 立即关门，高于 [GATE_OPEN] 后按采样时间每 [GATE_RAMP_MS] 升到 1。
     */
    private fun updateGate(eye: Int, lids: IntArray, a: Int, b: Int, teMs: Float, restart: Boolean) {
        val ew = max(hypot(outX[b] - outX[a], outY[b] - outY[a]), 1e-4f)
        var gap = 0f
        for (k in lids.indices step 2) gap += hypot(outX[lids[k]] - outX[lids[k + 1]], outY[lids[k]] - outY[lids[k + 1]])
        val open = gap / (lids.size / 2) / ew
        gateOpen[eye] = open
        if (gateBase[eye] < 0f) {
            gateBase[eye] = open.coerceIn(BASE_MIN, BASE_MAX)
        } else if (!restart && teMs > 0f) {
            val tau = if (open > gateBase[eye]) BASE_UP_S else BASE_DOWN_S
            gateBase[eye] = (gateBase[eye] + (open - gateBase[eye]) * (1f - exp(-teMs / 1000f / tau))).coerceIn(BASE_MIN, BASE_MAX)
        }
        val n = open / gateBase[eye]
        if (restart || !(teMs > 0f)) {
            gateX[eye] = n; gatePrev[eye] = n; gateDx[eye] = 0f
            gateClosed[eye] = n < GATE_CLOSE
            gateValue[eye] = if (gateClosed[eye]) 0f else 1f
            return
        }
        gateDx[eye] += alpha(GATE_DCUT, teMs) * ((n - gatePrev[eye]) / (teMs / 1000f) - gateDx[eye])
        gatePrev[eye] = n
        gateX[eye] += alpha(GATE_FC + GATE_BETA * abs(gateDx[eye]), teMs) * (n - gateX[eye])
        if (!gateClosed[eye] && gateX[eye] < GATE_CLOSE) gateClosed[eye] = true
        else if (gateClosed[eye] && gateX[eye] > GATE_OPEN) gateClosed[eye] = false
        gateValue[eye] = if (gateClosed[eye]) 0f else min(1f, gateValue[eye] + teMs / GATE_RAMP_MS)
    }

    companion object {
        const val CONTINUE = 0
        const val NEW_RUN = 1
        /** 姿态向量长度：`[tx, ty, a11, a12, a21, a22, w1, w2]`。 */
        const val POSE_SIZE = 8
        private const val FACE = FaceMeshFrame.FACE_COUNT
        private const val REGIONS = 3
        private const val IRIS_RIGHT = 468
        private const val IRIS_LEFT = 473
        private const val EYE_REGION = 1
        /** 两眼眼睑环（各 16 点）。 */
        private val LID_RING = intArrayOf(
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
        )
        /** 睑裂中部 3 对上下睑点（上、下交替）：解剖右眼 160/144、159/145、158/153，左眼 387/373、386/374、385/380。 */
        private val RIGHT_LIDS = intArrayOf(160, 144, 159, 145, 158, 153)
        private val LEFT_LIDS = intArrayOf(387, 373, 386, 374, 385, 380)

        /** 两个样本相隔超过这么久（丢脸、卡顿）就从头开始（与 FaceTimeline.MAX_GAP_MS 一致）。 */
        const val MAX_GAP_MS = 300L
        /** 速度与最小截止频率的参考采样间隔（30 Hz）。 */
        private const val TE_REF_MS = 1000f / 30f

        // 快层（姿态）One-Euro：静止 2 Hz；刚体速度超过死区（K_POSE × C_POSE × 噪声底）后每 1 脸宽 / 秒加 60 Hz。
        const val POSE_FC_MIN = 2.0f
        const val POSE_BETA = 60f
        const val POSE_DCUT = 5f
        const val K_POSE = 1.8f
        const val C_POSE = 0.25f
        // 慢层（残差）One-Euro：中上脸 / 眼 / 下半脸各一个截止频率；区域速度先按 RES_DCUT 低通。
        val RES_FC_MIN = floatArrayOf(0.9f, 1.6f, 1.6f)
        val RES_BETA = floatArrayOf(30f, 40f, 80f)
        const val RES_DCUT = 1.5f
        const val K_RES = 1.4f
        const val C_RES = 0.20f
        /** 残差死区随采样间隔放大的指数（间隔越长，单样本噪声速度越小）。 */
        const val RES_TE_EXP = 0.85f
        // 噪声底：功率 EMA 的时间常数与最小统计量的上升速度（每秒）。
        const val NF_TAU_MS = 250f
        const val NF_RISE = 0.3f
        // 身份跳变门：中心跳变超过 0.5 脸宽，或尺度对数变化超过 0.35；姿态病态门：|det A| < 0.2 s²。
        const val JUMP_T = 0.5f
        const val JUMP_LOGS = 0.35f
        const val DET_MIN = 0.2f
        // 虹膜：圆心 One-Euro（眼宽 / 秒）与半径比例低通。
        const val IRIS_FC = 1.0f
        const val IRIS_BETA = 10f
        const val IRIS_DCUT = 2f
        /** 虹膜速度死区：低通后噪声幅度的倍数（噪声不把截止频率打开，扫视照常跟上）。 */
        const val IRIS_K_DEAD = 2.0f
        const val IRIS_K_FC = 0.3f
        const val IRIS_K_MIN = 0.16f
        const val IRIS_K_MAX = 0.28f
        // 眨眼门控。
        const val GATE_FC = 1.2f
        const val GATE_BETA = 3.0f
        const val GATE_DCUT = 2f
        const val GATE_CLOSE = 0.40f
        const val GATE_OPEN = 0.55f
        const val GATE_RAMP_MS = 120f
        const val BASE_UP_S = 1f
        const val BASE_DOWN_S = 15f
        const val BASE_MIN = 0.15f
        const val BASE_MAX = 0.45f

        /** 规范脸宽 |p454 − p234|（规范单位）。 */
        val CANONICAL_FACE_WIDTH: Float
        /** 468 点规范坐标 / 规范脸宽，以稳定点加权质心为原点（x 右、y 上；Z 朝向相机）。 */
        private val L_X = FloatArray(FACE)
        private val L_Y = FloatArray(FACE)
        private val Z = FloatArray(FACE)
        /** 外推用的 478 点深度：虹膜取所在眼 4 个锚点的平均深度。 */
        private val Z_EXT = FloatArray(FaceMeshFrame.LANDMARK_COUNT)
        /** 31 个稳定点（去掉随下颌运动的 136/365）与归一化权重。 */
        private val BASIS: IntArray
        private val BASIS_W: FloatArray
        /** 最小二乘常量矩阵 K（8 × 2n，行主序）：q = K · [x_b..., y_b...]。 */
        private val K: FloatArray
        /** 区域软权重（468 × 3：中上脸、双眼、下半脸，每行和为 1）与各区权重和。 */
        private val REG = FloatArray(FACE * REGIONS)
        private val REG_SUM = FloatArray(REGIONS)

        init {
            val xyz = FaceMeshTopology.CANONICAL_XYZ
            val fw = hypot(xyz[454 * 3] - xyz[234 * 3], hypot(xyz[454 * 3 + 1] - xyz[234 * 3 + 1], xyz[454 * 3 + 2] - xyz[234 * 3 + 2]))
            CANONICAL_FACE_WIDTH = fw
            val basis = ArrayList<Int>(); val weights = ArrayList<Double>()
            for (j in FaceMeshTopology.PROCRUSTES_INDICES.indices) {
                val idx = FaceMeshTopology.PROCRUSTES_INDICES[j]
                if (idx == 136 || idx == 365) continue
                basis += idx; weights += FaceMeshTopology.PROCRUSTES_WEIGHTS[j].toDouble()
            }
            val wSum = weights.sum()
            BASIS = basis.toIntArray()
            BASIS_W = FloatArray(basis.size) { (weights[it] / wSum).toFloat() }
            var cx = 0.0; var cy = 0.0; var cz = 0.0
            for (j in BASIS.indices) {
                val b = BASIS[j]; val w = weights[j] / wSum
                cx += w * xyz[b * 3]; cy += w * xyz[b * 3 + 1]; cz += w * xyz[b * 3 + 2]
            }
            for (i in 0 until FACE) {
                L_X[i] = ((xyz[i * 3] - cx) / fw).toFloat()
                L_Y[i] = ((xyz[i * 3 + 1] - cy) / fw).toFloat()
                Z[i] = ((xyz[i * 3 + 2] - cz) / fw).toFloat()
                Z_EXT[i] = Z[i]
            }
            val rightEyeZ = (Z[33] + Z[133] + Z[159] + Z[145]) / 4f
            val leftEyeZ = (Z[362] + Z[263] + Z[386] + Z[374]) / 4f
            for (k in 0 until 5) { Z_EXT[IRIS_RIGHT + k] = rightEyeZ; Z_EXT[IRIS_LEFT + k] = leftEyeZ }
            // 设计矩阵 D（2n × 8）：x 行 [1, 0, Lx, Ly, 0, 0, Z, 0]，y 行 [0, 1, 0, 0, Lx, Ly, 0, Z]；K = (DᵀWD)⁻¹DᵀW。
            val n = BASIS.size
            val rows = 2 * n
            val d = Array(rows) { DoubleArray(POSE_SIZE) }
            val w = DoubleArray(rows)
            for (j in 0 until n) {
                val b = BASIS[j]
                d[j][0] = 1.0; d[j][2] = L_X[b].toDouble(); d[j][3] = L_Y[b].toDouble(); d[j][6] = Z[b].toDouble()
                d[n + j][1] = 1.0; d[n + j][4] = L_X[b].toDouble(); d[n + j][5] = L_Y[b].toDouble(); d[n + j][7] = Z[b].toDouble()
                w[j] = BASIS_W[j].toDouble(); w[n + j] = BASIS_W[j].toDouble()
            }
            val normal = Array(POSE_SIZE) { r -> DoubleArray(POSE_SIZE) { c -> (0 until rows).sumOf { k -> d[k][r] * w[k] * d[k][c] } } }
            val inv = invert(normal)
            K = FloatArray(POSE_SIZE * rows)
            for (r in 0 until POSE_SIZE) for (k in 0 until rows) {
                var v = 0.0
                for (c in 0 until POSE_SIZE) v += inv[r][c] * d[k][c] * w[k]
                K[r * rows + k] = v.toFloat()
            }
            // 区域：双眼（以两眼角中点为心、眼宽为尺度的椭圆，软边）、下半脸（鼻下点 2 以下，扣除眼区）、其余为中上脸。
            val uv = FaceMeshTopology.CANONICAL_UV
            fun smooth(e0: Float, e1: Float, x: Float): Float {
                val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
                return t * t * (3f - 2f * t)
            }
            fun eyeMember(i: Int, a: Int, b: Int): Float {
                val cu = 0.5f * (uv[a * 2] + uv[b * 2]); val cv = 0.5f * (uv[a * 2 + 1] + uv[b * 2 + 1])
                val ew = abs(uv[b * 2] - uv[a * 2])
                val dd = hypot((uv[i * 2] - cu) / ew, (uv[i * 2 + 1] - cv) / (0.75f * ew))
                return 1f - smooth(0.65f, 1.05f, dd)
            }
            val vSub = uv[2 * 2 + 1]
            for (i in 0 until FACE) {
                val eye = max(eyeMember(i, 33, 133), eyeMember(i, 362, 263))
                val low = min(smooth(vSub - 0.01f, vSub + 0.06f, uv[i * 2 + 1]), 1f - eye)
                REG[i * REGIONS] = 1f - eye - low
                REG[i * REGIONS + 1] = eye
                REG[i * REGIONS + 2] = low
                for (g in 0 until REGIONS) REG_SUM[g] += REG[i * REGIONS + g]
            }
        }

        private fun invert(m: Array<DoubleArray>): Array<DoubleArray> {
            val n = m.size
            val a = Array(n) { r -> DoubleArray(2 * n) { c -> if (c < n) m[r][c] else if (c - n == r) 1.0 else 0.0 } }
            for (col in 0 until n) {
                var pivot = col
                for (r in col + 1 until n) if (abs(a[r][col]) > abs(a[pivot][col])) pivot = r
                val tmp = a[col]; a[col] = a[pivot]; a[pivot] = tmp
                val p = a[col][col]
                check(abs(p) > 1e-12) { "procrustes_normal_singular" }
                for (c in 0 until 2 * n) a[col][c] /= p
                for (r in 0 until n) if (r != col) {
                    val f = a[r][col]
                    if (f != 0.0) for (c in 0 until 2 * n) a[r][c] -= f * a[col][c]
                }
            }
            return Array(n) { r -> DoubleArray(n) { c -> a[r][n + c] } }
        }

        /** 弱透视姿态最小二乘：[mx]/[my] 为度量坐标（至少 468 点），结果写入 [out]（[POSE_SIZE]）。 */
        fun fitPose(mx: FloatArray, my: FloatArray, out: FloatArray) {
            val n = BASIS.size
            val rows = 2 * n
            for (r in 0 until POSE_SIZE) {
                var v = 0f
                val base = r * rows
                for (j in 0 until n) {
                    val b = BASIS[j]
                    v += K[base + j] * mx[b] + K[base + n + j] * my[b]
                }
                out[r] = v
            }
        }

        /** det A（带符号；水平镜像点集为负）。 */
        fun detOf(pose: FloatArray): Float = pose[2] * pose[5] - pose[3] * pose[4]

        /** 姿态尺度 √|det A|（度量，正脸时约等于脸宽）。 */
        fun scaleOf(pose: FloatArray): Float = sqrt(abs(detOf(pose)))

        /**
         * 投影尺度平方 |r1 × r2|（r1 = (a11, a12, w1)，r2 = (a21, a22, w2)）：弱透视下等于 s²，与偏航、俯仰无关
         * （|det A| = s²·cos 会随转头缩小）。三角形投影面积 / (它 × 规范 3D 面积) 就是法线与视线夹角的余弦。
         */
        fun viewScale2(pose: FloatArray): Float {
            val cx = pose[3] * pose[7] - pose[6] * pose[5]
            val cy = pose[6] * pose[4] - pose[2] * pose[7]
            val cz = pose[2] * pose[5] - pose[3] * pose[4]
            return sqrt(cx * cx + cy * cy + cz * cz)
        }

        /**
         * 只按刚体姿态外推网格：对每个点求规范局部坐标 u = A⁻¹(m − t − Z·w)，位移 = h·(dt + dA·u + Z·dw)。
         * [src]/[out] 为 478 × 2 纹理坐标（可为同一数组），[poseVel] 为每毫秒的姿态变化，[horizonMs] 已含阻尼。
         */
        fun extrapolate(src: FloatArray, pose: FloatArray, poseVel: FloatArray, horizonMs: Float, aspect: Float, out: FloatArray) {
            val det = detOf(pose)
            if (abs(det) < 1e-10f || !(aspect > 0f)) { if (out !== src) src.copyInto(out, 0, 0, FaceMeshFrame.COORD_COUNT); return }
            val i11 = pose[5] / det; val i12 = -pose[3] / det
            val i21 = -pose[4] / det; val i22 = pose[2] / det
            val h = horizonMs
            for (i in 0 until FaceMeshFrame.LANDMARK_COUNT) {
                val z = Z_EXT[i]
                val dx = src[i * 2] - pose[0] - z * pose[6]
                val dy = src[i * 2 + 1] / aspect - pose[1] - z * pose[7]
                val ux = i11 * dx + i12 * dy; val uy = i21 * dx + i22 * dy
                val mx = h * (poseVel[0] + poseVel[2] * ux + poseVel[3] * uy + z * poseVel[6])
                val my = h * (poseVel[1] + poseVel[4] * ux + poseVel[5] * uy + z * poseVel[7])
                out[i * 2] = src[i * 2] + mx
                out[i * 2 + 1] = src[i * 2 + 1] + my * aspect
            }
        }

        /**
         * 与 [extrapolate] 同一位移场在脸中心（234、454、10、152 四点平均）处的度量位移，写入 [out]（dx, dy/aspect 前的度量值）。
         * 位移场对 u 与 Z 是线性的，四点位移的平均等于平均 u、平均 Z 处的位移。
         */
        fun centerShift(src: FloatArray, pose: FloatArray, poseVel: FloatArray, horizonMs: Float, aspect: Float, out: FloatArray) {
            out[0] = 0f; out[1] = 0f
            val det = detOf(pose)
            if (abs(det) < 1e-10f || !(aspect > 0f)) return
            val i11 = pose[5] / det; val i12 = -pose[3] / det
            val i21 = -pose[4] / det; val i22 = pose[2] / det
            var ux = 0f; var uy = 0f; var z = 0f
            for (i in CENTER_POINTS) {
                val zi = Z_EXT[i]
                val dx = src[i * 2] - pose[0] - zi * pose[6]
                val dy = src[i * 2 + 1] / aspect - pose[1] - zi * pose[7]
                ux += i11 * dx + i12 * dy; uy += i21 * dx + i22 * dy; z += zi
            }
            ux /= CENTER_POINTS.size; uy /= CENTER_POINTS.size; z /= CENTER_POINTS.size
            out[0] = horizonMs * (poseVel[0] + poseVel[2] * ux + poseVel[3] * uy + z * poseVel[6])
            out[1] = horizonMs * (poseVel[1] + poseVel[4] * ux + poseVel[5] * uy + z * poseVel[7])
        }

        private val CENTER_POINTS = intArrayOf(234, 454, 10, 152)

        /** One-Euro 平滑系数（与 FaceTimeline 同一写法）：1 − exp(−2π·fc·te)。 */
        private fun alpha(hz: Float, teMs: Float): Float = (1.0 - exp(-2.0 * PI * hz * teMs / 1000.0)).toFloat()
    }
}
