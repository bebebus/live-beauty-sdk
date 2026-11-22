package io.bebebus.livebeauty

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

class FramePacingTest {
    private val period = 33_333_333L // 30 fps 相机
    private val ms = 1_000_000L
    private val base = 10_000_000_000_000L // 开机约 2.8 小时（elapsedRealtime 纳秒）
    private val sleep = 7_200_000_000_000L // 深睡累计 2 小时：elapsedRealtime − nanoTime

    /** 0..25ms 的确定性「墙钟抖动」。 */
    private fun jitter(k: Int) = ((k * 7919L) % 26L) * ms

    @Test fun uniformContentGetsExactIntervalsDespiteJitteryDrawClock() {
        val clock = PresentationClock()
        val pts = ArrayList<Long>()
        for (k in 0 until 300) {
            val content = base + k * period
            val now = content + 40 * ms + jitter(k)
            val d = clock.decide(content, content, now, now - sleep)
            assertTrue(d.encode)
            pts += d.ptsNs
        }
        assertEquals(PresentationClock.Domain.REALTIME, clock.domain)
        // 100% 落在 30–36.7ms：间隔恒为一个相机周期。
        for (i in 1 until pts.size) assertEquals(period.toDouble(), (pts[i] - pts[i - 1]).toDouble(), 1_000.0)
    }

    @Test fun redrawOfSameContentIsSkippedUntilKeepalive() {
        val clock = PresentationClock()
        val content = base
        val first = clock.decide(content, content, content + 40 * ms, 0L)
        assertTrue(first.encode)
        // ForceRenderer 保活重绘同一内容：不送编码器。
        for (t in longArrayOf(5, 100, 300, 449)) {
            assertFalse(clock.decide(content, content, content + 40 * ms + t * ms, 0L).encode)
        }
        // 距上次送帧满 450ms：补一帧保活，PTS = 上次 + 经过时长。
        val keep = clock.decide(content, content, content + 40 * ms + 500 * ms, 0L)
        assertTrue(keep.encode)
        assertTrue(keep.keepalive)
        assertEquals(first.ptsNs + 500 * ms, keep.ptsNs)
    }

    @Test fun growingTheDelaySkipsOneDrawAndKeepsContentContinuous() {
        val clock = PresentationClock()
        val ring = ArrayList<Long>()
        var delay = 2
        val pts = ArrayList<Long>()
        var skipped = 0
        for (k in 0 until 300) {
            val capture = base + k * period
            ring += capture
            if (k == 150) delay = 3 // FrameDelay 加一帧：内容回退一帧
            val content = ring[ring.size - 1 - minOf(delay, ring.size - 1)]
            val d = clock.decide(content, capture, capture + 40 * ms + jitter(k), 0L)
            if (d.encode) pts += d.ptsNs else skipped++
        }
        assertEquals(1 + 2, skipped) // 开头缓冲攒满前 2 帧 + 加帧那一帧
        for (i in 1 until pts.size) assertEquals(period, pts[i] - pts[i - 1])
    }

    @Test fun realtimeSourceUsesZeroOffset() {
        val clock = PresentationClock()
        clock.declare(true)
        val newest = base
        val d = clock.decide(newest, newest, newest + 40 * ms, newest + 40 * ms - sleep)
        assertEquals(PresentationClock.Domain.REALTIME, clock.domain)
        assertEquals(0L, clock.offsetNs)
        assertEquals(newest, d.ptsNs)
        assertEquals(40 * ms, clock.captureLatencyNs)
    }

    @Test fun unknownSourceMapsMonotonicTimestampsToBoottime() {
        val clock = PresentationClock()
        clock.declare(false)
        val nowBoot = base
        val nowMono = nowBoot - sleep
        val newest = nowMono - 40 * ms // 相机时间戳在 nanoTime 时基
        val d = clock.decide(newest, newest, nowBoot, nowMono)
        assertEquals(PresentationClock.Domain.MONOTONIC, clock.domain)
        assertEquals(sleep, clock.offsetNs)
        assertEquals(nowBoot - 40 * ms, d.ptsNs)
        assertEquals(40 * ms, clock.captureLatencyNs)
    }

    @Test fun wrongDeclarationFallsBackToOtherDomainThenEmpirical() {
        // 声明 REALTIME，实际却是单调时钟：P 不合理，自动改用另一种解释。
        val wrong = PresentationClock()
        wrong.declare(true)
        val nowBoot = base
        val nowMono = nowBoot - sleep
        wrong.decide(nowMono - 30 * ms, nowMono - 30 * ms, nowBoot, nowMono)
        assertEquals(PresentationClock.Domain.MONOTONIC, wrong.domain)
        // 两种解释都不合理（时间戳在未来）：经验锚定，P 记为 50ms。
        val neither = PresentationClock()
        val future = nowBoot + 5_000 * ms
        val d = neither.decide(future, future, nowBoot, nowMono)
        assertEquals(PresentationClock.Domain.EMPIRICAL, neither.domain)
        assertEquals(nowBoot - future - 50 * ms, neither.offsetNs)
        assertEquals(nowBoot - 50 * ms, d.ptsNs)
    }

    @Test fun timebaseJumpReanchorsWithWallClockInterval() {
        val clock = PresentationClock()
        val pts = ArrayList<Long>()
        val nows = ArrayList<Long>()
        for (k in 0 until 10) {
            val now = base + k * period + 5 * ms
            val content = base + k * period + if (k < 5) 0L else 3_600_000 * ms
            val d = clock.decide(content, content, now, 0L)
            assertTrue(d.encode)
            pts += d.ptsNs; nows += now
        }
        assertEquals(1, clock.reanchors)
        for (i in 1 until pts.size) assertEquals(nows[i] - nows[i - 1], pts[i] - pts[i - 1])
    }

    @Test fun unknownContentFallsBackToWallClockMonotonically() {
        val clock = PresentationClock()
        val a = clock.decide(0L, 0L, base, 0L)
        val b = clock.decide(0L, 0L, base, 0L) // 同一时刻两次
        val c = clock.decide(0L, 0L, base - 10 * ms, 0L) // 墙钟回退
        assertEquals(base, a.ptsNs)
        assertEquals(base + PresentationClock.MIN_STEP_NS, b.ptsNs)
        assertEquals(base + 2 * PresentationClock.MIN_STEP_NS, c.ptsNs)
    }

    @Test fun ptsNeverGoBackwardsAfterReanchorOrCameraSwitch() {
        val clock = PresentationClock()
        var last = Long.MIN_VALUE
        var now = base
        var oldContent = 0L
        for (k in 0 until 200) {
            now += period + jitter(k)
            if (k == 100) {
                // 第 100 帧切镜头（生产顺序）：相机重开约 600ms，期间 ForceRenderer 保活重绘旧帧；切换完成后才声明新相机的
                // 时间戳来源，新相机时间戳在 nanoTime 时基（比 elapsedRealtime 小 2 小时）。
                val redraw = now + 500 * ms
                val d = clock.decide(oldContent, oldContent, redraw, redraw - sleep)
                if (d.encode) { assertTrue(d.ptsNs > last); last = d.ptsNs }
                now += 600 * ms
                clock.declare(false)
            }
            val content = if (k < 100) now - 40 * ms else now - sleep - 60 * ms
            if (k < 100) oldContent = content
            val d = clock.decide(content, content, now, now - sleep)
            if (d.encode) { assertTrue(d.ptsNs > last); last = d.ptsNs }
        }
        assertEquals(PresentationClock.Domain.MONOTONIC, clock.domain)
        assertEquals(now - 60 * ms, last) // 新相机的 PTS 即真实采集时刻
    }

    private enum class Ev { DECLARE, INFLIGHT, REDRAW, NEW }

    /** 切镜头之后新相机各帧的 PTS 偏差（PTS − 真实采集时刻，纳秒），以及从旧相机最后一个常规帧起的 PTS 间隔。 */
    private class Switch(val bias: List<Long>, val intervals: List<Long>)

    /**
     * FramePacing 层的切镜头：旧相机 30 fps 出 90 帧后按 [events]（毫秒相对旧相机最后一个常规帧）依次发生——DECLARE 声明新相机
     * 的时间戳来源，INFLIGHT 为旧相机的在途帧，REDRAW 为 ForceRenderer 保活重绘（最新帧不变），NEW 起新相机出 60 帧。
     * 两颗相机的真实时基由 [oldRealtime] / [newRealtime] 给出，[x] = elapsedRealtime − nanoTime（累计深睡），采集延迟 40ms。
     * [skew] / [oldSkew] 为新 / 旧相机时间戳相对其时基的额外偏差（时间戳不合理的设备），[firstLatency] 为新相机第一帧的
     * 采集延迟（切换后首帧出得慢），之后各帧照常 40ms。
     */
    private fun switchCamera(
        x: Long, oldRealtime: Boolean, newRealtime: Boolean, vararg events: Pair<Long, Ev>, skew: Long = 0L, oldSkew: Long = 0L,
        firstLatency: Long = 40 * ms,
    ): Switch {
        val pacing = FramePacing()
        fun stamp(capture: Long, realtime: Boolean) = if (realtime) capture else capture - x
        val pts = ArrayList<Long>()
        val bias = ArrayList<Long>()
        fun draw(content: Long, at: Long): PresentationClock.Decision {
            pacing.publish(content, content); pacing.beginDraw()
            return pacing.decide(content, at, at - x).also { if (it.encode) pts += it.ptsNs }
        }
        pacing.declareTimestampSource(oldRealtime)
        var now = base
        var capture = 0L
        repeat(90) { now += period; capture = now - 40 * ms; draw(stamp(capture, oldRealtime) + oldSkew, now) }
        val lastOld = pts.size - 1
        for ((t, ev) in events) {
            val at = now + t * ms
            when (ev) {
                Ev.DECLARE -> pacing.declareTimestampSource(newRealtime)
                Ev.INFLIGHT -> { capture = at - 40 * ms; draw(stamp(capture, oldRealtime) + oldSkew, at) }
                Ev.REDRAW -> draw(stamp(capture, oldRealtime) + oldSkew, at)
                Ev.NEW -> for (k in 0 until 60) {
                    val frameAt = at + k * period
                    val newCapture = frameAt - if (k == 0) firstLatency else 40 * ms
                    val d = draw(stamp(newCapture, newRealtime) + skew, frameAt)
                    if (d.encode) bias += d.ptsNs - newCapture
                }
            }
        }
        return Switch(bias, pts.subList(lastOld, pts.size).zipWithNext { a, b -> b - a })
    }

    @Test fun sameTimebaseSwitchIsNotResolvedOnAKeepaliveRedraw() {
        // 两颗相机都是 REALTIME（常见设备，X 为小时级）。声明之后相机还没出帧，ForceRenderer 先重绘了一次旧帧（年龄 640ms）：
        // 以前在这一帧上判定，两种解释都不合理而经验锚定，之后整场视频比声音晚约 590ms。
        val result = switchCamera(sleep, true, true, 0L to Ev.DECLARE, 600L to Ev.REDRAW, 700L to Ev.NEW)
        assertEquals(60, result.bias.size)
        assertTrue("偏差 ${result.bias.distinct()}", result.bias.all { it == 0L })
        assertTrue("间隔 ${result.intervals}", result.intervals.all { it >= 30 * ms })
    }

    @Test fun realtimeToMonotonicSwitchHasNoBurstAndNoLastingOffset() {
        // 后摄 REALTIME → 前摄 UNKNOWN（单调时钟），X 小于 1s（开机不久或很少深睡）。生产顺序：在途的旧帧在切换完成前画完，
        // 之后才声明；相机重开期间再有一次保活重绘。以前重绘消费声明，X=900ms 时经验锚定：新相机先跳过若干帧、再按 1ms 连发，
        // 整场偏差约 −320ms。
        for (x in longArrayOf(300 * ms, 900 * ms)) {
            val result = switchCamera(x, true, false, 33L to Ev.INFLIGHT, 400L to Ev.DECLARE, 600L to Ev.REDRAW, 700L to Ev.NEW)
            assertEquals("X=${x / ms}ms", 60, result.bias.size)
            assertTrue("X=${x / ms}ms 偏差 ${result.bias.distinct()}", result.bias.all { abs(it) <= period })
            assertTrue("X=${x / ms}ms 间隔 ${result.intervals}", result.intervals.all { it >= 30 * ms })
        }
    }

    @Test fun declarationWaitsForANewCameraFrame() {
        val clock = PresentationClock()
        val pacing = FramePacing(clock)
        val x = 300 * ms
        pacing.declareTimestampSource(true)
        val old = base
        pacing.publish(old, old); pacing.beginDraw()
        assertTrue(pacing.decide(old, old + 40 * ms, old + 40 * ms - x).encode)
        // 声明新相机为 UNKNOWN（单调时钟）之后，同一最新帧的重绘不消费声明，按重复内容跳过。
        pacing.declareTimestampSource(false)
        pacing.publish(old, old); pacing.beginDraw()
        assertFalse(pacing.decide(old, old + 400 * ms, old + 400 * ms - x).encode)
        assertEquals(PresentationClock.Domain.REALTIME, clock.domain)
        // 新相机首帧（时间戳在 nanoTime 时基）：这时才按声明重新判定，PTS 即真实采集时刻。
        val capture = old + 600 * ms
        pacing.publish(capture - x, capture - x); pacing.beginDraw()
        val d = pacing.decide(capture - x, capture + 40 * ms, capture + 40 * ms - x)
        assertEquals(PresentationClock.Domain.MONOTONIC, clock.domain)
        assertEquals(capture, d.ptsNs)
    }

    @Test fun implausibleReanchorKeepsOffsetUntilAFreshFrame() {
        // 已按 REALTIME 出帧后重新声明，判定的那一帧却已有 700ms（两种解释都不合理）：不经验锚定（会把旧帧的年龄带进偏移，
        // 之后整场错位），保留原偏移照常出这一帧，下一帧再判。
        val clock = PresentationClock()
        clock.declare(true)
        assertEquals(base, clock.decide(base, base, base + 40 * ms, base + 40 * ms - sleep).ptsNs)
        clock.declare(true)
        val stale = base + period
        val d = clock.decide(stale, stale, stale + 700 * ms, stale + 700 * ms - sleep)
        assertEquals(PresentationClock.Domain.REALTIME, clock.domain)
        assertEquals(0L, clock.offsetNs)
        assertEquals(stale, d.ptsNs)
        // 等新相机的帧期间对同一旧帧保活重绘：同一帧的重绘不计入保留上限（否则 2 次重绘就凑满上限、在旧帧上经验锚定），
        // 不送编码器，偏移不变。
        for (t in longArrayOf(720, 760, 800)) {
            val now = stale + t * ms
            assertFalse("重绘 +${t}ms", clock.decide(stale, stale, now, now - sleep).encode)
            assertEquals("重绘 +${t}ms", PresentationClock.Domain.REALTIME, clock.domain)
            assertEquals("重绘 +${t}ms", 0L, clock.offsetNs)
        }
        val fresh = stale + 800 * ms
        assertEquals(fresh, clock.decide(fresh, fresh, fresh + 40 * ms, fresh + 40 * ms - sleep).ptsNs)
        assertEquals(PresentationClock.Domain.REALTIME, clock.domain)
        assertEquals(0, clock.reanchors)
    }

    @Test fun continuityJumpOnALateOldCameraFrameDoesNotEndTheReanchor() {
        // 相机重开期间旧相机的帧迟到（按序到达，内容不倒退），判定落在它上面而保留原偏移；之后这一帧本身（迟到超过 1s）或它的
        // 保活重绘与墙钟的步进相差超过 1s，按墙钟接上。接上之后仍要等新相机的帧判定；就此结束重判的话新相机不再判定，滞后 2s
        // 的新相机整场错位约 −940ms。keepalive：判定前相机已停 500ms、先送过一帧保活（迟到的帧排在它之前而不送，之后的重绘
        // 相对上次送出的内容仍在前进）。
        for ((age, keepalive) in listOf(700L to false, 600L to true, 1_100L to true)) {
            val label = "迟到 ${age}ms 保活 $keepalive"
            val clock = PresentationClock()
            clock.declare(true)
            var now = base
            repeat(30) { now += period; clock.decide(now - 40 * ms, now - 40 * ms, now, now - sleep) }
            val last = now - 40 * ms
            if (keepalive) clock.decide(last, last, now + 500 * ms, now + 500 * ms - sleep)
            clock.declare(true)
            val late = last + period
            val at = late + age * ms
            clock.decide(late, late, at, at - sleep)
            for (t in longArrayOf(500, 1_100)) clock.decide(late, late, at + t * ms, at + t * ms - sleep) // 2fps 保活重绘
            val bias = ArrayList<Long>()
            for (k in 0 until 30) {
                val drawAt = at + 1_260 * ms + k * period
                val capture = drawAt - 40 * ms
                val d = clock.decide(capture - 2_000 * ms, capture - 2_000 * ms, drawAt, drawAt - sleep)
                if (d.encode) bias += d.ptsNs - capture
            }
            assertTrue("$label 出帧 ${bias.size}", bias.size >= 27)
            assertTrue("$label 偏差 ${bias.distinct()}", bias.drop(1).all { abs(it) <= period })
        }
    }

    @Test fun delayedFirstFrameOfNewCameraDoesNotLeaveSessionOffset() {
        // 新相机第一帧采集延迟偏大（切换后首帧出得慢或 GL 线程卡顿 600–1000ms），之后正常 40ms。第一帧两种解释都不合理、按原
        // 偏移像旧帧而保留原偏移；内容与墙钟的步进相差超过 1s，按墙钟接上，接上的偏移里带着第一帧的年龄。接上之后仍要等新相机的
        // 帧判定，就此结束重判的话之后的帧与墙钟的步进一致、连续性也纠正不了，整场留下 L0 − 40ms 的音画错位。
        // 后摄 REALTIME → 前摄 UNKNOWN（单调时钟）：下一帧按 MONOTONIC 合理解释、重新锚定，偏差 0。
        for (x in longArrayOf(sleep, 900 * ms)) for (l0 in longArrayOf(600, 800, 1_000)) {
            val label = "X=${x / ms}ms 首帧延迟 ${l0}ms"
            val result = switchCamera(
                x, true, false, 33L to Ev.INFLIGHT, 400L to Ev.DECLARE, 600L to Ev.REDRAW, 700L to Ev.NEW, firstLatency = l0 * ms,
            )
            assertEquals("$label 出帧", 60, result.bias.size)
            assertTrue("$label 偏差 ${result.bias.drop(1).distinct()}", result.bias.drop(1).all { abs(it) <= period })
            assertTrue("$label 间隔 ${result.intervals}", result.intervals.all { it >= 30 * ms })
        }
        // 新相机时间戳本身也不合理（声明 REALTIME 却滞后 1500ms）：下一帧两种解释仍不合理，按名义 50ms 经验锚定，偏差约 −10ms。
        // 只给可合理解释的帧留改判余地同样会把第一帧的年龄留到整场。名义锚定换掉接上的偏移时的那次间隔是已知代价，这里不断言。
        for (l0 in longArrayOf(600, 1_000)) {
            val result = switchCamera(
                sleep, true, true, 33L to Ev.INFLIGHT, 400L to Ev.DECLARE, 600L to Ev.REDRAW, 700L to Ev.NEW,
                skew = -1_500 * ms, firstLatency = l0 * ms,
            )
            assertEquals("滞后 1500ms 首帧延迟 ${l0}ms 出帧", 60, result.bias.size)
            assertTrue(
                "滞后 1500ms 首帧延迟 ${l0}ms 偏差 ${result.bias.drop(1).distinct()}", result.bias.drop(1).all { abs(it) <= period },
            )
        }
    }

    @Test fun implausibleNewCameraFallsBackToEmpiricalWithinThreeFrames() {
        // 新相机声明 REALTIME，时间戳却比真实采集时刻超前或滞后 700ms：两种解释都不合理。以前一直保留旧相机的偏移，
        // 整场错位 700ms（滞后时还先出一串 1ms 连发）。超前的帧按原偏移落在未来、不可能是旧帧，直接经验锚定；
        // 滞后的帧按原偏移已有 740ms、像旧帧，连续 3 个新相机帧仍不合理才经验锚定，等待期间排在已送 PTS 之前的帧不送。
        // 之后偏差约 −10ms。
        for (skew in longArrayOf(700 * ms, -700 * ms)) {
            val result = switchCamera(
                sleep, true, true, 33L to Ev.INFLIGHT, 400L to Ev.DECLARE, 600L to Ev.REDRAW, 700L to Ev.NEW, skew = skew,
            )
            assertTrue("skew=${skew / ms}ms 出帧 ${result.bias.size}", result.bias.size >= 57)
            assertTrue("skew=${skew / ms}ms 偏差 ${result.bias.distinct()}", result.bias.drop(3).all { abs(it) <= period })
            assertTrue("skew=${skew / ms}ms 间隔 ${result.intervals}", result.intervals.all { it >= 30 * ms })
        }
    }

    @Test fun freshFrameOfEmpiricalOldCameraDoesNotEndTheReanchor() {
        // 旧相机时间戳超前 900ms（两种解释都不合理，经验锚定，偏移按名义 50ms），声明却落在旧相机一个新鲜的在途帧上（违反
        // 「切换完成后才声明」的调用顺序，纵深防御）：这一帧按原偏移约 50ms。以前直接在它上面经验锚定并结束重判，新相机不再判定，
        // 整场错位约 −910ms。现在先保留原偏移：时间戳正常的新相机帧到来后照常判定。新相机同样不合理时：超前 700ms（按原偏移
        // 约 250ms，可能是旧帧）3 帧内（在途旧帧也计入）经验锚定；超前 1500ms（按原偏移落在未来，不是旧帧）直接经验锚定，
        // 保留原偏移的话 PTS 超前墙钟约 590ms，之后 PTS 不能倒退，整场带着这段超前。
        val fine = switchCamera(sleep, true, true, 0L to Ev.DECLARE, 33L to Ev.INFLIGHT, 600L to Ev.NEW, oldSkew = 900 * ms)
        assertEquals("出帧", 60, fine.bias.size)
        assertTrue("偏差 ${fine.bias.distinct()}", fine.bias.all { it == 0L })
        assertTrue("间隔 ${fine.intervals}", fine.intervals.all { it >= 30 * ms })
        for (skew in longArrayOf(700 * ms, 1_500 * ms)) {
            val skewed = switchCamera(
                sleep, true, true, 0L to Ev.DECLARE, 33L to Ev.INFLIGHT, 600L to Ev.NEW, skew = skew, oldSkew = 900 * ms,
            )
            assertEquals("skew=${skew / ms}ms 出帧", 60, skewed.bias.size)
            assertTrue("skew=${skew / ms}ms 偏差 ${skewed.bias.distinct()}", skewed.bias.drop(2).all { abs(it) <= period })
            assertTrue("skew=${skew / ms}ms 间隔 ${skewed.intervals}", skewed.intervals.all { it >= 30 * ms })
        }
    }

    @Test fun contentMappingFollowsPublishAndNeverFallsBackToNewest() {
        val pacing = FramePacing()
        assertEquals(base, pacing.contentFor(base)) // 没有滤镜：显示的就是最新帧
        pacing.publish(base, base - 2 * period)
        assertEquals(base - 2 * period, pacing.contentFor(base))
        // 下一帧没有发布（滤镜提前返回）：沿用同样的延后，不前跳到最新帧。
        assertEquals(base - period, pacing.contentFor(base + period))
    }

    @Test fun unpublishedDrawKeepsPtsCadence() {
        // FrameDelay 延后 2 帧；第 50 帧滤镜提前返回没有发布。
        val pacing = FramePacing()
        val pts = ArrayList<Long>()
        for (k in 0 until 100) {
            val newest = base + k * period
            if (k != 50) pacing.publish(newest, newest - 2 * period)
            pacing.beginDraw()
            val d = pacing.decide(newest, newest + 40 * ms + jitter(k), 0L)
            assertTrue("k=$k", d.encode)
            pts += d.ptsNs
        }
        for (i in 1 until pts.size) assertEquals(period, pts[i] - pts[i - 1])
    }

    @Test fun outputNotUpdatedIsSkippedInsteadOfDuplicated() {
        // 后处理失败：输出 FBO 仍是上一帧内容，发布上一次的内容 → 不送编码器，之后照常。
        val pacing = FramePacing()
        val sent = ArrayList<Long>()
        for (k in 0 until 10) {
            val newest = base + k * period
            pacing.publish(newest, if (k == 5) newest - 3 * period else newest - 2 * period)
            pacing.beginDraw()
            val d = pacing.decide(newest, newest + 40 * ms, 0L)
            assertEquals("k=$k", k != 5, d.encode)
            assertEquals(k != 5, pacing.shouldSwap())
            if (d.encode) sent += d.ptsNs
        }
        assertEquals(2 * period, sent[5] - sent[4]) // 真实少了一帧，不重复
    }

    @Test fun decisionIsSharedWithinOneDraw() {
        val pacing = FramePacing()
        assertTrue(pacing.shouldSwap()) // 没有决策时照常送
        pacing.publish(base, base)
        pacing.beginDraw()
        val encoder = pacing.decide(base, base + 40 * ms, 0L)
        val record = pacing.decide(base, base + 45 * ms, 0L) // 同一次绘制的录制路
        assertSame(encoder, record)
        pacing.beginDraw()
        assertFalse(pacing.decide(base, base + 60 * ms, 0L).encode) // 下一次绘制：同一内容
        assertFalse(pacing.shouldSwap())
    }

    @Test fun audioDelayIsPresetAtStreamStartThenLocksOnFirstEncodedFrame() {
        val pacing = FramePacing()
        // 预览期编码 Surface 未就绪，只发布、从不 decide：没有估计。
        pacing.publish(base, base - 2 * period)
        assertEquals(0L, pacing.audioDelayMs)
        // 开播：按最近一次发布的画面延后预置 D·T + 50ms（麦克风首块可能先于首帧送编码，不再从 0 起滑）。
        val start = base + 3 * period + 50 * ms
        pacing.markStreamStart(start)
        assertEquals((2 * period + PresentationClock.NOMINAL_LATENCY_NS) / ms, pacing.audioDelayMs)
        // 音频编码器以 S 为起点，首个送编码帧锁定 Δ = S − 首帧 PTS。
        val newest = base + 3 * period
        pacing.publish(newest, newest - 2 * period)
        pacing.beginDraw()
        val first = pacing.decide(newest, newest + 40 * ms, 0L)
        assertEquals((start - first.ptsNs) / ms, pacing.audioDelayMs)
        // 之后延后再变也不动。
        for (k in 4 until 10) {
            val n = base + k * period
            pacing.publish(n, n - 4 * period)
            pacing.beginDraw(); pacing.decide(n, n + 40 * ms, 0L)
        }
        assertEquals((start - first.ptsNs) / ms, pacing.audioDelayMs)
        // 上限 300ms；从未发布过（没有滤镜）时预置只有 P。
        val late = FramePacing()
        late.markStreamStart(base + 2_000 * ms)
        assertEquals(PresentationClock.NOMINAL_LATENCY_NS / ms, late.audioDelayMs)
        late.beginDraw(); late.decide(base, base + 40 * ms, 0L)
        assertEquals(300L, late.audioDelayMs)
        // 只录制、未开播（同样经过 decide）：估计为内容的年龄。
        val recording = FramePacing()
        recording.publish(base, base - 2 * period)
        recording.beginDraw(); recording.decide(base, base + 40 * ms, 0L)
        assertEquals((2 * period + 40 * ms) / ms, recording.audioDelayMs)
    }

    @Test fun failedDecisionUsesLegacyPtsAndStaysMonotonic() {
        val pacing = FramePacing()
        val pts = ArrayList<Long>()
        for (k in 0 until 20) {
            val newest = base + k * period
            val now = newest + 40 * ms
            pacing.publish(newest, newest)
            pacing.beginDraw()
            if (k == 10) {
                // 决策出错：本帧按原版绘制时钟送，之后不早于它。
                pacing.fail(now)
                assertTrue(pacing.shouldSwap())
                pts += now
            } else {
                pts += pacing.decide(newest, now, 0L).ptsNs
            }
        }
        for (i in 1 until pts.size) assertTrue(pts[i] > pts[i - 1])
        // 偏移不变：几帧后回到相机周期。
        assertEquals(period, pts[19] - pts[18])
        assertEquals(base + 19 * period, pts[19])
    }

    @Test fun swapTimeIsReadOnce() {
        val pacing = FramePacing()
        pacing.onSwap(3 * ms); pacing.onSwap(2 * ms)
        assertEquals(5 * ms, pacing.takeSwapNs())
        assertEquals(0L, pacing.takeSwapNs())
    }

    @Test fun statsCountAndResetPerWindow() {
        val pacing = FramePacing()
        for (k in 0 until 31) {
            val newest = base + k * period
            pacing.publish(newest, newest)
            pacing.beginDraw(); pacing.decide(newest, newest + 40 * ms, 0L)
            pacing.beginDraw(); pacing.decide(newest, newest + 45 * ms, 0L) // 保活重绘
        }
        val window = pacing.drainStats()
        assertEquals(31, window.encoded)
        assertEquals(31, window.skipped)
        assertEquals(33.3, window.intervalP50, 0.1)
        assertEquals(40.0, window.latencyP95, 0.1)
        assertEquals("REALTIME", window.domain)
        assertEquals(0, pacing.drainStats().encoded)
        assertEquals("effects=off", pacing.effectsSummary())
        pacing.noteEffects(2, 3, true)
        assertEquals("level=2 delay=3 readback=DEFERRED", pacing.effectsSummary())
    }
}
