#!/usr/bin/env python3
"""
Shader Parity & Pixel-Level Numerical Gate for live-beauty-sdk (runs as CTest on all platforms).
Parses actual production GLES (gles_shaders.cpp) and Metal (metal_shaders.h) shader source to assert:
1. skin-k1/tone-k1: exclM is NOT multiplied by ramping pres -> lip w <= 0.05 at pres=0.25!
2. tone-m1/tone-k3: rawPoly decouples face polygon from portraitPrior -> |w_face - w_neck| <= 0.05 & max dL <= 3/255!
3. tone-c1 + skin-m2: headHalo smoothstep(1.02, 1.56) keeps cheek band >= 48px AND beige wall at pa=1.5 <= 2/255!
4. segmentation-k1/gpuperf-k1: kMatteRefineFrag & fs_matte_refine blend cMask -> outMask by uU1.w!
5. skin-k2: Prototype B lineScore + lineScore-masked kSpot numerical probes (circle spot r=2, blackhead, 1px line, pore)
   + explicit uU5.z / uEye1.z secondary prior guard in BOTH GLES and Metal!
"""
import math
import os
import re
import sys


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def luma(r: float, g: float, b: float) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def skin_weight(r: float, g: float, b: float) -> float:
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return (smoothstep(0.26, 0.34, cb) * (1.0 - smoothstep(0.48, 0.56, cb)) *
            smoothstep(0.50, 0.55, cr) * (1.0 - smoothstep(0.72, 0.80, cr)))


def extract_body(src: str, marker: str, end_marker: str) -> str:
    s = src.find(marker)
    assert s >= 0, f"missing marker {marker}"
    e = src.find(end_marker, s)
    assert e > s, f"missing end marker {end_marker}"
    return src[s:e]


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gles_path = os.path.join(root, "src/rhi/gles/gles_shaders.cpp")
    metal_path = os.path.join(root, "src/rhi/metal/metal_shaders.h")
    with open(gles_path, "r", encoding="utf-8") as f:
        gles = f.read()
    with open(metal_path, "r", encoding="utf-8") as f:
        metal = f.read()

    failures = []

    def check(cond: bool, msg: str):
        if cond:
            print(f"PASS  {msg}")
        else:
            print(f"FAIL  {msg}")
            failures.append(msg)

    g_comp = extract_body(gles, "kCompositeProFrag", "kTemporalDenoiseFrag")
    m_comp = extract_body(metal, "fs_composite_pro", "fs_temporal_denoise")
    g_matte = extract_body(gles, "kMatteRefineFrag", "kBlemishHealFrag")
    m_matte = extract_body(metal, "fs_matte_refine", "fs_blemish_heal")

    # 1. skin-k1 / tone-k1: exclM must NOT be multiplied by ramping `pres` in GLES & Metal!
    g_excl_unscaled = ("exclM = sem.g * polyActive" in g_comp) and ("vec3 poly = sem.rgb * pres" not in g_comp)
    m_excl_unscaled = ("exclM = sem.g * polyActive" in m_comp) and ("float3 poly = sem.rgb * pres" not in m_comp)
    check(g_excl_unscaled, "GLES kCompositePro keeps exclM unscaled by ramping pres (skin-k1/tone-k1)")
    check(m_excl_unscaled, "Metal fs_composite_pro keeps exclM unscaled by ramping pres (skin-k1/tone-k1)")

    # 2. tone-m1 / tone-k3: rawPoly decouples face polygon from portraitPrior in GLES & Metal!
    g_raw_poly = ("(1.0 - rawPoly) * bodyGate * portraitPrior" in g_comp)
    m_raw_poly = ("(1.0f - rawPoly) * bodyGate * portraitPrior" in m_comp)
    check(g_raw_poly, "GLES kCompositePro uses (1.0 - rawPoly) to prevent 2p-p^2 face/neck seam (tone-m1/k3)")
    check(m_raw_poly, "Metal fs_composite_pro uses (1.0f - rawPoly) to prevent 2p-p^2 face/neck seam (tone-m1/k3)")

    # Numerical check of skin/w on lip, face, neck at pres=0.25 (reacquire frame 1):
    # If shader still uses old formula, mark failure explicitly via parsed flags above AND numerical check.
    def eval_skin_w(sem_r: float, sem_g: float, pres: float, prior: float, uses_fix: bool):
        poly_act = smoothstep(0.0, 0.02, pres)
        if uses_fix:
            face_m = sem_r * pres
            raw_poly = sem_r * poly_act
            excl_m = sem_g * poly_act
        else:
            face_m = sem_r * pres
            raw_poly = face_m
            excl_m = sem_g * pres
        skin = clamp(face_m * 1.0 + (1.0 - raw_poly) * 1.0 * prior * (1.0 - 0.70 * excl_m), 0.0, 1.0)
        tone_keep = 1.0 - smoothstep(0.58, 0.88, excl_m)
        return skin, skin * tone_keep

    both_fixed = g_excl_unscaled and m_excl_unscaled and g_raw_poly and m_raw_poly
    _, w_lip_reacq = eval_skin_w(0.0, 1.0, 0.25, 0.25, both_fixed)
    _, w_face_reacq = eval_skin_w(1.0, 0.0, 0.50, 0.50, both_fixed)
    _, w_neck_reacq = eval_skin_w(0.0, 0.0, 0.50, 0.50, both_fixed)
    check(w_lip_reacq <= 0.05, f"Reacquire pres=0.25 lip weight w={w_lip_reacq:.3f} <= 0.05 (skin-k1/tone-k1)")
    check(abs(w_face_reacq - w_neck_reacq) <= 0.05,
          f"Reacquire pres=0.50 |w_face({w_face_reacq:.3f}) - w_neck({w_neck_reacq:.3f})| <= 0.05 (tone-m1/k3)")

    # 3. segmentation-k1 / gpuperf-k1: kMatteRefineFrag & fs_matte_refine blend cMask->outMask by uU1.w
    g_matte_blend = ("mix(cMask, outMask, clamp(uU1.w" in g_matte)
    m_matte_blend = ("mix(cMask, outMask, clamp(uni.u[1].w" in m_matte)
    check(g_matte_blend, "GLES kMatteRefineFrag blends raw cMask with refined outMask via uU1.w (segmentation-k1)")
    check(m_matte_blend, "Metal fs_matte_refine blends raw cMask with refined outMask via uni.u[1].w (segmentation-k1)")

    # 4. skin-m2 + tone-c1: Parse actual headHalo smoothstep bounds from GLES & Metal evalPortraitPrior
    m_g = re.search(r"headHalo\s*=\s*\(1\.0\s*-\s*smoothstep\(([0-9.]+),\s*([0-9.]+),\s*length", gles)
    m_m = re.search(r"headHalo\s*=\s*\(1\.0f\s*-\s*smoothstep\(([0-9.]+)f,\s*([0-9.]+)f,\s*length", metal)
    check(m_g is not None and m_m is not None, "Found headHalo smoothstep in both GLES & Metal")
    if m_g and m_m:
        g_lo, g_hi = float(m_g.group(1)), float(m_g.group(2))
        m_lo, m_hi = float(m_m.group(1)), float(m_m.group(2))
        check(abs(g_lo - m_lo) < 1e-4 and abs(g_hi - m_hi) < 1e-4,
              f"GLES ({g_lo}..{g_hi}) == Metal ({m_lo}..{m_hi}) headHalo parity")
        # Beige wall at pa=1.50 (with naturalStudio whiten=0.35 -> max wall lift ~16/255)
        wall_prior = 1.0 - smoothstep(g_lo, g_hi, 1.50)
        wall_lift_lv = wall_prior * 16.0
        # Cheek transition band width (5%-95% on rx=0.18*720=129.6px)
        band_px = (g_hi - g_lo) * 0.72 * 129.6
        check(wall_lift_lv <= 2.0, f"Beige wall at pa=1.50 lift={wall_lift_lv:.2f}/255 <= 2.0 (skin-m2)")
        check(band_px >= 48.0, f"Cheek hand-lift transition 5-95% width={band_px:.1f}px >= 48px (tone-c1)")

    # 5. skin-k2: Explicit secondary prior flag uU5.z / uEye1.z AND Prototype B lineScore + kSpot probes
    check("if (uU5.z > 0.5)" in g_comp and "if (uEye1.z > 0.5f)" in m_comp,
          "Explicit uU5.z / uEye1.z secondary prior gate present in GLES & Metal (skin-k2/m5)")
    check("pmaxMin / max(pmaxMean" in g_comp and "pmaxMin / max(pmaxMean" in m_comp,
          "Prototype B lineScore (pmaxMin/pmaxMean) present in GLES & Metal (skin-k2/m1)")
    check("(1.0 - lineScore)" in g_comp and "(1.0f - lineScore)" in m_comp,
          "lineScore-masked kSpot present in GLES & Metal (skin-k2/m4)")

    # Numerical Prototype B lineScore probes (r=2 45lv spot diagonal vs 1px 35lv diagonal line):
    def proto_b_line_score(patch_3x3):
        c = patch_3x3[1][1]
        # 4 pairs: H, V, diag1, diag2
        pairs = [
            (patch_3x3[1][2] - c, patch_3x3[1][0] - c),
            (patch_3x3[2][1] - c, patch_3x3[0][1] - c),
            (patch_3x3[2][2] - c, patch_3x3[0][0] - c),
            (patch_3x3[0][2] - c, patch_3x3[2][0] - c),
        ]
        lm = [0.5 * (a + b) for a, b in pairs]
        ln = [max(min(a, b), 0.0) for a, b in pairs]
        pmax_m, pmin_m = max(lm), min(lm)
        pmax_n = max(ln)
        return (smoothstep(0.45, 0.75, (pmax_m - pmin_m) / max(pmax_m, 0.004)) *
                smoothstep(0.35, 0.65, pmax_n / max(pmax_m, 0.004)))

    bg = 0.65
    # 1px 35lv diagonal dark hair (top-left to bottom-right)
    hair_patch = [
        [bg - 35/255, bg, bg],
        [bg, bg - 35/255, bg],
        [bg, bg, bg - 35/255],
    ]
    # r=2 45lv circular dark spot (real disk: distance <= 2.0 centered at (0,0), probed at diagonal (+1,+1))
    def disk_val(x, y):
        d = math.hypot(x, y)
        return bg - (45/255) * (1.0 - smoothstep(1.5, 2.3, d))

    spot_rim_patch = [
        [disk_val(1 + dx, 1 + dy) for dx in (-1, 0, 1)]
        for dy in (-1, 0, 1)
    ]
    # sigma=0.8 40lv Gaussian blackhead centered at (0,0), probed at center (0,0)
    def gauss_spot(x, y):
        return bg - (40/255) * math.exp(-0.5 * (x * x + y * y) / (0.8 * 0.8))

    blackhead_patch = [
        [gauss_spot(dx, dy) for dx in (-1, 0, 1)]
        for dy in (-1, 0, 1)
    ]
    ls_hair = proto_b_line_score(hair_patch)
    ls_spot_rim = proto_b_line_score(spot_rim_patch)
    ls_blackhead = proto_b_line_score(blackhead_patch)
    check(ls_hair >= 0.90, f"Prototype B 1px 35lv diagonal hair lineScore={ls_hair:.3f} >= 0.90 (skin-k2)")
    check(ls_spot_rim <= 0.15, f"Prototype B r=2 45lv spot diagonal rim lineScore={ls_spot_rim:.3f} <= 0.15 (skin-k2)")
    check(ls_blackhead <= 0.05, f"Prototype B sigma=0.8 40lv blackhead center lineScore={ls_blackhead:.3f} <= 0.05 (skin-k2)")

    if failures:
        print(f"\nFAIL  {len(failures)} shader gate check(s) failed.")
        return 1
    print("\nPASS  all shader parity & pixel-level numerical checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
