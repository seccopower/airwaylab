"""Analisi della sezione bronchiale: lume (half-max) + parete (FWHM per settore).

Per ogni punto di misura: ricentraggio sul lume, N raggi nel piano
perpendicolare, bordo interno del lume al half-max, e per ogni settore il
bordo esterno della parete (attraversamento a meta' altezza tra picco di
parete e fondo parenchimale). I settori dove la parete si fonde con vasi o
mediastino non hanno discesa oltre il picco e vengono marcati NON validi:
la trasparenza su questa selezione e' parte del metodo.
"""
import os

import numpy as np
from scipy import ndimage

N_ANGLES = 64
STEP_MM = 0.25

# Finestra stretta per rami piccoli (maschere DL generose): i raggi non
# scavalcano la parete sottile agganciando strutture lontane. Attiva con
# AIRWAYLAB_TIGHT_SMALL=1 (il backend DL della CLI la attiva da solo).
TIGHT_SMALL = os.environ.get('AIRWAYLAB_TIGHT_SMALL') == '1'

def perp_basis(t):
    ref = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(t, ref); u /= np.linalg.norm(u)
    v = np.cross(t, u)
    return u, v

def pca_tangent(path, i, w=5):
    """Tangente stabile: primo autovettore dei punti in una finestra locale."""
    a, b = max(0, i - w), min(len(path), i + w + 1)
    P = path[a:b] - path[a:b].mean(axis=0)
    if len(P) < 3:
        d = path[min(i + 1, len(path) - 1)] - path[max(0, i - 1)]
        n = np.linalg.norm(d)
        return d / n if n > 0 else np.array([0, 0, 1.0])
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    t = vt[0]
    ref = path[min(i + 2, len(path) - 1)] - path[max(0, i - 2)]
    if np.dot(t, ref) < 0:
        t = -t
    return t

def analyze_section(ct, center, t, r_est_mm, iso):
    """Analisi completa della sezione. Ritorna un dict o None.

    Chiavi: inner (radii mm, N_ANGLES), outer (mm, nan dove non valido),
    d_eq (mm), wall_med (mm o None), wall_valid_frac, u, v,
    shift (du,dv mm nel piano), quality {ax_ratio, good_frac}.
    """
    u, v = perp_basis(t)
    if r_est_mm < 0.6:
        return None
    tight = TIGHT_SMALL and r_est_mm < 2.5
    if tight:
        rmax_mm = float(np.clip(r_est_mm * 2.0 + 1.5, 3.0, 7.0))
    else:
        rmax_mm = float(np.clip(r_est_mm * 2.5 + 6.5, 7, 34))
    ns = int(rmax_mm / STEP_MM)
    radii_ax = np.arange(ns) * STEP_MM

    # --- ricentraggio sul baricentro dell'aria connessa ---
    rc = float(np.clip(r_est_mm * 1.5 + 1.5, 2.0, 8.0))
    nc = int(rc / STEP_MM)
    axc = np.arange(-nc, nc + 1) * (STEP_MM / iso)
    Uc, Vc = np.meshgrid(axc, axc)
    cc = center[None, :] + Uc.reshape(-1, 1) * u[None, :] + Vc.reshape(-1, 1) * v[None, :]
    pc = ndimage.map_coordinates(ct, cc.T, order=1, mode='nearest').reshape(Uc.shape)
    air = pc < -650
    if not air.any():
        return None
    lab, _ = ndimage.label(air)
    ys, xs = np.nonzero(air)
    k = np.argmin((ys - nc) ** 2 + (xs - nc) ** 2)
    comp = lab == lab[ys[k], xs[k]]
    cy, cx = ndimage.center_of_mass(comp)
    du, dv = (cx - nc) * STEP_MM, (cy - nc) * STEP_MM
    shift_u, shift_v = 0.0, 0.0
    if np.hypot(du, dv) < rc:
        center = center + (du / iso) * u + (dv / iso) * v
        shift_u, shift_v = du, dv
    lumen_lvl = float(np.percentile(pc[comp], 15))
    if lumen_lvl > -400:
        return None

    # --- profili radiali ---
    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    dirs = (np.cos(angles)[:, None] * u[None, :] +
            np.sin(angles)[:, None] * v[None, :])
    coords = center[None, None, :] + (radii_ax[None, :, None] / iso) * dirs[:, None, :]
    prof = ndimage.map_coordinates(ct, coords.reshape(-1, 3).T, order=1,
                                   mode='nearest').reshape(N_ANGLES, ns)

    inner = np.full(N_ANGLES, np.nan)
    outer = np.full(N_ANGLES, np.nan)
    thick = np.full(N_ANGLES, np.nan)
    # --- passata 1: solo bordo interno del lume ---
    j_in_all = np.full(N_ANGLES, -1, dtype=int)
    for a in range(N_ANGLES):
        p = prof[a]
        j0 = max(1, int(max(0.3, r_est_mm * 0.5) / STEP_MM))
        pk = j0 + int(np.argmax(p[j0:]))
        peak = p[pk]
        min_jump = 250 if r_est_mm >= 1.6 else 120
        if peak - lumen_lvl < min_jump:
            continue
        half_in = lumen_lvl + 0.5 * (peak - lumen_lvl)
        for j in range(1, pk + 1):
            if p[j] >= half_in:
                f = (half_in - p[j - 1]) / max(1e-3, p[j] - p[j - 1])
                inner[a] = radii_ax[j - 1] + f * STEP_MM
                j_in_all[a] = j
                break

    good = np.isfinite(inner)
    if good.sum() < N_ANGLES * 0.5:
        return None
    d_eq_rough = 2 * float(np.nanmedian(inner))

    # --- passata 2: parete. La finestra di ricerca e' un limite FISICO fisso
    # (5 mm): il tetto fisiologico (~18% del calibro) NON censura piu' la
    # misura, ma diventa un flag QC (frazione di settori oltre il tetto),
    # cosi' l'ispessimento patologico resta misurabile e dichiarato.
    wall_cap_mm = float(np.clip(0.18 * d_eq_rough + 0.6, 1.0, 3.2))
    Wcap = int(5.0 / STEP_MM)
    W4 = int(4.0 / STEP_MM)
    for a in range(N_ANGLES):
        j_in = j_in_all[a]
        if j_in < 0:
            continue
        p = prof[a]
        # picco di parete cercato SOLO entro il tetto fisiologico
        seg = p[j_in:j_in + Wcap]
        if len(seg) < 2:
            continue
        pk2 = j_in + int(np.argmax(seg))
        peak2 = p[pk2]
        tail = p[pk2:pk2 + W4]
        if len(tail) < 4:
            continue
        outer_bg = float(np.percentile(tail, 20))
        if peak2 - outer_bg < 150:
            continue                     # nessuna discesa: vaso/mediastino a contatto
        half_out = outer_bg + 0.5 * (peak2 - outer_bg)
        # la discesa a meta' altezza deve avvenire ENTRO il tetto dal bordo interno
        j_max = min(ns - 1, j_in + Wcap)
        Wpar = int(1.5 / STEP_MM)
        for j in range(pk2, j_max):
            if p[j + 1] < half_out <= p[j]:
                f = (p[j] - half_out) / max(1e-3, p[j] - p[j + 1])
                r_out = radii_ax[j] + f * STEP_MM
                th = r_out - inner[a]
                # requisito positivo: oltre la parete deve esserci PARENCHIMA
                # aerato, non mediastino ne' vaso
                win = p[j + 1:j + 1 + Wpar]
                is_parenchyma = len(win) >= 2 and float(np.median(win)) < -650
                if 0.2 < th < 5.0 and is_parenchyma:
                    outer[a] = r_out
                    thick[a] = th
                break

    # --- filtro di coerenza circolare: lo spessore non salta tra settori ---
    tmed_all = float(np.nanmedian(thick)) if np.isfinite(thick).any() else None
    if tmed_all is not None:
        for a in range(N_ANGLES):
            if not np.isfinite(thick[a]):
                continue
            w = [thick[(a + k) % N_ANGLES] for k in range(-3, 4) if k != 0]
            w = [x for x in w if np.isfinite(x)]
            local = float(np.median(w)) if len(w) >= 2 else tmed_all
            if abs(thick[a] - local) > max(0.6, 0.5 * local):
                thick[a] = np.nan
                outer[a] = np.nan

    good = np.isfinite(inner)
    if good.sum() < N_ANGLES * 0.5:
        return None
    med = float(np.nanmedian(inner))
    inner = np.clip(inner, 0.3, med * 2.2)
    idx = np.arange(N_ANGLES)
    inner[~good] = np.interp(idx[~good], idx[good], inner[good], period=N_ANGLES)
    ker = np.array([0.25, 0.5, 0.25])
    inner = np.convolve(np.r_[inner[-1], inner, inner[0]], ker, mode='valid')

    area = 0.5 * np.sum(inner ** 2) * (2 * np.pi / N_ANGLES)
    d_eq = 2 * np.sqrt(area / np.pi)
    d_cap = (2 * r_est_mm * 1.4 + 1.0) if tight else (2 * r_est_mm * 1.8 + 2.0)
    if d_eq > d_cap:
        return None

    angs2 = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    xs_, ys_ = inner * np.cos(angs2), inner * np.sin(angs2)
    cov = np.cov(np.stack([xs_ - xs_.mean(), ys_ - ys_.mean()]))
    ev = np.linalg.eigvalsh(cov)
    ax_ratio = float(np.sqrt(max(ev) / max(1e-6, min(ev))))

    wall_ok = np.isfinite(thick)
    wall_med = float(np.nanmedian(thick)) if wall_ok.sum() >= N_ANGLES * 0.25 else None
    # flag QC: frazione dei settori validi con spessore oltre il tetto
    # fisiologico (possibile patologia O contaminazione residua da vaso)
    over = float((thick[wall_ok] > wall_cap_mm).mean()) if wall_ok.any() else 0.0

    return {
        'inner': inner, 'outer': outer, 'd_eq': float(d_eq),
        'wall_med': wall_med, 'wall_valid_frac': float(wall_ok.mean()),
        'wall_over_cap_frac': over, 'wall_cap_mm': round(wall_cap_mm, 2),
        'u': u, 'v': v, 'shift': (shift_u, shift_v),
        'quality': {'ax_ratio': ax_ratio, 'good_frac': float(good.mean())},
    }

def lumen_polygon(ct, center, t, r_est_mm, iso):
    """Wrapper di compatibilita': (radii, d_eq, u, v, shift, quality)."""
    s = analyze_section(ct, center, t, r_est_mm, iso)
    if s is None:
        u, v = perp_basis(t)
        return None, None, u, v, (0.0, 0.0), None
    return s['inner'], s['d_eq'], s['u'], s['v'], s['shift'], s['quality']
