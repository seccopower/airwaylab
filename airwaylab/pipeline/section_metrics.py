"""Metriche della sezione di MASCHERA sul piano perpendicolare al centerline.

Step 1 (solo audit, gate invariato): grandezze OMOGENEE con la misura half-max,
per confrontare correttamente calibro CT e calibro suggerito dalla maschera su
vie eccentriche (trachea a sciabola, bronchi deformati) e per raccogliere un
dataset NON censurato con cui ricalibrare il gate (Step 3).

Ogni sezione porta il proprio STATUS, così un audit non viene mai presentato
come pulito quando non lo è:
  d_eq         2*sqrt(area/pi)            AREA-equivalente (omologo al half-max)
  d_min/d_maj  Feret minimo/massimo       larghezza minima (resolution floor) e asse maggiore
  center_in_mask                          il centro cade dentro la maschera?
  touches_border                          la componente tocca il bordo della ROI (area sottostimata)
  n_components                            componenti nel piano (audit; NON usato per validare)
  solidity                                area/convex-hull della componente scelta (bassa = bilobata)
  valid_mask_section                      center_in_mask AND not touches_border
  mask_radii   distanza radiale alla parete, interpolata; NaN se nessun attraversamento

Le biforcazioni NON si censurano con soglie morfologiche non validate: si
esportano gli ingredienti grezzi (solidity, distance_to_junction_mm,
local_area_ratio) e si escludono dalle SINTESI solo le sezioni topologicamente
prossime alla giunzione. Un futuro `suspected_bifurcation` combinera' bassa
solidity e aumento locale d'area DOPO calibrazione annotata.

Puro (numpy + scipy.ndimage). Campionamento trilineare (order=1) + soglia 0.5:
niente nearest-neighbor (aree a scalini, bias d'orientamento).
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image

from lumen import N_ANGLES, perp_basis

MASK_LEVEL = 0.5


def _plane_sample(maskf, center, u, v, iso, half_mm, step_mm):
    """Campiona la maschera (float) su una griglia 2D fisica isotropa nel piano
    (u, v) centrata su `center`. Ritorna (occupancy_bool, n2)."""
    n2 = int(half_mm / step_mm)
    ax = np.arange(-n2, n2 + 1) * (step_mm / iso)
    U, V = np.meshgrid(ax, ax)
    coords = (center[None, :] + U.reshape(-1, 1) * u[None, :]
              + V.reshape(-1, 1) * v[None, :])
    plane = ndimage.map_coordinates(maskf, coords.T, order=1, mode='constant')
    return (plane.reshape(U.shape) > MASK_LEVEL), n2


def _feret(comp, step_mm, n_orient=90):
    """Min e max Feret (larghezza proiettata) della componente booleana, in mm,
    su n_orient orientazioni in [0, pi). Audit-grade (contorno a scalini): per
    uso DECISIONALE preferire SDF + marching squares + rotating calipers."""
    ys, xs = np.nonzero(comp)
    if len(xs) == 0:
        return None, None
    x = xs.astype(float)
    y = ys.astype(float)
    th = np.linspace(0.0, np.pi, n_orient, endpoint=False)
    # proiezioni: (n_orient, n_pixel)
    proj = np.outer(np.cos(th), x) + np.outer(np.sin(th), y)
    widths = (proj.max(axis=1) - proj.min(axis=1) + 1.0) * step_mm
    return float(widths.min()), float(widths.max())


def _touches_border(comp):
    return bool(comp[0, :].any() or comp[-1, :].any()
                or comp[:, 0].any() or comp[:, -1].any())


def mask_section(maskf, center, t, iso, r_est_mm, step_mm=0.4,
                 expand=True, nearest_tol_mm=None):
    """Metriche della sezione di maschera perpendicolare a `t` in `center`.

    Seleziona la componente connessa che CONTIENE il centro. Se il centro cade
    su background, ripiega sulla componente più vicina SOLO entro una tolleranza
    fisica (`nearest_tol_mm`, default r_est) e lo dichiara (`center_in_mask=False`).
    Con `expand` la ROI si allarga se la componente tocca il bordo (fino a 40 mm).

    Ritorna sempre un dict con lo status. `valid_mask_section` = centro nella
    maschera AND componente non troncata (NON usa n_components: una componente
    estranea distante non deve invalidare un lume centrale misurabile). Il
    numero di componenti resta come audit; la biforcazione va rilevata con
    distanza dalla giunzione e/o `solidity` (bassa su una sezione bilobata
    connessa), non con n_components==1."""
    u, v = perp_basis(t)
    if nearest_tol_mm is None:
        nearest_tol_mm = max(2.0, r_est_mm)
    half = min(20.0, max(6.0, r_est_mm * 3 + 3))

    def _invalid(nlab, cim):
        return {'valid_mask_section': False, 'center_in_mask': cim,
                'n_components': int(nlab), 'touches_border': False,
                'solidity': None, 'centroid_offset_mm': None,
                'area_mm2': 0.0, 'd_eq': None, 'd_min': None, 'd_maj': None,
                'aspect': None, 'u': u, 'v': v, 'center_used': center}

    comp = None
    n2 = 0
    center_in_mask = False
    nlab = 0
    for _ in range(4):
        occ, n2 = _plane_sample(maskf, center, u, v, iso, half, step_mm)
        lab, nlab = ndimage.label(occ)
        if nlab == 0:
            return _invalid(0, False)
        cc = lab[n2, n2]
        center_in_mask = cc > 0
        if not center_in_mask:
            ys, xs = np.nonzero(lab)
            k = int(np.argmin((ys - n2) ** 2 + (xs - n2) ** 2))
            dist_mm = np.hypot(ys[k] - n2, xs[k] - n2) * step_mm
            if dist_mm > nearest_tol_mm:
                return _invalid(nlab, False)
            cc = lab[ys[k], xs[k]]
        comp = (lab == cc)
        touches = _touches_border(comp)
        if touches and expand and half < 40.0:
            half = min(40.0, half * 1.6)
            continue
        break

    area = float(comp.sum()) * step_mm ** 2
    d_min, d_maj = _feret(comp, step_mm)
    aspect = (d_min / d_maj) if (d_min and d_maj) else None
    try:
        hull = convex_hull_image(comp)
        solidity = float(comp.sum()) / float(max(1, hull.sum()))
    except Exception:
        solidity = None
    cy, cx = ndimage.center_of_mass(comp)
    centroid_offset_mm = float(np.hypot(cy - n2, cx - n2) * step_mm)
    center_used = (center + (cx - n2) * (step_mm / iso) * u
                   + (cy - n2) * (step_mm / iso) * v)
    return {
        'area_mm2': area, 'd_eq': 2.0 * np.sqrt(area / np.pi),
        'd_min': d_min, 'd_maj': d_maj, 'aspect': aspect,
        'solidity': solidity, 'centroid_offset_mm': centroid_offset_mm,
        'center_in_mask': bool(center_in_mask), 'n_components': int(nlab),
        'touches_border': bool(touches),
        # NB: n_components NON entra qui (audit-only)
        'valid_mask_section': bool(center_in_mask and not touches),
        'u': u, 'v': v, 'center_used': center_used,
    }


def mask_ray_radii(maskf, center, u, v, iso, rmax_mm, step_mm=0.25,
                   n_angles=N_ANGLES):
    """Distanza radiale (mm) dal centro alla parete della maschera per ciascun
    angolo. Il bordo è INTERPOLATO all'attraversamento del livello 0.5; se un
    raggio non attraversa entro rmax ritorna NaN (censura), non rmax (che
    sarebbe un falso raggio reale)."""
    ns = max(2, int(rmax_mm / step_mm))
    radii_ax = np.arange(ns) * step_mm
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    dirs = (np.cos(angles)[:, None] * u[None, :]
            + np.sin(angles)[:, None] * v[None, :])
    coords = (center[None, None, :]
              + (radii_ax[None, :, None] / iso) * dirs[:, None, :])
    prof = ndimage.map_coordinates(maskf, coords.reshape(-1, 3).T, order=1,
                                   mode='constant').reshape(n_angles, ns)
    out = np.full(n_angles, np.nan)
    for a in range(n_angles):
        below = np.nonzero(prof[a] < MASK_LEVEL)[0]
        if not len(below):
            continue                      # nessun attraversamento -> NaN
        j = below[0]
        if j == 0:
            out[a] = 0.0                   # centro già fuori maschera
            continue
        p0, p1 = prof[a, j - 1], prof[a, j]
        f = (p0 - MASK_LEVEL) / max(1e-6, p0 - p1)
        out[a] = radii_ax[j - 1] + f * step_mm
    return out


def radial_delta_stats(ct_inner_mm, mask_radii_mm, spacing_mm=None,
                       margins=(0.3, 0.5, 1.0)):
    """Statistiche del delta radiale signed (CT half-max − maschera) angolo per
    angolo. Distingue underfill globale (delta positivo diffuso e piccolo) da
    fuga localizzata (pochi angoli con delta molto grande, contigui).

    Ritorna dict con mediana/P90/P95/max del delta, frazioni oltre le soglie in
    `margins`, frazione di raggi validi, no_crossing_frac e la massima corsa di
    angoli contigui in overshoot (> min(margins))."""
    ct = np.asarray(ct_inner_mm, dtype=float)
    mk = np.asarray(mask_radii_mm, dtype=float)
    n = len(ct)
    no_cross = float(np.isnan(mk).mean()) if n else 0.0
    ok = np.isfinite(ct) & np.isfinite(mk)
    frac_valid = float(ok.mean()) if n else 0.0
    if not ok.any():
        return {'delta_med': None, 'delta_p90': None, 'delta_p95': None,
                'delta_max': None, 'frac_over': {str(m): None for m in margins},
                'frac_valid': frac_valid, 'no_crossing_frac': no_cross,
                'max_contig_overshoot': 0}
    d = ct[ok] - mk[ok]
    over = {str(m): float((d > m).mean()) for m in margins}
    # massima corsa contigua (sull'anello completo) di angoli in overshoot
    thr = min(margins)
    flag = np.zeros(n, dtype=bool)
    flag[np.nonzero(ok)[0]] = d > thr
    best = cur = 0
    for k in range(2 * n):                 # anello: due giri per catturare il wrap
        if flag[k % n]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    best = min(best, n)
    if spacing_mm:
        dn = d / spacing_mm
    else:
        dn = d
    return {'delta_med': float(np.median(d)),
            'delta_p90': float(np.percentile(d, 90)),
            'delta_p95': float(np.percentile(d, 95)),
            'delta_max': float(np.max(d)),
            'delta_med_norm': float(np.median(dn)),
            'frac_over': over, 'frac_valid': frac_valid,
            'no_crossing_frac': no_cross, 'max_contig_overshoot': int(best)}


def overshoot_fraction(ct_inner_mm, mask_radii_mm, margin_mm=0.3):
    """Frazione di angoli con bordo half-max CT oltre la maschera (+margine).
    Parametrico nel margine; i NaN (no_crossing) sono censurati, non contati."""
    ct = np.asarray(ct_inner_mm, dtype=float)
    mk = np.asarray(mask_radii_mm, dtype=float)
    ok = np.isfinite(ct) & np.isfinite(mk)
    if not ok.any():
        return 0.0
    return float((ct[ok] > mk[ok] + margin_mm).mean())


# --- schema fisso del record di sezione (JSON versionato, valori null) --------
SCHEMA_VERSION = 1
SECTION_KEYS = (
    'branch_id', 'aid', 'gen', 'i', 's_mm', 'distance_to_junction_mm',
    'ct_available', 'ax_ratio', 'ct_reportable', 'd_eq_ct',
    'd_mask_eq', 'd_mask_min', 'd_mask_maj', 'aspect', 'solidity',
    'local_diameter_ratio', 'center_in_mask', 'component_centroid',
    'centroid_offset_mm', 'touches_border', 'n_components',
    'near_junction', 'valid_mask_section', 'radial',
)


def blank_section_record():
    """Record di sezione con TUTTE le chiavi presenti a null: lo schema resta
    uniforme anche per le sezioni senza lume CT (niente chiavi mancanti)."""
    return {k: None for k in SECTION_KEYS}


def _med(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 3) if vals else None


RADIAL_MIN_VALID = 0.5     # frazione minima di raggi validi per usare l'overshoot


def branch_mask_summary(records):
    """Aggrega le metriche di ramo separando popolazioni OMOLOGHE:
      - `mask_valid`   : sezioni valide e NON giunzionali -> morfometria di
                         maschera. NESSUNA censura morfologica (no soglie
                         d'area non validate): l'esclusione e' solo topologica.
      - `paired_diam`  : mask_valid AND CT reportabile con entrambi i diametri
                         -> ct_mask_ratio (mediana dei RAPPORTI per-sezione).
      - `paired_radial`: paired_diam AND confronto radiale con abbastanza raggi
                         validi (radial.frac_valid >= RADIAL_MIN_VALID)
                         -> overshoot_frac (rimedio: non contare radial vuoti)."""
    mask_valid = [r for r in records if r.get('valid_mask_section')
                  and not r.get('near_junction')]
    paired_diam = [r for r in mask_valid
                   if r.get('ct_available') and r.get('ct_reportable')
                   and r.get('d_eq_ct') and r.get('d_mask_eq')]
    paired_radial = [r for r in paired_diam
                     if r.get('radial') is not None
                     and r['radial'].get('frac_valid', 0.0) >= RADIAL_MIN_VALID
                     and r['radial'].get('frac_over', {}).get('0.5') is not None]
    ratios = [r['d_eq_ct'] / r['d_mask_eq'] for r in paired_diam]
    overs = [r['radial']['frac_over']['0.5'] for r in paired_radial]
    return {
        'd_mask_eq': _med([r['d_mask_eq'] for r in mask_valid]),
        'd_mask_min': _med([r['d_mask_min'] for r in mask_valid]),
        'd_mask_maj': _med([r['d_mask_maj'] for r in mask_valid]),
        'aspect_mask': _med([r['aspect'] for r in mask_valid]),
        'ct_mask_ratio': round(float(np.median(ratios)), 3) if ratios else None,
        'overshoot_frac': round(float(np.median(overs)), 3) if overs else None,
        'n_sez_mask_valide': len(mask_valid),
        'n_sez_paired_diam': len(paired_diam),
        'n_sez_paired_radial': len(paired_radial),
    }
