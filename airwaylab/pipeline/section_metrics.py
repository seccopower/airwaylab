"""Metriche della sezione di MASCHERA sul piano perpendicolare al centerline.

Step 1 (solo audit, gate invariato): fornisce le grandezze OMOGENEE con la
misura half-max, per confrontare correttamente calibro CT e calibro suggerito
dalla maschera su vie eccentriche (trachea a sciabola, bronchi deformati).

  d_mask_eq   = 2*sqrt(area_maschera/pi)   diametro AREA-equivalente (per il
               confronto col d_eq half-max: grandezze omologhe)
  d_mask_min  = minor Feret nel piano      dimensione MINIMA (per il resolution
               floor: un lume 4x1 mm ha d_eq 2 mm ma resta non risolto a 1 mm)
  d_mask_maj  = max Feret nel piano
  mask_radii  = distanza radiale alla parete della maschera per ciascun angolo
               (per distinguere underfill globale da fuga half-max localizzata)

Puro (numpy + scipy.ndimage): nessun I/O, testato in tests/.
Il campionamento della maschera e' trilineare (order=1) con soglia 0.5 — niente
nearest-neighbor, che darebbe aree a scalini e bias dipendente dall'orientamento.
"""
import numpy as np
from scipy import ndimage

from lumen import N_ANGLES, perp_basis


def _plane_sample(maskf, center, u, v, iso, half_mm, step_mm):
    """Campiona la maschera (float) su una griglia 2D fisica isotropa nel piano
    (u, v) centrata su `center`. Ritorna (occupancy_bool, n2) dove n2 e' l'indice
    del pixel centrale."""
    n2 = int(half_mm / step_mm)
    ax = np.arange(-n2, n2 + 1) * (step_mm / iso)
    U, V = np.meshgrid(ax, ax)
    coords = (center[None, :] + U.reshape(-1, 1) * u[None, :]
              + V.reshape(-1, 1) * v[None, :])
    plane = ndimage.map_coordinates(maskf, coords.T, order=1, mode='constant')
    return (plane.reshape(U.shape) > 0.5), n2


def _feret(comp, step_mm, n_orient=30):
    """Min e max Feret (larghezza proiettata) della componente booleana, in mm.
    Proiezione dei pixel su n_orient orientazioni in [0, pi)."""
    ys, xs = np.nonzero(comp)
    if len(xs) == 0:
        return None, None
    x = xs.astype(float)
    y = ys.astype(float)
    widths = []
    for th in np.linspace(0.0, np.pi, n_orient, endpoint=False):
        proj = x * np.cos(th) + y * np.sin(th)
        widths.append((proj.max() - proj.min() + 1.0) * step_mm)  # +1 px di spessore
    return float(min(widths)), float(max(widths))


def mask_section(maskf, center, t, iso, r_est_mm, step_mm=0.4):
    """Metriche della sezione di maschera perpendicolare a `t` in `center`.

    Ritorna dict {area_mm2, d_eq, d_min, d_maj} sulla componente connessa che
    contiene il centro (evita di gonfiare l'area con una figlia o una regione
    adiacente), oppure None se il centro non cade su maschera."""
    u, v = perp_basis(t)
    half = min(20.0, max(6.0, r_est_mm * 3 + 3))
    occ, n2 = _plane_sample(maskf, center, u, v, iso, half, step_mm)
    lab, nlab = ndimage.label(occ)
    if nlab == 0:
        return None
    c = lab[n2, n2]
    if c == 0:                                  # centro sulla parete per interp.
        ys, xs = np.nonzero(lab)
        k = int(np.argmin((ys - n2) ** 2 + (xs - n2) ** 2))
        c = lab[ys[k], xs[k]]
    comp = (lab == c)
    area = float(comp.sum()) * step_mm ** 2
    d_min, d_maj = _feret(comp, step_mm)
    return {'area_mm2': area, 'd_eq': 2.0 * np.sqrt(area / np.pi),
            'd_min': d_min, 'd_maj': d_maj, 'u': u, 'v': v}


def mask_ray_radii(maskf, center, u, v, iso, rmax_mm, step_mm=0.25,
                   n_angles=N_ANGLES):
    """Distanza radiale (mm) dal centro alla parete della maschera, per ciascun
    angolo del piano (u, v). Usa gli stessi angoli del campionamento half-max,
    cosi' i raggi CT e maschera sono confrontabili angolo per angolo."""
    ns = max(2, int(rmax_mm / step_mm))
    radii_ax = np.arange(ns) * step_mm
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    dirs = (np.cos(angles)[:, None] * u[None, :]
            + np.sin(angles)[:, None] * v[None, :])
    coords = (center[None, None, :]
              + (radii_ax[None, :, None] / iso) * dirs[:, None, :])
    prof = ndimage.map_coordinates(maskf, coords.reshape(-1, 3).T, order=1,
                                   mode='constant').reshape(n_angles, ns)
    out = np.full(n_angles, rmax_mm)
    for a in range(n_angles):
        below = np.nonzero(prof[a] < 0.5)[0]
        if len(below):
            j = below[0]
            out[a] = radii_ax[j]      # primo attraversamento verso l'esterno
    return out


def overshoot_fraction(ct_inner_mm, mask_radii_mm, margin_mm=0.3):
    """Frazione di angoli in cui il bordo half-max CT sta OLTRE la parete della
    maschera (ct > mask + margine). Alta e diffusa -> underfill globale della
    maschera; bassa ma con outlier estremi -> fuga half-max localizzata."""
    ct = np.asarray(ct_inner_mm, dtype=float)
    mk = np.asarray(mask_radii_mm, dtype=float)
    ok = np.isfinite(ct) & np.isfinite(mk)
    if not ok.any():
        return 0.0
    return float((ct[ok] > mk[ok] + margin_mm).mean())
