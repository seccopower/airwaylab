"""Tests for the refined centerline (refine.py logic) and the air witness
gates (witness.py logic) on synthetic data — no patient data required.

refine.py and witness.py are workdir scripts, so their core behaviours are
tested here through the same primitives they use: recentering must pull a
zig-zag path back to the tube axis and the B-spline must de-inflate the
staircase length; the witness thresholds must accept an air-filled tube and
reject a soft-tissue rod.
"""
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.interpolate import splev, splprep

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from lumen import pca_tangent, perp_basis  # noqa: E402

ISO = 0.7


def make_tube_mask(lumen_d_mm=6.0, shape=(80, 60, 60)):
    """Straight axial air tube mask along z."""
    yy, xx = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    yc, xc = shape[1] // 2, shape[2] // 2
    r_mm = np.hypot((yy - yc) * ISO, (xx - xc) * ISO)
    mask = np.zeros(shape, dtype=np.float32)
    mask[:, r_mm <= lumen_d_mm / 2] = 1.0
    return mask, yc, xc


def zigzag_path(shape, yc, xc, n=60, amp=1.5):
    """Staircase path along the tube axis, oscillating around the true axis."""
    zs = np.linspace(5, shape[0] - 6, n)
    off = amp * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return np.stack([zs, yc + off, xc + 0.0 * zs], axis=1)


def _recenter(P, mask, radii_mm):
    Q = P.copy()
    for i in range(1, len(P) - 1):
        t = pca_tangent(P, i)
        u, v = perp_basis(t)
        r = float(np.clip(radii_mm[i] * 1.5, 1.5, 4.0)) / ISO
        ax = np.arange(-r, r + 1e-6, 0.5)
        U, V = np.meshgrid(ax, ax)
        grid = P[i][None, :] + U.reshape(-1, 1) * u[None, :] + V.reshape(-1, 1) * v[None, :]
        w = ndimage.map_coordinates(mask, grid.T, order=1, mode="constant", cval=0.0)
        if w.sum() < 1e-3:
            continue
        cu = float((w * U.reshape(-1)).sum() / w.sum())
        cv = float((w * V.reshape(-1)).sum() / w.sum())
        Q[i] = P[i] + cu * u + cv * v
    return Q


def test_recentering_pulls_to_axis_and_spline_deflates_length():
    mask, yc, xc = make_tube_mask()
    P = zigzag_path(mask.shape, yc, xc)
    radii = np.full(len(P), 3.0)
    Q = _recenter(_recenter(P, mask, radii), mask, radii)
    # lateral error vs the true axis must shrink a lot
    err_raw = np.abs(P[1:-1, 1] - yc).mean()
    err_ref = np.abs(Q[1:-1, 1] - yc).mean()
    assert err_ref < 0.35 * err_raw, f"recentering weak: {err_raw:.2f}->{err_ref:.2f}"
    # spline length must approach the chord (staircase de-inflation)
    wgt = np.ones(len(Q))
    wgt[0] = wgt[-1] = 50.0
    tck, _ = splprep(Q.T, k=3, s=len(Q) * 0.25, w=wgt)
    S = np.array(splev(np.linspace(0, 1, 80), tck)).T
    chord = np.linalg.norm(P[-1] - P[0]) * ISO
    len_raw = np.linalg.norm(np.diff(P, axis=0) * ISO, axis=1).sum()
    len_spl = np.linalg.norm(np.diff(S, axis=0) * ISO, axis=1).sum()
    assert len_raw > 1.05 * chord            # the zig-zag really inflates
    assert len_spl < 1.02 * chord, f"spline {len_spl:.1f} vs chord {chord:.1f}"
    # junction endpoints preserved
    assert np.allclose(S[0], P[0], atol=1e-6) or np.linalg.norm(S[0] - P[0]) < 0.6


def test_air_witness_production_gate_and_borderlines():
    """Exercise the SAME function and constants the pipeline uses (no copied
    thresholds), including the borderline cases around both criteria."""
    from qc_params import air_witness, HU_AIR, AIR_FRAC

    air = np.full(50, -980.0) + np.random.RandomState(0).normal(0, 20, 50)
    rod = np.full(50, -350.0) + np.random.RandomState(1).normal(0, 40, 50)
    assert air_witness(air) is True
    assert air_witness(rod) is False

    # borderline sulla mediana: -751 passa, -749 no (frazione ampiamente ok)
    assert air_witness(np.full(100, HU_AIR - 1.0)) is True
    assert air_witness(np.full(100, HU_AIR + 1.0)) is False

    # borderline sulla frazione: mediana buona, aria al 59% / 61%
    n = 100
    for frac, expect in ((0.59, False), (0.61, True)):
        k = int(round(frac * n))
        hu = np.concatenate([np.full(k, -900.0), np.full(n - k, -550.0)])
        # forza la mediana sotto soglia solo quando la maggioranza e' aria
        if k > n // 2:
            assert air_witness(hu) is expect
        else:
            assert air_witness(hu) is False   # mediana sopra soglia: boccia

    # lume parzialmente occupato (bimodale): meta' aria, meta' muco denso
    half = np.concatenate([np.full(50, -950.0), np.full(50, -100.0)])
    assert air_witness(half) is False
    assert air_witness(np.array([])) is False


def test_contour_escape_gate_is_pure_and_has_exact_boundary():
    """Il gate `contour_escaped` e' un puro test geometrico (nessuna logica di
    aid): d_mean > K*d_mask + C. Verifica il bordo ESATTO e appena sopra."""
    from qc_params import ESCAPE_C, ESCAPE_K, contour_escaped

    d_mask = 8.0
    thr = ESCAPE_K * d_mask + ESCAPE_C          # soglia esatta
    assert contour_escaped(thr, d_mask) is False        # uguale: NON sfugge (>)
    assert contour_escaped(thr + 1e-6, d_mask) is True  # appena sopra: sfugge
    assert contour_escaped(thr - 1e-6, d_mask) is False
    assert contour_escaped(None, d_mask) is False
    assert contour_escaped(5.0, None) is False


def test_central_aids_come_from_the_anatomy_contract():
    """CENTRAL_AIDS non deve duplicare una lista: e' DERIVATO dalla sorgente
    anatomica normativa (anatomy.ALR4_AIDS), cosi' le due non divergono."""
    import anatomy
    from qc_params import CENTRAL_AIDS
    assert CENTRAL_AIDS is anatomy.ALR4_AIDS or CENTRAL_AIDS == anatomy.ALR4_AIDS
    assert {'TRACHEA', 'RMB', 'LMB', 'BI'} <= set(CENTRAL_AIDS)


def test_escape_decision_exempts_central_but_flags_it():
    """Policy del chiamante (witness.py), estratta in funzione pura testabile.
    Quando la maschera DL sotto-riempie una grande via centrale, la half-max
    corretta 'sfugge': l'endpoint va PRESERVATO ma con soft-flag di audit, mai
    reso indistinguibile da una misura concordante con la maschera. In periferia
    resta hard-demotion."""
    from qc_params import (CENTRAL_AIDS, ESCAPE_EXEMPT_NOTE, escape_decision)

    d_mask, d_true = 11.0, 18.0                 # trachea sotto-riempita ~18 mm
    # via centrale: NON demolita, ma flaggata
    for aid in CENTRAL_AIDS:
        demote, note = escape_decision(d_true, d_mask, aid)
        assert demote is False and note == ESCAPE_EXEMPT_NOTE
    # via periferica reale (aid dal contratto anatomico) con stesso overshoot:
    # demolita, nessun flag
    from anatomy import to_aid
    periph = to_aid('B10 dx')                   # -> 'B10_R'
    assert periph and periph not in CENTRAL_AIDS
    assert escape_decision(d_true, d_mask, periph) == (True, '')
    assert escape_decision(d_true, d_mask, None) == (True, '')
    # misura concordante con la maschera: gate non scatta -> nessun flag,
    # nemmeno sulle vie centrali (un 'ok' centrale resta distinguibile)
    assert escape_decision(12.0, 12.5, 'TRACHEA') == (False, '')
    assert escape_decision(3.0, 2.6, periph) == (False, '')


def test_resolution_floor_conservative_and_orientation_dependent():
    """The conservative floor uses the WORST native axis (slice thickness
    included); the per-branch floor projects the native spacings into the
    section plane, so in-plane branches feel the slice direction."""
    from qc_params import orientation_floor_mm, resolution_floor_mm

    native = (0.7246, 0.7246, 1.25)          # xyz, anisotropo
    # fallback conservativo: comanda l'asse peggiore (1.25), upsampling inerte
    assert abs(resolution_floor_mm(0.5, native) - 3 * 1.25) < 1e-9
    assert abs(resolution_floor_mm(1.4, native) - 3 * 1.4) < 1e-9
    # ramo assiale (t lungo z, ordine zyx): piano di sezione = in-plane 0.72
    assert abs(orientation_floor_mm(0.5, native, (1, 0, 0)) - 3 * 0.7246) < 1e-6
    # ramo in-plane (t lungo y): la direzione fetta 1.25 entra nel piano
    assert abs(orientation_floor_mm(0.5, native, (0, 1, 0)) - 3 * 1.25) < 1e-6
    # ramo obliquo 45 gradi z-y: la direzione PEGGIORE del piano mescola gli
    # assi — worst-plane = sqrt(u'Mu) con u=(sin45,-cos45,0) = 1.0217 mm,
    # NON la proiezione per-asse max(s_i*sin) = 0.884 (sottostima ~13%)
    t45 = (1 / np.sqrt(2), 1 / np.sqrt(2), 0)
    worst_plane = np.sqrt(0.5 * (1.25 ** 2 + 0.7246 ** 2))
    got = orientation_floor_mm(0.5, native, t45)
    assert abs(got - 3 * worst_plane) < 1e-6, got
    assert got > 3 * 1.25 * np.sqrt(0.5) + 0.1   # rifiuta la formula per-asse
    # senza info native: fallback sul passo di processing
    assert abs(orientation_floor_mm(0.7, None, (1, 0, 0)) - 2.1) < 1e-9
    # il floor orientato non scende mai sotto quello del processing ISO
    assert orientation_floor_mm(0.9, native, (1, 0, 0)) >= 3 * 0.9 - 1e-9


def test_branch_floor_uses_local_tangents_not_the_chord():
    """A branch curving from axial to in-plane must be floored by its worst
    LOCAL orientation (the in-plane end), even when its global chord at 45
    degrees would give a lower floor."""
    from qc_params import branch_floor_mm, orientation_floor_mm

    native = (0.7246, 0.7246, 1.25)
    R = 30.0
    th = np.linspace(0, np.pi / 2, 40)
    arc = np.stack([R * np.cos(th), R * np.sin(th), np.zeros_like(th)], axis=1)
    got = branch_floor_mm(0.5, native, arc)
    # il tratto finale corre lungo y (in-plane): floor pieno da 1.25 mm
    assert abs(got - 3 * 1.25) < 0.02, got
    # la corda globale (45 gradi) avrebbe dato meno: il chord NON basta
    chord = orientation_floor_mm(0.5, native, arc[-1] - arc[0])
    assert got > chord + 0.2
    # ramo dritto assiale: resta al floor in-plane fine
    line = np.stack([np.linspace(0, 30, 20), np.zeros(20), np.zeros(20)], axis=1)
    assert abs(branch_floor_mm(0.5, native, line) - 3 * 0.7246) < 1e-6


def test_split_reportable_invariant_and_csv_cardinality():
    """UNIT-LEVEL invariant: split_reportable() (the single function through
    which CPR routes every sample) can only put a value on one channel; CSV
    schema and generated row both have 24 fields. This does NOT execute the
    pipeline: artifact-level verification (profiles.json/cpr.json of a full
    run) was done manually on the development case; a synthetic end-to-end
    phantom test is tracked in issue #2."""
    from qc_params import CSV_COLUMNS, csv_row, split_reportable

    for v in (4.2, None, 0.0):
        for rep in (True, False):
            clin, raw = split_reportable(v, rep)
            if rep:
                assert clin == v and raw is None
            else:
                assert clin is None and raw == v

    assert len(CSV_COLUMNS) == 28            # +qc_note +4 metriche di maschera
    for col in ('qc_note', 'd_maschera_eq_mm', 'd_maschera_min_mm',
                'ct_mask_ratio', 'overshoot_frac'):
        assert col in CSV_COLUMNS
    dummy = {'id': 'br000', 'name': 'trachea', 'gen': 0, 'length': 100.0}
    assert len(csv_row(dummy)) == len(CSV_COLUMNS)
    # una via centrale esente porta il soft-flag nella riga CSV
    dummy2 = {'id': 'br001', 'name': 'trachea', 'gen': 0, 'length': 100.0,
              'qc': 'ok', 'qc_note': 'central-mask-underfill'}
    assert 'central-mask-underfill' in csv_row(dummy2)


def test_provenance_keys_written_and_exposed():
    """SOURCE-LEVEL guard, not an end-to-end test: asserts the provenance
    key set is stable and that cli.py writes / viz.py exposes those keys.
    It does not generate or read a real map_data.json (that requires a full
    pipeline run; done manually on the development case, synthetic
    end-to-end coverage tracked in issue #2)."""
    from qc_params import PROVENANCE_KEYS

    assert set(PROVENANCE_KEYS) == {'backend', 'refined_centerline',
                                    'tight_small_window', 'airwaylab_version'}
    root = os.path.join(os.path.dirname(__file__), '..', 'airwaylab')
    cli_src = open(os.path.join(root, 'cli.py')).read()
    viz_src = open(os.path.join(root, 'pipeline', 'viz.py')).read()
    for k in ('backend', 'refined_centerline', 'tight_small_window',
              'airwaylab_version'):
        assert f'"{k}"' in cli_src or f"'{k}'" in cli_src, k
    assert 'PROVENANCE_KEYS' in viz_src


def test_curved_phantom_arc_length():
    """Refined length must approach the TRUE ARC length of a curved tube —
    not the chord (which on a quarter circle is ~10% shorter than the arc)."""
    from scipy.interpolate import splev, splprep

    R, r_tube = 22.0, 4.0                      # voxels
    shape = (40, 64, 64)
    z0, y0, xc = 6.0, 6.0, shape[2] // 2
    zz, yy, xx = np.meshgrid(*(np.arange(s) for s in shape), indexing='ij')
    rad = np.sqrt((zz - z0) ** 2 + (yy - y0) ** 2)
    dist = np.sqrt((rad - R) ** 2 + (xx - xc) ** 2)
    mask = ((dist < r_tube) & (zz >= z0) & (yy >= y0)).astype(np.float32)

    thetas = np.linspace(0.05, np.pi / 2 - 0.05, 40)
    true_pts = np.stack([z0 + R * np.sin(thetas), y0 + R * np.cos(thetas),
                         np.full_like(thetas, float(xc))], axis=1)
    # input = percorso arrotondato alla griglia (staircase)
    P = np.round(true_pts)
    radii = np.full(len(P), r_tube * ISO)
    Q = _recenter(_recenter(P, mask, radii), mask, radii)
    wgt = np.ones(len(Q)); wgt[0] = wgt[-1] = 50.0
    tck, _ = splprep(Q.T, k=3, s=len(Q) * 0.25, w=wgt)
    S = np.array(splev(np.linspace(0, 1, 120), tck)).T

    arc_true = R * (thetas[-1] - thetas[0])
    chord = np.linalg.norm(true_pts[-1] - true_pts[0])
    len_spl = np.linalg.norm(np.diff(S, axis=0), axis=1).sum()
    assert chord < 0.95 * arc_true             # il chord NON e' la verita'
    err = abs(len_spl - arc_true) / arc_true
    assert err < 0.04, f'arc {arc_true:.1f} vs spline {len_spl:.1f} ({err:.1%})'
