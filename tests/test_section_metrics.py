"""Test del modulo puro section_metrics (Step 1: metriche di maschera sulla
sezione perpendicolare). Nessun dato paziente: fantocci sintetici."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from section_metrics import (mask_ray_radii, mask_section,   # noqa: E402
                             overshoot_fraction, radial_delta_stats)

T_Z = np.array([1.0, 0.0, 0.0])   # tangente lungo z (array zyx) -> piano (x, y)


def _cylinder(shape_zyx, iso, cy_mm, cx_mm, a_mm, b_mm, phi=0.0):
    """Cilindro lungo z con sezione ellittica (semiassi a,b in mm su x,y),
    eventualmente RUOTATA di phi nel piano."""
    Nz, Ny, Nx = shape_zyx
    yy, xx = np.mgrid[0:Ny, 0:Nx].astype(float)
    dx = (xx - cx_mm / iso) * iso
    dy = (yy - cy_mm / iso) * iso
    xr = dx * np.cos(phi) + dy * np.sin(phi)
    yr = -dx * np.sin(phi) + dy * np.cos(phi)
    ell = ((xr / a_mm) ** 2 + (yr / b_mm) ** 2 <= 1.0)
    return np.repeat(ell[None], Nz, axis=0).astype(np.float32)


def _oblique_cylinder(shape_zyx, iso, center_vox, axis_zyx, R_mm):
    """Cilindro con asse ARBITRARIO (obliquo rispetto alla griglia)."""
    Nz, Ny, Nx = shape_zyx
    zz, yy, xx = np.mgrid[0:Nz, 0:Ny, 0:Nx].astype(float)
    a = np.array(axis_zyx, float)
    a /= np.linalg.norm(a)
    vz, vy, vx = zz - center_vox[0], yy - center_vox[1], xx - center_vox[2]
    dot = vz * a[0] + vy * a[1] + vx * a[2]
    px, py, pz = vx - dot * a[2], vy - dot * a[1], vz - dot * a[0]
    perp = np.sqrt(px ** 2 + py ** 2 + pz ** 2) * iso
    return (perp <= R_mm).astype(np.float32)


def test_circular_section_area_equivalent_and_feret():
    """Sezione circolare: d_eq, d_min e d_maj coincidono col diametro vero."""
    iso = 0.3
    D = 16.0
    m = _cylinder((5, 140, 140), iso, cy_mm=21.0, cx_mm=21.0, a_mm=D / 2, b_mm=D / 2)
    center = np.array([2.0, 21.0 / iso, 21.0 / iso])
    s = mask_section(m, center, T_Z, iso, r_est_mm=D / 2)
    assert abs(s['d_eq'] - D) < 1.0
    assert abs(s['d_min'] - D) < 1.5 and abs(s['d_maj'] - D) < 1.5


def test_elliptical_section_matches_axes_and_area_eq():
    """Sezione ellittica 26x12 mm (trachea a sciabola sintetica): il diametro
    AREA-equivalente e' sqrt(26*12)=17.66 mm, ben diverso dalla dimensione
    minima 12 mm — la distinzione al centro dello Step 1."""
    iso = 0.3
    maj, mnr = 26.0, 12.0
    m = _cylinder((5, 200, 200), iso, cy_mm=30.0, cx_mm=30.0,
                  a_mm=maj / 2, b_mm=mnr / 2)
    center = np.array([2.0, 30.0 / iso, 30.0 / iso])
    s = mask_section(m, center, T_Z, iso, r_est_mm=maj / 2)
    assert abs(s['d_eq'] - np.sqrt(maj * mnr)) < 1.0     # ~17.66
    assert abs(s['d_min'] - mnr) < 1.5                   # ~12
    assert abs(s['d_maj'] - maj) < 1.5                   # ~26
    # il d_eq NON deve essere scambiato per la dimensione minima
    assert s['d_eq'] - s['d_min'] > 3.0


def test_connected_component_contains_center():
    """Due lumi vicini: l'area conta SOLO la componente che contiene il centro,
    non la somma (evita di gonfiare con una figlia o struttura adiacente)."""
    iso = 0.3
    D = 10.0
    a = _cylinder((5, 200, 200), iso, cy_mm=30.0, cx_mm=22.0, a_mm=D / 2, b_mm=D / 2)
    b = _cylinder((5, 200, 200), iso, cy_mm=30.0, cx_mm=40.0, a_mm=D / 2, b_mm=D / 2)
    m = np.clip(a + b, 0, 1)
    center = np.array([2.0, 30.0 / iso, 22.0 / iso])     # dentro il primo lume
    s = mask_section(m, center, T_Z, iso, r_est_mm=D / 2)
    assert abs(s['d_eq'] - D) < 1.2                      # misura SOLO la componente centrale
    assert s['n_components'] == 2                        # audit: due componenti
    # una componente estranea DISTANTE non invalida un lume centrale misurabile
    # (rimedio review #2: n_components non entra in valid_mask_section)
    assert s['valid_mask_section'] is True
    assert s['solidity'] > 0.9                           # cerchio: quasi convesso


def test_mask_ray_radii_on_circle():
    """I raggi alla parete della maschera su un cerchio sono ~ il raggio vero."""
    iso = 0.3
    R = 8.0
    m = _cylinder((5, 160, 160), iso, cy_mm=24.0, cx_mm=24.0, a_mm=R, b_mm=R)
    from section_metrics import perp_basis
    u, v = perp_basis(T_Z)
    center = np.array([2.0, 24.0 / iso, 24.0 / iso])
    rr = mask_ray_radii(m, center, u, v, iso, rmax_mm=16.0)
    assert abs(float(np.median(rr)) - R) < 0.5


def test_overshoot_fraction_discriminates():
    """Underfill globale: CT piu' larga in tutte le direzioni -> frazione ~1.
    Concordante: nessun overshoot -> 0. Fuga localizzata: pochi angoli."""
    n = 64
    ct = np.full(n, 10.0); mk = np.full(n, 8.0)
    assert overshoot_fraction(ct, mk) == 1.0
    assert overshoot_fraction(np.full(n, 6.0), mk) == 0.0
    loc = np.full(n, 6.0); loc[:4] = 20.0                # 4/64 angoli in fuga
    assert abs(overshoot_fraction(loc, mk) - 4.0 / n) < 1e-9


def test_oblique_cylinder_area_invariant():
    """Cilindro con asse obliquo: la sezione perpendicolare al vero asse deve
    dare comunque d_eq ~ diametro (invarianza all'obliquita')."""
    iso = 0.3
    D = 12.0
    axis = np.array([1.0, 0.4, 0.3])
    ctr = np.array([40.0, 60.0, 60.0])
    m = _oblique_cylinder((80, 120, 120), iso, ctr, axis, R_mm=D / 2)
    s = mask_section(m, ctr, axis, iso, r_est_mm=D / 2)
    assert abs(s['d_eq'] - D) < 1.2


def test_in_plane_rotation_invariance_of_feret():
    """Ellisse ruotata nel piano: min/max Feret restano ~ minore/maggiore veri,
    indipendenti dall'orientamento (a differenza di assi coronale/sagittale)."""
    iso = 0.3
    maj, mnr = 24.0, 10.0
    for phi in (0.0, np.pi / 6, np.pi / 3, np.pi / 2):
        m = _cylinder((5, 200, 200), iso, 30.0, 30.0, maj / 2, mnr / 2, phi=phi)
        c = np.array([2.0, 30.0 / iso, 30.0 / iso])
        s = mask_section(m, c, T_Z, iso, r_est_mm=maj / 2)
        assert abs(s['d_min'] - mnr) < 1.6, (phi, s['d_min'])
        assert abs(s['d_maj'] - maj) < 1.6, (phi, s['d_maj'])


def test_center_outside_mask_flagged():
    """Centro appena fuori dal lume: center_in_mask False e sezione non valida
    (fallback alla componente vicina solo entro tolleranza, dichiarato)."""
    iso = 0.3
    D = 8.0
    m = _cylinder((5, 160, 160), iso, 24.0, 24.0, D / 2, D / 2)
    off = np.array([2.0, 24.0 / iso, (24.0 + D / 2 + 1.5) / iso])   # 1.5 mm fuori
    s = mask_section(m, off, T_Z, iso, r_est_mm=D / 2)
    assert s['center_in_mask'] is False
    assert s['valid_mask_section'] is False
    # oltre la tolleranza fisica: nessuna componente accettata
    far = np.array([2.0, 24.0 / iso, (24.0 + 40.0) / iso])
    s2 = mask_section(m, far, T_Z, iso, r_est_mm=D / 2, nearest_tol_mm=3.0)
    assert s2['valid_mask_section'] is False and s2['d_eq'] is None


def test_touches_border_flagged_without_expand():
    """Con ROI piccola e senza espansione, una componente troncata dal bordo
    viene segnalata (area/Feret sottostimati) e la sezione non e' valida."""
    iso = 0.3
    D = 30.0
    m = _cylinder((5, 200, 200), iso, 30.0, 30.0, D / 2, D / 2)
    c = np.array([2.0, 30.0 / iso, 30.0 / iso])
    s = mask_section(m, c, T_Z, iso, r_est_mm=2.0, expand=False)   # half piccolo
    assert s['touches_border'] is True
    assert s['valid_mask_section'] is False


def test_ray_no_crossing_is_nan():
    """Maschera piena: nessun raggio attraversa la parete entro rmax -> NaN
    (censura), MAI rmax spacciato per raggio reale."""
    m = np.ones((5, 60, 60), np.float32)
    from section_metrics import perp_basis
    u, v = perp_basis(T_Z)
    rr = mask_ray_radii(m, np.array([2.0, 30.0, 30.0]), u, v, 0.3, rmax_mm=6.0)
    assert np.isnan(rr).all()


def test_overshoot_fraction_exact_margin_boundary():
    """Uguaglianza esatta al margine: ct == mask + margine NON conta come fuga
    (serve strettamente maggiore)."""
    mk = np.full(16, 8.0)
    ct = mk + 0.3
    assert overshoot_fraction(ct, mk, margin_mm=0.3) == 0.0
    assert overshoot_fraction(ct + 1e-6, mk, margin_mm=0.3) == 1.0


def test_d_eq_convergence_over_step():
    """Il d_eq della maschera converge al ridurre step_mm (0.4/0.2/0.1)."""
    iso = 0.2
    D = 16.0
    m = _cylinder((5, 200, 200), iso, 24.0, 24.0, D / 2, D / 2)
    c = np.array([2.0, 24.0 / iso, 24.0 / iso])
    vals = [mask_section(m, c, T_Z, iso, D / 2, step_mm=s)['d_eq']
            for s in (0.4, 0.2, 0.1)]
    assert all(abs(v - D) < 1.0 for v in vals)
    assert abs(vals[-1] - vals[0]) < 0.6            # stabile tra passi


def test_radial_delta_stats_distinguishes_underfill_from_leak():
    """Le statistiche signed distinguono underfill diffuso da fuga localizzata,
    contano i NaN come no_crossing e misurano la corsa contigua in overshoot."""
    n = 64
    # underfill diffuso: piccolo delta positivo ovunque
    ct = np.full(n, 8.4); mk = np.full(n, 8.0)
    r = radial_delta_stats(ct, mk, spacing_mm=0.7)
    assert r['frac_over']['0.3'] == 1.0 and r['frac_over']['1.0'] == 0.0
    assert r['max_contig_overshoot'] == n
    # fuga localizzata: 3 angoli contigui molto oltre, resto concordante
    ct2 = np.full(n, 6.0); ct2[10:13] = 20.0
    r2 = radial_delta_stats(ct2, mk, spacing_mm=0.7)
    assert r2['frac_over']['1.0'] < 0.1 and r2['delta_max'] > 10
    assert r2['max_contig_overshoot'] == 3
    # raggi non attraversati censurati come no_crossing
    mk3 = mk.copy(); mk3[:8] = np.nan
    r3 = radial_delta_stats(ct, mk3, spacing_mm=0.7)
    assert abs(r3['no_crossing_frac'] - 8.0 / n) < 1e-9
    assert abs(r3['frac_valid'] - (n - 8) / n) < 1e-9


def test_connected_dumbbell_has_low_solidity():
    """Sezione bilobata CONNESSA (biforcazione a manubrio): n_components==1 ma
    la solidity e' bassa -> e' il segnale giusto, non n_components (rimedio #2)."""
    iso = 0.3
    D = 10.0
    # due lobi distanti uniti da un collo sottile: manubrio classico (connesso)
    a = _cylinder((5, 220, 220), iso, 33.0, 24.0, D / 2, D / 2)
    b = _cylinder((5, 220, 220), iso, 33.0, 42.0, D / 2, D / 2)
    Ny = Nx = 220
    yy, xx = np.mgrid[0:Ny, 0:Nx].astype(float)
    neck = (np.abs((yy - 33.0 / iso) * iso) < 1.0) & \
           ((xx * iso > 24.0) & (xx * iso < 42.0))
    m = np.clip(a + b + np.repeat(neck[None], 5, axis=0), 0, 1).astype(np.float32)
    c = np.array([2.0, 33.0 / iso, 24.0 / iso])
    s = mask_section(m, c, T_Z, iso, r_est_mm=D / 2)   # ROI espansa (default)
    assert s['n_components'] == 1                     # connessa dal collo
    assert s['valid_mask_section'] is True            # centro in maschera, non troncata
    assert s['solidity'] < 0.85                       # ma NON convessa (manubrio)
    # confronto: un cerchio singolo ha solidity alta
    m1 = _cylinder((5, 220, 220), iso, 33.0, 33.0, D / 2, D / 2)
    s1 = mask_section(m1, np.array([2.0, 33.0 / iso, 33.0 / iso]), T_Z, iso, D / 2)
    assert s1['solidity'] > s['solidity']


def test_radii_depend_on_origin():
    """I raggi dipendono dall'origine: spostando il centro rispetto al lume, i
    raggi diventano asimmetrici. Motiva l'uso di UNA origine comune (cadj) per
    CT e maschera nel confronto radiale (rimedio #1)."""
    iso = 0.3
    R = 8.0
    m = _cylinder((5, 200, 200), iso, 30.0, 30.0, R, R)
    from section_metrics import perp_basis
    u, v = perp_basis(T_Z)
    c0 = np.array([2.0, 30.0 / iso, 30.0 / iso])            # centrato
    c1 = np.array([2.0, 30.0 / iso, (30.0 + 3.0) / iso])    # 3 mm fuori centro
    rr0 = mask_ray_radii(m, c0, u, v, iso, rmax_mm=18.0)
    rr1 = mask_ray_radii(m, c1, u, v, iso, rmax_mm=18.0)
    assert np.nanstd(rr0) < 0.4                             # centrato: quasi costante
    assert np.nanstd(rr1) > 1.5                             # decentrato: asimmetrico


def test_blank_record_schema_is_uniform():
    """Ogni record ha TUTTE le chiavi (null se assenti): niente chiavi mancanti
    per le sezioni senza lume CT (rimedio #4)."""
    from section_metrics import SECTION_KEYS, blank_section_record
    rec = blank_section_record()
    assert set(rec) == set(SECTION_KEYS)
    assert all(rec[k] is None for k in SECTION_KEYS)


def test_branch_summary_uses_homologous_populations():
    """ct_mask_ratio e overshoot solo su paired_valid (mask valida AND CT
    reportabile); la morfometria su mask_valid; esclude giunzione e salti d'area
    (rimedio #3)."""
    from section_metrics import blank_section_record, branch_mask_summary

    def mk(**kw):
        r = blank_section_record(); r.update(kw); return r

    recs = [
        # paired completa: CT reportabile + maschera valida + raggi validi
        mk(valid_mask_section=True, ct_available=True, ct_reportable=True,
           d_eq_ct=20.0, d_mask_eq=19.0, d_mask_min=13.0, d_mask_maj=25.0,
           aspect=0.52, radial={'frac_valid': 1.0, 'frac_over': {'0.5': 0.4}}),
        # maschera valida ma CT NON reportabile: entra in morfometria, NON in ratio
        mk(valid_mask_section=True, ct_available=True, ct_reportable=False,
           d_eq_ct=99.0, d_mask_eq=18.0, d_mask_min=12.0, d_mask_maj=24.0,
           aspect=0.5, radial={'frac_valid': 1.0, 'frac_over': {'0.5': 0.99}}),
        # diam paired ma radial con pochi raggi validi: conta nel ratio, NON in overshoot
        mk(valid_mask_section=True, ct_available=True, ct_reportable=True,
           d_eq_ct=21.0, d_mask_eq=20.0, d_mask_min=13.0, d_mask_maj=26.0,
           aspect=0.5, radial={'frac_valid': 0.1, 'frac_over': {'0.5': 0.95}}),
        # giunzionale: esclusa da tutto
        mk(valid_mask_section=True, near_junction=True, ct_available=True,
           ct_reportable=True, d_eq_ct=5.0, d_mask_eq=5.0,
           radial={'frac_valid': 1.0, 'frac_over': {'0.5': 0.9}}),
    ]
    s = branch_mask_summary(recs)
    assert s['n_sez_mask_valide'] == 3          # le tre valide non-giunzionali
    assert s['n_sez_paired_diam'] == 2          # due CT reportabili
    assert s['n_sez_paired_radial'] == 1        # solo una con raggi sufficienti
    assert abs(s['ct_mask_ratio'] - np.median([20/19, 21/20])) < 2e-3
    assert s['overshoot_frac'] == 0.4           # solo la radial valida, non 0.95/0.99
