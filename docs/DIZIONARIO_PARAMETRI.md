# AirwayLab — Dizionario dei parametri

Riferimento completo dei parametri del report, allineato a **v1.0.0**. Ogni voce:
*cos'è · come si calcola (dal codice) · cosa significa · limiti · file sorgente*.
I file sono in `airwaylab/pipeline/` salvo diversa indicazione.

> Convenzione trasparenza: un valore di calibro/parete esiste **solo** sui rami
> `qc == 'ok'` (regola a due regimi). Sugli altri è `null`, con il grezzo nelle
> colonne `*_raw_nonreportable`.

> **Stato dei descrittori:** i descrittori aggiunti nel 2026-08 (Pi10, tapering,
> AFD, parenchima, gradiente di pruning, composizione corporea) sono **esplorativi,
> non endpoint validati**. Ogni JSON lo dichiara (`status: exploratory` + blocco
> `provenance`). I limiti metodologici profondi e le condizioni minime prima di un
> methods paper sono raccolti in `docs/VALIDATION_BACKLOG.md`, sezione E.

---

## 0. Fondamenta (rendono leggibile tutto il resto)

**Spacing isotropico (`iso`)** — La TC viene ricampionata a voxel cubici, al passo
nativo più fine con pavimento 0.5 mm (regolabile `--spacing` / `AIRWAYLAB_SPACING`).
Tutte le misure vivono su questa griglia. *Limite:* le fette spesse limitano ciò che
è misurabile (→ floor). *Codice:* `segment.py`, `anonymize.py`; valore in
`out/seg_info.json`.

**Floor di risoluzione (`floor_calibro_mm`)** — Calibro minimo misurabile con
onestà, **dipendente dall'orientamento** del ramo: lo spacing nativo proiettato nel
piano della sezione, autovalore maggiore della metrica anisotropa `P·M·P`
(P = I − t·tᵀ, M = diag(s²)). Aggregato per ramo come massimo dei floor locali lungo
il percorso. *Significato:* limite di **processo**, non fisico validato; non si
aggira con l'upsampling. *Codice:* `measure.py`, soglie in `qc_params.py`.

**Classi QC (`qc`)** — `ok` (misurabile) · `sotto-risoluzione` (sotto il floor:
tenuto per topologia/territori, calibro nullo) · `fuga-contorno` (half-max sfora la
maschera) · `no-lume` (air-witness fallito) · `moncone` (troppo corto) ·
`poche-sezioni`. *Codice:* `measure.py`, `witness.py`.

**Air-witness (`hu_lume`, `aria_pct`)** — Ogni ramo verificato contro la TC: HU
mediana e frazione d'aria sulla linea centrale (campionamento tri-lineare
sub-voxel). Se non c'è aria lungo l'asse → `no-lume`. *Significato:* impedisce a una
maschera generosa di inventare vie aeree. *Codice:* `witness.py`, soglie
`qc_params.py`.

---

## 1. Vie aeree, per ramo (calibro e parete)

**Geometria di misura** — Sezioni perpendicolari al centerline ogni ~1 mm; tangente
= primo autovettore PCA locale (robusta sui rami curvi); ricentraggio sul baricentro
dell'aria connessa (`< −650 HU`) prima di misurare; si misurano tutte le sezioni
valide (no sezione "rappresentativa"). *Codice:* `lumen.py` (`analyze_section`,
`pca_tangent`), `measure.py`.

**`d_mean` — calibro medio del lume (mm)** — *Come:* per ogni sezione, 64 raggi;
bordo interno al **half-max** `lumen_lvl + 0.5·(picco − lumen_lvl)` (con `lumen_lvl` =
15° percentile dell'aria connessa), interpolato sub-voxel; `d_eq = 2·√(area/π)` con
`area = ½·Σ r²·Δθ` (area-equivalente); **d_mean = mediana** dei d_eq di sezione
accettati. *Significato:* calibro tipico del ramo, robusto ai lumi ellittici.
*Limiti:* solo rami `ok`. *Codice:* `lumen.py`, aggregazione `measure.py`.

**`d_min` — calibro minimo (mm)** — Il **minimo** dei d_eq di sezione → la sezione
più stretta. *Significato:* sensibilità alla **stenosi**. *Codice:* `measure.py`.

**`wall` — spessore di parete (mm)** — *Come:* per settore, picco di parete entro un
tetto fisico (5 mm) e discesa a half-max verso il fondo parenchimale
`fondo + 0.5·(picco − fondo)`; **requisito**: oltre la parete deve esserci parenchima
aerato (`< −650 HU`), altrimenti settore NON valido (vaso/mediastino a contatto);
filtro di coerenza circolare tra settori; **wall = mediana** dei settori validi (se
≥ 25%). *Codice:* `lumen.py`, aggregazione `measure.py`.

**`wa_pct` — WA% (area di parete)** — `r_out = d_mean/2 + wall`;
`WA% = 100·(r_out² − (d_mean/2)²)/r_out²`. *Significato:* frazione della sezione
occupata dalla parete → **il** parametro del *remodeling* nell'asma. *Limiti:*
half-max sovrastima la parete sui rami piccoli (volume parziale); più affidabile su
prossimali/segmentari. *Codice:* `measure.py`.

**`wall_ok_pct` / `wall_over_cap_pct`** — Copertura dei settori con parete valida; e
frazione di settori oltre il tetto fisiologico (`~0.18·calibro + 0.6`). *Significato:*
il secondo è un **flag QC**, non una censura: alto = possibile patologia *o*
contaminazione da vaso. *Codice:* `lumen.py`, `measure.py`.

**Diametri di maschera (`d_mask_eq`, `d_mask_min`, `d_mask_maj`, `aspect_mask`)** —
Metriche geometriche della sezione di **maschera** (non della TC): area-equivalente,
Feret min/max, rapporto d'aspetto. Presenti anche dove il lume CT non è misurabile
(servono al leak-QC e come audit). *Codice:* `section_metrics.py`.

**`ct_mask_ratio`, `overshoot_frac`** — Confronto per-sezione tra calibro CT half-max
e calibro maschera (mediana dei rapporti; frazione di raggi CT oltre la maschera).
*Significato:* audit di coerenza CT↔maschera. *Codice:* `section_metrics.py`.

**`pi10` — Pi10_AirwayLab (√WA a Pi = 10 mm)** — *Cos'è:* la nostra implementazione
dell'indice di Nakano — √WA **predetta** a un perimetro interno del lume di 10 mm.
È un descrittore esplorativo per soggetto, non un endpoint validato; il suffisso
`_AirwayLab` ricorda che dipende dalla nostra pipeline di misura. *Come:* per ogni
via `ok` (con calibro e parete) il punto `(Pi, √WA)` con `Pi = π·d_lume` e
`WA = π·(r_out² − r_in²)` (`r_in = d/2`, `r_out = r_in + parete`); regressione
**√WA = a + b·Pi** su tutte le vie del soggetto → **Pi10 = a + b·10**. Metrica a
livello di **soggetto** (un numero per esame). *Significato:* riporta lo spessore di
parete a un perimetro fisso (10 mm) per confrontare vie di calibro diverso — **non**
è una "normalizzazione di WA%", è il valore letto dalla retta √WA–Pi in un punto
fisso. *Diagnostica esportata:* range dei perimetri osservati (`pi_min`/`pi_max`),
flag `extrapolation` quando 10 mm cade fuori dal range (il valore è estrapolato, non
interpolato), copertura, CI 95% bootstrap e sensibilità leave-one-out. *Limiti:*
dipende da quali vie sono misurabili (protocollo/inspirazione) e dall'approssimazione
circolare del perimetro; serve n ≥ 10; confronti solo a parità di protocollo.
*Codice:* `pi10_core.py`, `pi10.py` → `out/pi10.json`.

**Tapering (`taper_ratio_med`, `taper_rate_pct_per_cm`)** — *Cos'è:* la rastremazione
del calibro verso la periferia; la sua perdita è il segno delle bronchiectasie.
*Come:* (1) **rapporto figlio/genitore** del lume su ogni **adiacenza topologica
padre-figlio** tra rami `ok` (incluse le catene mono-figlio, non solo le biforcazioni
vere) → mediana e `frac_no_taper` (frazione con ratio > 0.9); (2) **gradiente
globale**: fit `ln(d_mean) = a + b·L` sulla distanza cumulativa **dalla radice** al
punto medio del ramo → `rate = (1 − e^{b·10})·100` = % di riduzione del calibro per
cm. *Significato:* `2^(−1/3) ≈ 0.79` è il valore atteso per un ramo che si biforca in
modo simmetrico (Murray) — è un **riferimento teorico**, non una soglia clinica di
normalità; ratio → 1 o rate → 0 indica meno rastremazione. *Limiti:* dipende dai rami
misurabili; le soglie 0.79/0.9 sono operative, i riferimenti di normalità vanno
stabiliti sulla coorte a parità di protocollo. *Codice:* `tapering_core.py`,
`tapering.py` → `out/tapering.json`.

**BA ratio (`ba`)** — Rapporto `diametro lume bronco / diametro arteria satellite`,
su coppie vicine, parallele (`|cos| > 0.6`) e di calibro plausibile; mediana per
generazione e globale. *Significato:* BA > 1 = bronco dilatato (bronchiectasia).
*Limiti:* solo coppie rappresentate; richiede le maschere arteriose (TS). *Codice:*
`pair.py` → `out/pairing.json`.

---

**Morfometria dell'albero — conteggi + AFD** — *Cos'è:* quanto in profondità e quanto
complesso arriva l'albero visibile (indice di completezza + rimodellamento). *Come:*
conteggi (`n_branches`, `n_terminals`, per generazione, lunghezza totale) dai rami; e
**AFD** (Airway Fractal Dimension) = box-counting sullo scheletro 3D — per una serie
di lati di cella `s` si contano le celle occupate `N(s)`; `AFD = pendenza di
log N(s) su log(1/s)`. Metodo `afd_3d_fixedgrid` (griglia a origine singola, poche
scale) — **non** è il FracLac a posizioni multiple: è una nostra variante, non una
misura validata. *Significato:* descrivono quanto è profondo/ramificato l'albero
**segmentato**; non sono endpoint biologici. *Denominatori:* i conteggi includono
**tutti** i rami del grafo (anche no-lume/sotto-risoluzione); esportiamo separati
`n_branches_grafo` e `n_branches_qc_ok` perché il numero non venga letto come "vie
aeree visibili". *Limiti:* conteggi e AFD dipendono dalla **profondità di
segmentazione e dal protocollo** → confronti solo mela-con-mela (stesso
backend/versione); la generazione si gonfia con i rami spuri. Conteggi e AFD sono
descrittori complementari: nessuno dei due è validato come endpoint. *Codice:*
`treestats_core.py`, `treestats.py` → `out/treestats.json`.

## 2. Struttura globale

**Generazione (`gen`)** — Profondità topologica dalla trachea: gen 0 = trachea
(radice → carena), +1 **solo a una biforcazione vera** (≥ 2 figli); un ramo che
prosegue senza dividersi mantiene la generazione. *Limite:* rami spuri creano
biforcazioni finte e **gonfiano** le generazioni (per questo il matching pre/post va
fatto per `aid`, non per gen). *Codice:* `tree.py`.

**Dysanapsis (Smith/MESA)** — `geo = exp(media(log d_mean))` sui rami etichettati e
`ok` (fallback: `ok` e gen ≤ 3 se < 8); `dysanapsis = (geo/10) / ³√(volume_ml)`,
adimensionale. Riferimento MESA ~0.033 ± 0.004. *Significato:* rapporto
calibro-vie/taglia-polmone (tratto di sviluppo). *Limiti:* stabile → buona covariata,
cattivo endpoint di trattamento; dipende dall'inspirazione. *Codice:* `lung.py` →
`out/lung_metrics.json`.

**ALR4 (Shimada 2025)** — Media geometrica dei `d_mean` delle 4 vie centrali
(trachea, principale dx, principale sx, bronco intermedio) `/ ³√(volume_ml)`; solo se
tutte e 4 presenti. Riferimento ~0.09 M / 0.08 F. *Codice:* `lung.py`.

**`diam_geo_mean_mm`** — Media geometrica dei calibri gen 0–3 (contesto). *Codice:*
`lung.py`.

**Profili longitudinali** — Calibro half-max `d` e parete `w` ogni ~1 mm lungo ogni
ramo, in funzione dell'arclength `s`; punti nei coni di giunzione esclusi.
*Significato:* stenosi/dilatazioni **dentro** un ramo. *Codice:* `profile.py` →
`out/profiles.json`.

**CPR (rettificata)** — Una **rotta** (trachea → ramo distale) raddrizzata:
ricampionamento del centerline, immagine orizzontale + profilo calibro/parete sullo
stesso asse mm, tacche dei rami attraversati, carena a `s = 0`. *Codice:* `cpr.py` →
`out/cpr.json`; rotte in `out/routes.csv`.

---

## 3. Polmone

**Maschera polmonare** — Aria (`< −500 HU`) dentro il corpo, **meno le vie aeree**
(dilatate), 2 componenti maggiori > 150 ml, chiusura leggera. *Nota:* le vie aeree
sono sottratte per costruzione (rilevante per il leak-QC). *Codice:* `lung.py` →
`out/lung_mask_ds.nii.gz`.

**`lung_volume_l`** — Voxel di polmone × volume voxel. *Significato:* il numero di
**comparabilità** pre/post (inspirazione). *Codice:* `lung.py`.

**`mld_hu` — MLD** — Media degli HU nella **componente aerata** (soglia aria di
`lung.py`, non il polmone anatomico). Meno negativa = più densa (più tessuto o meno
inspirata) — **non** è specifica per fibrosi/ground-glass. *Codice:* `lung.py`.

**`laa950_pct` — LAA-950** — `% voxel < −950 HU`. Indice di enfisema/iperinflazione.
*Limite:* su singola inspiratoria **non** è air-trapping (serve l'espiratoria).
*Codice:* `lung.py`.

**`perc15_hu` — Perc15** — 15° percentile dell'istogramma HU polmonare. Misura di
enfisema robusta alla soglia. *Codice:* `lung.py`.

**Parenchima oltre la densità media** — *Cos'è:* descrittori di disomogeneità e
organizzazione della bassa attenuazione, che MLD e LAA-950 nascondono. Calcolati sulla
**componente aerata**, non sul polmone anatomico. *Come:* (1) **forma dell'istogramma**
HU — media, SD, **skewness**, **curtosi** (descrittori nominati); (2) **eterogeneità**
= SD e IQR delle medie di densità calcolate a blocchi (~15 mm): è disomogeneità
regionale della densità, **non** una misura specifica di mosaic attenuation o
air-trapping; (3) **cluster LAA** (< −950): numero, frazione del cluster maggiore, ed
esponente **D** (Mishima) dalla legge di potenza (OLS rango-dimensione, non MLE/xmin).
**D** descrive la distribuzione dimensionale dei cluster LAA; **non** è specifico per
enfisema — Gupta non trovò differenze nell'asma, e il sottocampionamento ×3 altera la
topologia dei cluster. *Limiti:* dipende da soglia HU / kernel / volume; griglia di
analisi ×3; su inspiratoria non è air-trapping. *Codice:* `parenchyma_core.py`,
`parenchyma.py` → `out/parenchyma.json`.

**Territori (`terr_ml`)** — Partizione di Voronoi del parenchima: ogni voxel al
**seme terminale più vicino** (metà distale di ogni ramo terminale, EDT euclidea);
territorio del ramo = somma del sottoalbero; per lobo risalendo all'antenato lobare.
*Limite:* volume determinato dalle maschere; un ramo mancante fa gonfiare i vicini
(conseguenza topologica, non validazione anatomica); griglia ×3. *Codice:*
`territory.py` → `out/territories.json`, `out/territory_index.json`.

**Murray (`slope`, `r2`)** — Regressione `log10(territorio) ~ log10(d_mean)` sui rami
`ok` con territorio > 10 ml (n ≥ 8). Teoria: slope ≈ **3**. *Significato:* aderenza al
disegno metabolico + **controllo di coerenza** delle misure. *Codice:* `territory.py`
→ `out/murray.json`.

---

## 4. Vasi (TC senza contrasto → volume di vaso, mai perfusione)

**`tbv_ml` — TBV** — Volume della maschera di **strutture dense candidate a vaso** nel
polmone (escluse le pareti bronchiali); su TC senza contrasto non è puro vaso.
Arterie + vene insieme. Guardia se < 20 ml. *Codice:* `vessels.py` →
`out/vessel_metrics.json`.

**`bv5_ml`/`bv5_frac`, `bv10_*`** — **Residuo di apertura morfologica** a r ≈ 1.26 mm
(`BV5 = TBV − opened_volume(r5)`): un proxy morfologico delle strutture sottili, **non**
la sezione ortogonale del vaso e **non** il BV5 scale-space validato (Estépar).
*Limiti:* esplorativo; assoluto non confrontabile, dipende da kernel/dose → solo
confronti mela-con-mela. *Codice:* `vessels.py` (`opened_volume`, apertura a r ≈ 1.26
mm) → `out/vessel_metrics.json`. NB: `av_core.bvn_volumes`/`radius_for_csa` servono
l'analisi A/V **per lobo** (sotto), non il TBV/BV5 globale.

**Gradiente di pruning vascolare** — *Cos'è:* la perdita centro→periferia dei piccoli
vasi (pruning del piccolo circolo), localizzata invece che riassunta da BV5 globale.
*Come:* distanza dalla pleura dal campo EDT della maschera polmonare (griglia ×3);
"piccole strutture" = **residuo di apertura** a r ≈ 1.26 mm (stessa operazione di
`vessels.py`, un proxy — non la sezione ortogonale); per gusci di distanza si calcola
la loro **densità** (ml piccole strutture / ml polmone); si aggregano **periferia**
(dist ≤ 15 mm) vs **centro**. `pruning_ratio` = densità periferica / centrale (ridotto
= meno piccole strutture in periferia; **esplorativo**, non una diagnosi di pruning),
più la pendenza densità–distanza. *Limiti:* dipende da soglia vasi / kernel / volume e
dall'approssimazione ×3 del campo di distanza (aliasing di fase) → mela-con-mela.
*Codice:* `vascular_gradient_core.py`, `vascular_gradient.py` →
`out/vascular_gradient.json`.

**A/V per lobo (`art_ml`, `vein_ml`, `av_ratio`)** — Volumi arterioso/venoso per lobo
dalle maschere separate di TotalSegmentator; `av_ratio = art_ml / vein_ml`. *Codice:*
`vasculature.py`, `av_core.py` (`av_ratio`, `aggregate_by_lobe`) →
`out/vascular_av.json`.

**Discordanza — `coverage_gap_frac`** — Frazione di parenchima del lobo vicino a un
vaso (≤ 10 mm) ma **non coperto** dalla maschera delle vie aeree. È **copertura
algoritmica** ("via aerea non rappresentata"), NON occlusione né morfometria; cause
indistinguibili (risoluzione/profondità/errore/interruzione reale). *Codice:*
`discordance.py`, mappa in `dual.py` → `out/discordance_regional.json`.

**Discordanza — `ba_med`, `ba_gt1_frac`** — Morfometria bronco-arteria per lobo,
**solo su coppie rappresentate e riportabili**; `ba_gt1_frac` = frazione con BA > 1.
Non distingue dilatazione bronchiale da assottigliamento arterioso. **Mai fuso** con
la copertura. *Codice:* `discordance.py`.

---

## 5. Tappi di muco (prototipo proponi-e-conferma)

**Candidati plug** — Rami **terminali** con calibro ≥ 2.5 mm half-max (o
d_maschera ≥ floor 3-voxel se `sotto-risoluzione`; mai `no-lume`/`fuga-contorno`). Si
sonda un **cilindro di continuazione** (1.5–9 mm oltre il moncone): se oltre c'è aria
(`< −500 HU`) → non è plug; se c'è **tessuto molle** (≈ −500..200 HU) e **non vaso**
(`frac_vaso ≤ 0.55`) → candidato. *Codice:* `plugs.py` → `out/plugs.json`.

**Campi per candidato** — `zona` (nome anatomico più vicino), `d_moncone`, `hu_oltre`
(densità oltre), `frac_vaso`, **`testimone`** (vasi nel manicotto attorno alla
continuazione — l'arteria satellite dovrebbe restare), `terr_ml` (territorio a valle),
`img` (vista longitudinale, moncone a `s = 0`). Ordinamento `(hu+500)·(1+testimone)`.
*Limiti:* rilevatore di **candidati**, da confermare a occhio; cattura solo il
terminale che si ferma su tessuto molle; confondenti = atelettasia, linfonodo, vaso.

---

## 6. Analisi esplorative (mai misura clinica)

**Mappa multi-asse — indice `I` (`cond_to_destroyed`)** — Due assi separati per
lobo: conduttanza `q` (flusso del modello) e LAA inspiratoria. Indice = media pesata
`w_i = q_i/Σq`, `I = Σ w_i·LAA_i`. Campi: `cond_frac` (quota di flusso), `laa`,
`ds_share` (quota dell'indice), `prevalenza` (etichetta a soglie). **Non** è
ventilazione/perfusione/distruzione. *Limiti:* LAA ≠ distruzione su inspiratoria; le
`q` da diametri in gran parte imputati; dipende da soglia HU e volume. *Codice:*
`morphomap_core.py`, `morphomap.py` → `out/morphomap.json`.

**Modello di flusso 1D** — Rete di resistenze: **Poiseuille** per ramo + correzione
**Pedley** alle biforcazioni (`Z = 0.327·√(Re·d/L)`); `μ = 1.81e-5 Pa·s`,
`ρ = 1.20 kg/m³`, flusso totale fisso `Q = 0.5 L/s`, completamento periferico alla
Murray, diametri imputati dove non misurati. Output: `q` per ramo, pressione
cumulativa, ventilazione per territorio. *Limiti:* simulazione su geometria statica,
periferia imputata; non è il flusso reale. *Codice:* `flow_model.py`, fisica pura
`flow_core.py`, produzione `flow.py` → `out/flow.json`.

**`imputed_diameter`** — `d = min(d_parent, max(dfloor, d_parent·frac^(1/nexp)))`,
`frac` = quota di territorio del sottoalbero, `nexp = 3` (Murray). *Codice:*
`impute_core.py`.

**Ensemble di incertezza** — **Robustezza** a scenari prespecificati (non incertezza
calibrata): perturba diametri misurati e imputati, ablazione OAT (misurato vs
imputato), ablazione strutturale (territori on/off), rami bloccati; convergenza
N=300/600/1200. Statistiche: percentili **5/50/95**, `worst_rank_prob`,
`label_stability`, `rank_stability`, `variance_share`. *Codice:* `uncertainty_core.py`,
`uncertainty.py` → `out/uncertainty.json`.

---

## 7. Reti di sicurezza + provenienza

**Completezza etichettatura** — Verifica i 6 lobi (RUL, RML, RLL, LUL, LING, LLL);
`complete` se tutti presenti, altrimenti banner rosso + avviso. Se incompleta, i
per-lobo rappresentano solo parte del polmone. *Codice:* `label_qc.py`.

**Plausibilità lobare** — Flag sulle proporzioni volumetriche: lobo < 3%, emitorace
< 30%, RML non il più piccolo a destra, lingula > LUL. Intercetta partizioni
"complete ma sbagliate". *Limite:* flag di verifica; `lingula > LUL` scatta spesso
(candidato taratura). *Codice:* `plausibility_core.py`; esposto in `morphomap.json`.

**Leak / connettività** — `radius_explosion` (ramo col diametro-maschera ≥ 1.6× il
genitore, floor 4 mm, vie centrali escluse → leak in cisti/bolla/esofago, visibile
anche senza lume) e isole (componenti staccati). **La severità scala con la
dimensione** del ramo che si gonfia: ≥ 10 mm = alto (sospetto leak), ≥ 6 mm = medio,
sotto = basso (informativo, non fa scattare l'allarme: a quel calibro è quasi sempre
una biforcazione/rumore). Le soglie 10/6 mm sono **operative/esplorative**, non
clinicamente validate. *Limite dichiarato:* il leak
extrapolmonare/esofageo NON è in v1 (serve un inviluppo delle vie aeree). Non
penalizza i rami senza lume (possibili tappi). *Codice:* `leak_qc_core.py`,
`leak_qc.py` → `out/leak_qc.json`.

**Provenienza (`backend_info.json`)** — Backend usato, versione, checksum SHA-256
della maschera. Rende ogni risultato riproducibile e auto-documentato. *Codice:*
`airway_backend.py` → `<caso>_seg/backend_info.json`.

---

## 8. Biomarcatori opportunistici (opt-in, stesso torace)

Dallo stesso esame, *se* sono presenti le maschere di composizione corporea (task
`total`/`tissue_types` di TotalSegmentator, prodotte a parte). Meccanica opt-in come
il ponte AeroPath — non appesantisce il comando unico. Strumento: `tools/bodycomp.py`.

**Osso (`bone`)** — attenuazione trabecolare media per corpo vertebrale (HU, con lieve
erosione per togliere il corticale); media e minimo. **Nessun flag automatico**: la
soglia ~110 HU non è validata per questa ROI (corpo vertebrale intero eroso di 2 vox,
livelli toracici misti) — resta come nota, non come allarme. *Significato:* indicatore
grezzo di densità ossea, **esplorativo**; non è uno screening osteoporosi validato.
*Limiti:* **non diagnostico**; dipende da livello vertebrale, kernel, kV. *Codice:*
`bodycomp_core.py`, `tools/bodycomp.py`.

**Muscolo (`muscle`)** — volume del muscolo scheletrico (ml) e attenuazione media (HU
bassa = più infiltrazione adiposa). *Significato:* massa e qualità muscolare grezze;
**non** è una misura validata di sarcopenia (che richiede indici e soglie EWGSOP2).

**Grasso (`fat`)** — volume sottocutaneo (`sat_ml`) e grasso **interno del tronco**
(`internal_fat_ml`, dal `torso_fat` di TotalSegmentator — **non** è VAT segmentato),
con rapporto `internal_sat_ratio`.

*Limiti comuni:* valori dipendenti da kernel/dose/kV e campo di vista → confronti solo
a parità di protocollo. Screening, non diagnosi. Output: `bodycomp.json`.

## Dove vivono i valori

| File | Contenuto |
|---|---|
| `<caso>_branches.csv` | tutte le metriche per-ramo (calibro, parete, WA%, QC, diametri maschera, floor, colonne `*_raw_nonreportable`) |
| `<caso>_routes.csv` | rotte per la CPR |
| `out/seg_info.json` | iso, spacing nativo, backend, versione, provenienza |
| `out/lung_metrics.json` | volume, MLD, LAA-950, Perc15, dysanapsis, ALR4 |
| `out/parenchyma.json` | istogramma HU (skew/kurt), eterogeneità, cluster LAA (D) |
| `out/pi10.json` | Pi10 (√WA a Pi=10), slope, R², n, range Pi, flag estrapolazione, CI95 bootstrap, leave-one-out |
| `out/tapering.json` | rapporto figlio/genitore, gradiente %/cm |
| `out/treestats.json` | conteggi rami/terminali/gen, lunghezza, AFD |
| `out/territories.json`, `out/murray.json`, `out/territory_index.json` | territori, fit di Murray, indice lobo |
| `out/vessel_metrics.json`, `out/vascular_av.json` | TBV/BV5/BV10, A/V per lobo |
| `out/vascular_gradient.json` | pruning: densità periferica/centrale, ratio, gradiente |
| `out/pairing.json`, `out/discordance_regional.json` | BA ratio, discordanza regionale |
| `out/plugs.json` | candidati mucus plug |
| `out/morphomap.json`, `out/flow.json`, `out/uncertainty.json` | esplorativi (+ `plausibilita_lobare`, `labeling` in morphomap.json) |
| `out/leak_qc.json` | leak/connettività |
| `<caso>_seg/backend_info.json` | provenienza segmentazione |
| `bodycomp.json` (opt-in) | osso (HU vertebrale), muscolo (ml/HU), grasso (SAT / interno del tronco) |

---

*Documento allineato ad AirwayLab v1.0.0. I riferimenti ai file puntano a
`airwaylab/pipeline/`. Aggiornare a ogni cambio di formula o soglia.*
