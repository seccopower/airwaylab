# AirwayLab — Protocollo minimo (v1.0 CONGELATO)

Descrittori strutturali esplorativi e propagazione dell'incertezza

> **Stato: CONGELATO v1.0.** Estimandi e scenario fissati *prima* dell'ensemble di
> incertezza (#6). Ogni modifica agli estimandi richiede un nuovo numero di versione.
> Riferimento: review GPT (RESPINGO → recuperabile), condizioni #2 (chiusa), #5, #6, #7.
> Uso **interno ed esplorativo**: nessun endpoint scientifico validato, nessuna misura
> funzionale (ventilazione, perfusione, V/Q, spazio morto, distruzione).

---

## 1. Scopo

Fissare, per ciascun descrittore, **cosa** stimiamo (estimando), **su quale popolazione**,
**come** trattiamo il dato mancante, in **quale scenario** del modello, **quali output di
incertezza** riportiamo e **quando** un risultato è non informativo. Questo documento è la
precondizione congelata per l'implementazione della #6.

---

## 2. Estimandi

### 2.1 Indice di attenuazione pesato dal modello resistivo (morphomap)

**Estimando.** `I = Σ_i w_i · LAA_i`, con `w_i = q_i / Σ_j q_j` e
`LAA_i = frazione di voxel del territorio i con HU < −950` (piena risoluzione iso).
Media della frazione LAA inspiratoria pesata dalle quote di flusso del modello resistivo.
NON ventilazione/perfusione/spazio morto/distruzione.

**Popolazione.** Territori terminali con (a) ramo terminale nell'albero misurato e
(b) quota di flusso `q_i` definita dal modello. Territori senza `q_i`: **esclusi** ed
elencati (conteggio + volume), **mai** `q_i = 0`.

**Missingness.** Territorio senza flusso → `unevaluable`, fuori dalla somma. Territorio senza
voxel LAA validi (fuori maschera) → escluso dal calcolo di `LAA_i`.

**Soglia LAA = −950 HU** (fissata). Confronto tra pazienti solo a parità di protocollo.
`f_tessuto` resta descrittore affiancato, non entra in `I`.

### 2.2 Copertura della maschera delle vie aeree (discordanza, asse 1)

**Estimando.** Per lobo, `coverage_gap_frac` = frazione di voxel di parenchima (griglia ds)
con `d_aereo − d_vascolare > 10 mm` (**soglia 10 mm fissata**). Misura di **copertura
algoritmica**: parenchima vicino a un vaso ma **non rappresentato** nella maschera aerea.

**Popolazione.** Voxel di parenchima del lobo con delta finito.

**Missingness.** La via non rappresentata **è** l'oggetto della misura e resta `missing`:
non convertita in diametro/rapporto. Cause (risoluzione, profondità di segmentazione, errore,
interruzione reale) **indistinguibili** senza revisione. `ct_verified_occlusion` riservato a
futura annotazione positiva. **Non** fondere con l'asse 2.3.

### 2.3 Rapporto morfometrico bronco–arteria (discordanza, asse 2)

**Estimando.** Per lobo: `ba_med`, `ba_gt1_frac` (soglia BA = 1). **In aggiunta** si riporta
`log(BA)` e la **stratificazione per generazione** (vista complementare, decisa in v1.0).

**Popolazione (unità = coppia reportabile).** Bronchi con `qc == ok` e `d_mean` presente,
appaiati a un vaso satellite valido (`pair.py`). Aggregazione **per ramo, poi per lobo**.
Minimo **5 coppie reportabili per lobo**, sotto il quale → `dati insufficienti`.

**Missingness.** Bronchi non reportabili/non appaiati non contribuiscono (nessuno zero,
nessun rapporto estremo). Il rapporto **non** distingue dilatazione da assottigliamento arterioso.

### 2.4 Volumi della maschera vascolare e rapporto A/V

**Estimando.** `art_ml`, `vein_ml` (volumi delle **maschere**), `av_ratio`, globali e per lobo.
**Non** volume ematico, **non** perfusione.

**Limiti.** Dipendenza da segmentazione e profondità; possibile diversa sensibilità del
segmentatore per A vs V. **BV5/pruning ritirato**; un futuro scale-space sarà misura nuova.

---

## 3. Scenario del modello di flusso (prespecificato)

Quote `q_i` dal modello resistivo 1D, **scenario di riferimento**:

- flusso inspiratorio totale **fisso** (allocazione relativa, non conduttanza intrinseca);
- Poiseuille + termine di **Pedley attivo**;
- esponente di **Murray = 3.0**; completamento periferico e diametro d'arresto ai default;
  imputazione asimmetrica dei diametri periferici basata sul territorio.

Snapshot esatto di `PARAMS` allegato all'esecuzione (da `flow.json`). Ogni cambio di scenario
è un estimando diverso e va ri-dichiarato.

---

## 4. Output di incertezza (#6)

Per `q_i` (lobare), per l'indice `I` e per la classificazione a soglia:

- **mediana** e **intervallo 5–95%** da ensemble;
- **probabilità di rango peggiore** per ciascun lobo (indice e copertura);
- **stabilità della classificazione** (probabilità di cambio etichetta a soglia);
- **quota dell'incertezza da segmenti imputati** vs misurati;
- **convergenza numerica** del solver riportata **separatamente** dall'incertezza epistemica.

### 4.1 Ensemble — perturbazioni CONGIUNTE e correlate (fissato)

**300 repliche** (seed registrato). Per replica, perturbazioni congiunte che mantengono alberi
morfometricamente plausibili:

- diametri **misurati**: rumore per-misura **±10%** (o dal coefficiente di ripetibilità se
  disponibile — altrimenti ±10%, dichiarato);
- diametri **imputati**: perturbazione **correlata entro lo stesso sottoalbero** (non indipendente);
- esponente di taper/**Murray in 2.6–3.3**;
- rapporto L/d;
- completamento periferico e diametro d'arresto (range della sensibilità esistente);
- topologia/**pruning** (inclusione-esclusione di rami dubbi);
- **Pedley on/off**;
- assegnazione dei territori terminali.

### 4.2 Ablazione strutturale (#5, dentro la #6)

`q` **airway-only** vs `q` **condizionato dal territorio/completamento**. Se cambia i **ranghi
lobari** → **incertezza strutturale del modello**, riportata come tale (non collinearità).

---

## 5. Criterio di "instabile / non informativo"

Un risultato lobare è **non informativo** se, sull'ensemble:

- il suo **rango** non è stabile, cioè non mantiene la posizione relativa in **≥ 80%** delle
  repliche; oppure
- la sua **etichetta a soglia** cambia in **> 20%** delle repliche; oppure
- la popolazione minima non è raggiunta (≥ 5 coppie BA per lobo; territori `unevaluable` non
  eccessivi).

I non informativi sono mostrati come tali, non nascosti.

---

## 6. Fuori ambito adesso (validazione, segue)

- validazione tecnica bronco–arteria (annotazione radiologica cieca; accordo interlettore/ripetibilità);
- validazione tecnica vascolare (phantom + scale-space/centerline; ripetibilità a kernel/dose/spessore, A vs V);
- validazione del modello q (SPECT ventilatoria / Galligas-PET / ¹²⁹Xe-MRI; spirometria = globale, non regionale);
- claim di "spazio morto": richiede riferimento di perfusione (V/Q SPECT, DECT, MRI) — non perseguibile ora;
- inspirio/espirio + PRM (air-trapping/fSAD, non misura diretta);
- longitudinale biologici: scan–rescan, protocollo identico, controllo volume inspiratorio,
  endpoint prespecificato, variazione oltre il repeatability coefficient.

---

## 7. Versionamento

- **v1.0 CONGELATO.** Destinazione in repo: `docs/protocollo_incertezza_v1.0.md`.
- L'estimando non cambia senza nuovo numero di versione.
- L'ensemble (#6) si implementa su questo protocollo.
