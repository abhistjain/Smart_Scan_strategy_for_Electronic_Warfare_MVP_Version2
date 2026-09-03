# Smart Scan Strategy for Electronic Warfare ML-based ES Receiver Scheduler

**SIH26055** · An Electronic Support (ES) receiver scheduler that learns *where*
and *when* to look across a wide RF spectrum **with no prior reliable
intelligence** on the emitters  minimising intercept time and maximising the
interception rate against periodic (spatially-scanning) and frequency-agile
emitters.

It combines a **Bayesian belief filter**, **Thompson sampling** (to learn
unknown channel dynamics online), a closed-form **Whittle-index** restless-bandit
scheduler, and **Lomb-Scargle / von-Mises periodicity detection** for
"intercept-ahead" tracking  and shows it beating the incumbent open-loop sweep
(and random / greedy baselines) live, on the **identical environment
trajectory**, in both a controllable **Synthetic** mode and a **Real Data** mode
derived from the Turing Synthetic Radar Dataset (TSRD).

---

## Table of contents

- [Why this satisfies the problem statement](#why-this-satisfies-the-problem-statement)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [The TSRD dataset (Real Data mode)](#the-tsrd-dataset-real-data-mode)
- [How the algorithms work (plain language)](#how-the-algorithms-work-plain-language)
- [Figures of merit](#figures-of-merit)
- [Emitter classification & AI Analyst (add-on)](#emitter-classification--ai-analyst-add-on)
- [API reference](#api-reference)
- [Testing & scientific validation](#testing--scientific-validation)
- [Project layout](#project-layout)
- [Stretch goals](#stretch-goals)

---

## Why this satisfies the problem statement

| Requirement (official PS) | Where it lives |
| --- | --- |
| 2-D search: right frequency at the right time | Top-M scheduler over belief-ranked bands each tick (`algo/smart_scheduler.py`) |
| Simulated RF environment with ground-truth per-band, per-slot status | `sim/synthetic_environment.py` and TSRD Stare-mode occupancy grid (`sim/tsrd_environment.py`) |
| No prior emitter intelligence | Thompson sampling with `Beta(1,1)` priors (`algo/thompson.py`) |
| Frequency-agile emitters | Hopper archetype + TSRD's native hoppers |
| Periodic scan / spatially-scanning emitters | Periodic archetype + Lomb-Scargle intercept-ahead (`algo/periodicity.py`); AoA is a Phase-2 extension |
| Robust ML-based scheduler minimising intercept time, maximising rate | Whittle index + belief filter + Thompson composite ("smart") |
| Beat the open-loop sweep | Sequential-sweep baseline on the identical trajectory (`algo/baselines.py`) |
| All required figures of merit, live | `metrics/metrics.py`, streamed per strategy over WebSocket |
| Expected solution: ML-based ES receiver scheduler software | This repo (backend scheduler + mission-control dashboard) |

---

## Architecture

```
 Synthetic Generator            TSRD Cache Loader
 (Markov/periodic/hopper)       (occupancy grid + emitter stats from Stare-mode PDWs)
            \                         /
             \                       /
          Unified RFEnvironment interface   (same interface regardless of source)
                        |  noisy observation o_i(t) on scanned bands
                        v
          Bayesian Belief Filter (HMM)  ->  b_i(t) = P(state_i = ON | obs)
                        |
          Whittle Index + Thompson Sampling (+ Lomb-Scargle intercept-ahead boost)
                        |  index score w_i(t)
                        v
          Top-M Scheduler  ->  hit/miss feedback loop
                        |
          Metrics Engine + WebSocket broadcaster
          (also runs the 3 baselines in parallel on the identical trajectory)
                        |
                        v
          Next.js real-time "mission control" dashboard
```

- **Backend:** Python 3.10+, FastAPI (REST + WebSocket), NumPy, SciPy
  (`scipy.signal.lombscargle`), Pydantic, `huggingface_hub`, `pandas`/`pyarrow`.
- **Frontend:** Next.js 14 (App Router), TypeScript, TailwindCSS,
  Plotly.js via `react-plotly.js`, Framer Motion, Zustand, native WebSocket.

**Fairness guarantee:** one environment instance owns the hidden ground-truth
trajectory (advanced by its own seeded RNG, independent of what any strategy
scans). Each strategy has its *own* observation RNG, so sensor noise is
independent per strategy while the underlying world is byte-identical.

---

## Quick start

Prerequisites: **Python 3.10+** and **Node 18+**.

### 1) Backend

```bash
# from the repo root
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r backend/requirements.txt
```

Build a small **Real Data** cache (offline, no network needed  realistic fake
PDWs run through the exact same binning pipeline as the real dataset):

```bash
python backend/data/prepare_tsrd_cache.py --stub --num 3 --n-bands 24 --slot-us 2000
```

Run the API server:

```bash
cd backend
python -m uvicorn api.main:app --port 8000
```

### 2) Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Open **http://localhost:3000**, pick a preset (or the **Real Data · TSRD**
tab), and hit **Launch Simulation →**.

> The frontend proxies `/api/*` to `http://127.0.0.1:8000` via Next.js rewrites,
> and connects the dashboard's live stream to `ws://127.0.0.1:8000`. Override
> with `NEXT_PUBLIC_BACKEND_HTTP` / `NEXT_PUBLIC_BACKEND_WS` (see `.env.example`).

### Runs fully offline

If no TSRD cache is present, the **Real Data** tab is automatically disabled with
a clear message and the app runs entirely in **Synthetic** mode  the dataset is
never a hard runtime dependency.

---

## The TSRD dataset (Real Data mode)

Real Data mode replays a **pre-processed occupancy grid** built offline from the
**Turing Synthetic Radar Dataset (TSRD)**  specifically its **Stare Mode**
pulse trains, which observe the whole 0–18 GHz spectrum simultaneously and are
the closest real-world analogue to the problem statement's "ground-truth status
of each band at each time slot".

The dataset is **gated** (~70 GB, ~4 billion pulses). We **never** bulk-download
it or touch the raw pulse stream at runtime. Instead, a one-time offline script
downloads only a handful of pulse trains, bins them, and caches compact
NPZ + JSON files that the live simulation loads.

### Building the real cache (optional)

1. Create a free account at <https://huggingface.co>.
2. Accept the dataset terms at
   <https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset>.
3. Create a read token and export it (it is read via `os.environ["HF_TOKEN"]`
   and **never written to disk or committed**):
   ```bash
   # PowerShell
   $env:HF_TOKEN = "hf_xxx"
   # bash
   export HF_TOKEN=hf_xxx
   ```
4. Build the cache:
   ```bash
   python backend/data/prepare_tsrd_cache.py --real --num 5 --n-bands 24 --slot-us 2000
   ```

The script:
1. Downloads a small sample of Stare-mode **test-split** pulse trains.
2. Parses the 5-D PDWs (ToA, Centre Frequency, Pulse Width, AoA, Amplitude).
3. Picks the busiest ~500 MHz sub-band and discretises it into `N` bands.
4. Discretises the 10 s duration into fixed time slots.
5. Marks each `(band, slot)` cell **transmit (1)** / **silent (0)** by ToA +
   pulse-width overlap → the **ground-truth occupancy grid**.
6. Extracts per-emitter-cluster stats (dominant PRI/period, hop rate/range, duty
   cycle) via simple frequency+PRI clustering  *not* a full deinterleaving
   solution (that is a different challenge), just enough to (a) parameterise the
   Synthetic generator's defaults and (b) validate periodicity detection.
7. Caches everything to `backend/data/cache/` (git-ignored).

If `--real` fails (no token / no network), it **automatically falls back to
`--stub`** so the build never blocks.

> **Note on noise:** the raw TSRD data has no artificial sensor noise. When our
> scheduler "scans" a band we add our own `P_miss` / `P_fa` on top to model *our*
> receiver's imperfection  documented in `sim/tsrd_environment.py` so it is not
> confused with the dataset's native fidelity.

**Citation:** Gunn, Hosford, Jones, Zeitler, Groves, Nockles  *"The Turing
Synthetic Radar Dataset: A dataset for pulse deinterleaving."* License:
Apache-2.0. Companion utilities:
`github.com/alan-turing-institute/turing-deinterleaving-challenge`.

---

## How the algorithms work (plain language)

**1 · Bayesian belief filter** (`algo/belief_filter.py`)  for every band we keep
`b_i(t) = P(band i active | observations so far)`. Each tick we *predict* with the
Markov model `b ← b·(1−P10) + (1−b)·P01`, and for scanned bands we *update* with
exact Bayes (denominator written out, no approximation). Unscanned bands evolve by
prediction only  so we reason about bands we aren't even looking at.

**2 · Thompson sampling** (`algo/thompson.py`)  we don't know each band's
transition probabilities, so we put `Beta(1,1)` priors on `P01` and `P10`, update
them from observed consecutive-scan transitions, and **sample** from the
posteriors each decision epoch. Sampling *is* the exploration mechanism (no ad-hoc
epsilon-greedy).

**3 · Whittle index** (`algo/whittle_index.py`)  each band gets a scan-priority
score from the **closed-form Whittle index** for the 2-state restless bandit
(Liu & Zhao, 2010, Theorem 2, discounted criterion with β→1). We compute it for
**all** bands each tick and scan the Top-M. A UCB-style bonus
`+c·√(log t / n_i)` is added because the transition probabilities are learned
online. The index is provably **monotonically increasing in belief** in the
positively-correlated regime (asserted in tests).

**4 · Intercept-ahead / periodicity** (`algo/periodicity.py`)  a Lomb-Scargle
periodogram (correct for irregular/sparse sampling, which is exactly our
intermittent-scan setting) finds a dominant period from each band's hit
timestamps; a von Mises circular fit estimates the expected active phase and a
confidence. Just before the predicted window, that band's Whittle index is
boosted  directly addressing "optimally intercept a periodic scan
receiver/emitter". The predicted next-active time also drives the **average
intercept time error** metric.

**Baselines** (`algo/baselines.py`): Sequential open-loop sweep (the incumbent),
Random/round-robin, and Greedy recent-hit  all run on the identical trajectory.

---

## Figures of merit

Computed per strategy, live (`metrics/metrics.py`):

- **Pd**  intercepted active cells / all truly-active `(band, slot)` cells.
- **Pfa**  false alarms / scans of truly-idle bands.
- **Sensitivity**  detections / scans of truly-active bands (1 − miss rate),
  reported distinctly from Pd.
- **Average Intercept Rate**  successful intercepts per simulated tick.
- **Average Reward/Cost**  running mean of `R_hit − C_dwell − C_miss_penalty`.
- **% Correct Predictions**  Top-M scans matching ground-truth active state.
- **Average Intercept Time Error**  MAE between predicted and actual next-active
  tick for periodic emitters (the metric the PS singles out).

---

## Emitter classification & AI Analyst (add-on)

Two presentation-layer capabilities sit on top of the scheduler. **Both are
strictly a sensing / classification / presentation aid.**

> ### Hard scope boundary
> This layer **never** selects, recommends, or pairs a weapon/interceptor
> against a detected emitter; **never** computes trajectories, guidance, launch
> envelopes, or engagement timing; and **never** makes an "engage/fire/
> intercept-with-X" recommendation. It stops at *"here is what this signal's
> pattern resembles, a plain-English description, and a relative priority for an
> operator's attention"* — where priority only **sorts the dashboard**.

### 1 · Signature-based behaviour classification (`backend/classification/`)

Per band, a rolling feature vector is derived **only from operator-available
data** (what the receiver observed when it scanned, the belief vector, and the
existing periodicity output — never hidden ground truth):

- `features.py` — duty cycle, burst/onset rate, agility (`hop_rate`), neighbour
  co-activity (`bandwidth`), periodicity strength & period (from
  `algo/periodicity.py`), period **trend** (for tightening detection), plus
  amplitude/pulse-width proxies.
- `classifier.py` — a **transparent rule-based** decision table mapping features
  to illustrative behaviour-pattern labels — *Steady Comms-Like*,
  *Scanning/Rotating-Pattern*, *Frequency-Agile / Hopping*, *Tightening-Pattern*
  (flagged higher priority, phrased behaviour-first), or *Unclassified*. Every
  call logs a `MATCHED_RULE` string for explainability, and carries a confidence
  that grows with evidence. Labels are re-evaluated on a throttle with light
  hysteresis so chips stay **stable**, not flickering.
- `priority_score.py` — **Threat Priority Score** =
  `w_belief·belief + w_conf·confidence + w_urgency·urgency`, where urgency comes
  from the existing intercept-ahead "time to next active window". Weights are
  live-adjustable in the UI and **only** sort the belief panel.

> **Framing (shown in-app):** category labels describe *signal behaviour
> patterns* and are illustrative analogues loosely inspired by general EW
> literature — **not** a validated IFF system or real platform identification.

### 2 · AI Analyst layer (`backend/ai_analyst/`)

A thin Anthropic Claude wrapper (`claude_client.py`) powers three features:
**live band narration**, an **operator chat panel**, and an **end-of-run
summary report**. The chat's system prompt encodes the scope boundary above, so
out-of-scope questions (e.g. *"which missile should intercept this?"*) are
declined and redirected to signal-behaviour/scheduler topics.

**Fully optional & fail-safe.** Everything above runs without any API key — set
`ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`, default
`claude-3-5-haiku-latest`) in `.env` to enable narration. If the key is missing
or a call errors, the simulation, classification, priority sorting, and dashboard
all keep working; AI panels simply show *"AI narration unavailable"*.

### UI additions

Classification chips + priority sort on the belief panel, a click-through **band
detail popover** (narration, matched rule, priority breakdown, feature vector),
a collapsible **AI Analyst chat** (with the priority-weight sliders), an
**End-of-Run report modal**, and a **persistent disclaimer** shown wherever a
label or score appears.

---

## API reference

REST (JSON):

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/scenario` | create a scenario → `{ scenario_id, band_info, config }` |
| GET | `/api/scenario/{id}` | fetch config + band info |
| POST | `/api/scenario/{id}/start` \| `/pause` \| `/reset` | transport control |
| POST | `/api/scenario/{id}/speed` | set tick rate (`ticks_per_sec`) |
| GET | `/api/scenario/{id}/metrics/summary` | cumulative snapshot |
| GET | `/api/scenario/{id}/export?fmt=csv\|json` | run-history export |
| GET | `/api/data/tsrd/samples` | list cached TSRD samples for the picker |
| GET | `/api/scenario/{id}/band/{band}` | band detail: classification, priority breakdown, features |
| POST | `/api/scenario/{id}/priority-weights` | set Threat-Priority-Score weights (sort-only) |
| GET | `/api/ai/status` | AI Analyst availability (key present / model) |
| POST | `/api/ai/narrate-band` | 1–2 sentence narration for one band (cached per label) |
| POST | `/api/ai/chat` | scope-bounded operator chat over a scenario snapshot |
| POST | `/api/ai/summarize-run` | end-of-run written summary of the run |
| GET | `/api/health` | liveness |

WebSocket: `WS /ws/scenario/{id}` streams one JSON tick per timestep with, per
strategy, the `scanned_bands`, `hits`, `beliefs` (smart), and `metrics`, plus the
`ground_truth` vector and current `periodicity` estimates.

---

## Testing & scientific validation

Unit tests (no network / no HF credentials required  CI-safe):

```bash
cd backend
python -m pytest -q
```

Covers: the belief filter against a **hand-computed 3-step toy example**, Whittle
index **monotonicity** (both correlation regimes) + continuity + boundary values,
synthetic environment transition/period/hop statistics + observation-noise model,
Thompson posterior concentration, periodicity detection (including validation
against TSRD-derived periodic emitters), the TSRD cache pipeline + loader on a
small in-process fixture, and the **behaviour classifier** — feature extraction
and rule matching on hand-crafted periodic / hopping / steady / tightening bands,
plus priority-score/urgency ordering (`tests/test_classification.py`).

Headless comparison (asserts smart beats all baselines on both modes):

```bash
cd backend
python validate.py --ticks 4000
```

Representative result (reproduces the proposal's headline claims):

```
=== Synthetic (Dense) ===
  smart       rate=1.31  Pd=0.238  ...   smart / sequential = 2.71x
=== Real Data (TSRD cache sample 0) ===
  smart       rate=1.83  Pd=0.227  ...   smart / sequential = 2.04x
VALIDATION: PASS
```

---

## Project layout

```
smart-scan-ew/
├── backend/
│   ├── data/prepare_tsrd_cache.py   # offline: download/stub -> occupancy grid + stats
│   ├── data/cache/                  # generated NPZ/JSON (git-ignored)
│   ├── sim/                         # base_environment, synthetic_environment, tsrd_environment
│   ├── algo/                        # belief_filter, thompson, whittle_index, periodicity,
│   │                                #   baselines, reward, smart_scheduler
│   ├── classification/             # features, classifier (rule-based), priority_score, engine
│   ├── ai_analyst/                 # claude_client (graceful), prompts (scope-bounded)
│   ├── metrics/metrics.py
│   ├── api/                         # main (FastAPI), schemas, simulation orchestrator
│   ├── tests/                       # pytest suite
│   └── validate.py                  # headless smart-vs-baselines validation
├── frontend/
│   ├── app/                         # setup wizard (/) + dashboard (/dashboard)
│   ├── components/                  # SpectrumHeatmap, BeliefBarPanel, MetricsRow,
│   │                                #   ComparisonChart, PeriodicityRadar, ScenarioControls,
│   │                                #   DatasetAttributionFooter, TopMHighlightOverlay, AboutDrawer,
│   │                                #   AiAnalystPanel, BandDetailPopover, EndOfRunModal,
│   │                                #   ClassificationDisclaimer
│   └── lib/                         # ws-client, store (zustand), api, types
├── .env.example
└── README.md
```

---

## Stretch goals

Deliberately deferred (scaffolded interfaces make them additive, not blocking):

- **Neural Whittle (WIBQL)** PyTorch module + INT8 ONNX export + CPU latency
  benchmark to substantiate an edge-deployment claim.
- **AoA spatial dimension** (Phase-2): use the TSRD Angle-of-Arrival field to
  model spatially-scanning emitters with an angle-of-arrival belief dimension
  alongside frequency.
