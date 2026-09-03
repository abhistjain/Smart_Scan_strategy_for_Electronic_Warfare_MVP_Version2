# Project Report — Smart Scan Strategy for Electronic Warfare

**Problem Statement:** SIH26055 · ML-based Electronic Support (ES) Receiver Scheduler
**Repository:** `DRDO-SIH` (local: `f:\DRDO-SIH`)
**Report date:** 3 September 2026
**Backend version:** 0.1.0 · **Frontend version:** 0.1.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement and Requirement Mapping](#2-problem-statement-and-requirement-mapping)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Backend — Detailed Design](#5-backend--detailed-design)
6. [Frontend — Detailed Design](#6-frontend--detailed-design)
7. [Data Pipeline — Turing Synthetic Radar Dataset](#7-data-pipeline--turing-synthetic-radar-dataset)
8. [API Surface](#8-api-surface)
9. [Testing and Scientific Validation](#9-testing-and-scientific-validation)
10. [Project Statistics](#10-project-statistics)
11. [Configuration, Secrets and Deployment](#11-configuration-secrets-and-deployment)
12. [Scope Boundaries and Responsible-Use Design](#12-scope-boundaries-and-responsible-use-design)
13. [Known Limitations](#13-known-limitations)
14. [Future Work](#14-future-work)
15. [File Inventory](#15-file-inventory)

---

## 1. Executive Summary

This project is a full-stack demonstrator of an **ML-based Electronic Support (ES) receiver scheduler**. An ES receiver can only listen to a small number of frequency bands at once (capacity *M*) out of a much larger set (*N*). The scheduler's job is to decide, every time slot, *which M bands to scan* so that hostile emitters are intercepted as quickly and as often as possible — **with no prior intelligence** about where or when they transmit.

The core scheduler ("Smart") composes four well-founded techniques into one online policy:

| Component | Role | Reference |
| --- | --- | --- |
| **Bayesian belief filter** (2-state HMM) | Tracks P(band active) for every band, including unscanned ones | Standard HMM forward filter |
| **Thompson sampling** (Beta priors) | Learns unknown per-band transition probabilities online; sampling *is* the exploration | Thompson (1933); Beta-Bernoulli bandits |
| **Whittle index** (closed form) | Ranks all bands by scan priority each tick; provably indexable restless bandit | Liu & Zhao, IEEE Trans. IT, 2010, Thm. 2 |
| **Lomb-Scargle + von Mises periodicity** | Detects periodic (rotating-antenna) emitters from sparse hits and boosts the index just before the next predicted illumination ("intercept-ahead") | Lomb (1976), Scargle (1982) |

The Smart scheduler is benchmarked live against three baselines — **Sequential Sweep** (the open-loop incumbent), **Random**, and **Greedy Recent-Hit** — all running on the *byte-identical* hidden ground-truth trajectory. A headless validation run (3 000 ticks, seed 7) reproduces the headline result:

- **Synthetic (Dense):** Smart intercept rate **2.57×** the sequential sweep (1.229 vs 0.478 intercepts/tick).
- **Real-Data replay (TSRD cache):** Smart intercept rate **2.00×** the sequential sweep (1.789 vs 0.893).
- All **31 unit tests pass**.

Three optional presentation-layer add-ons sit on top: a **transparent rule-based emitter behaviour classifier** with a sortable **Threat Priority Score**; an **AI Scene Analogues** panel that maps each behaviour label to an illustrative analogue (fighter-like / UAV-like / missile-like / datalink-like), tallies the scene, and proposes a **recommended next scan** outside the receiver's current dwell; and an **AI Analyst** (Anthropic Claude) for plain-English narration, operator chat, and end-of-run summaries. All three are strictly sensing/presentation aids with a hard, code-enforced scope boundary against any weapon or engagement logic.

The deliverable is a Python/FastAPI backend streaming one JSON payload per tick over WebSocket to a Next.js "mission-control" dashboard.

---

## 2. Problem Statement and Requirement Mapping

SIH26055 asks for an ES receiver scheduler that performs a **2-D search (frequency × time)** against emitters whose behaviour is unknown a priori, in a **simulated RF environment with per-band, per-slot ground truth**, and that **beats the conventional open-loop sweep** on a defined set of figures of merit.

| Official requirement | Implementation | Location |
| --- | --- | --- |
| Right frequency at the right time (2-D search) | Top-M selection over belief-ranked Whittle indices every tick | `backend/algo/smart_scheduler.py` |
| Simulated RF environment with ground-truth status per band per slot | `SyntheticEnvironment` (Markov / periodic / hopper / quiet archetypes) and `TSRDEnvironment` (replayed occupancy grid) behind one `RFEnvironment` interface | `backend/sim/` |
| No prior reliable intelligence on emitters | `Beta(1,1)` priors on P01 / P10 per band; nothing about the environment is passed to the scheduler | `backend/algo/thompson.py` |
| Frequency-agile emitters | `hopper` archetype: pseudo-random hop sequence across a hop-set of 3–6 bands | `backend/sim/synthetic_environment.py` |
| Periodic / spatially-scanning emitters | `periodic` archetype (period, dwell, jitter, phase) + Lomb-Scargle intercept-ahead | `backend/algo/periodicity.py` |
| Robust ML-based scheduler minimising intercept time and maximising intercept rate | Whittle + belief + Thompson + UCB bonus composite | `backend/algo/` |
| Must beat the open-loop sweep | `SequentialSweep` baseline on the identical trajectory; asserted by `validate.py` | `backend/algo/baselines.py`, `backend/validate.py` |
| Figures of merit: Pd, Pfa, sensitivity, intercept rate, reward/cost, % correct, intercept time error | All seven computed online per strategy and streamed live | `backend/metrics/metrics.py` |
| Expected output: ES receiver scheduler software | This repository — backend scheduler + interactive dashboard | — |

---

## 3. System Architecture

```
                ┌──────────────────────────┐      ┌──────────────────────────────┐
                │  Synthetic Generator     │      │  TSRD Cache Loader           │
                │  (Markov/periodic/hopper)│      │  (occupancy grid from PDWs)  │
                └────────────┬─────────────┘      └──────────────┬───────────────┘
                             └──────────────┬─────────────────────┘
                                            ▼
                             ┌──────────────────────────────┐
                             │  RFEnvironment (abstract)    │  hidden truth  s_i(t) ∈ {0,1}
                             │  step() / observe(bands,rng) │  noisy obs     o_i(t) on scanned bands
                             └──────────────┬───────────────┘
                                            │
        ┌───────────────────────────────────┼────────────────────────────────────┐
        ▼                                   ▼                                    ▼
┌──────────────┐               ┌───────────────────────────┐           ┌──────────────────┐
│ Sequential   │               │  SMART SCHEDULER          │           │ Random / Greedy  │
│ Sweep        │               │  1. Thompson sample P01,P10│           │ baselines        │
└──────┬───────┘               │  2. Belief predict        │           └────────┬─────────┘
       │                       │  3. Whittle index ∀ bands │                    │
       │                       │  4. + UCB bonus           │                    │
       │                       │     + periodicity boost   │                    │
       │                       │  5. Top-M                 │                    │
       │                       └─────────────┬─────────────┘                    │
       └─────────────────────────────────────┼──────────────────────────────────┘
                                             ▼
                      ┌────────────────────────────────────────────┐
                      │  Simulation orchestrator (api/simulation.py)│
                      │  · one env, four strategies, same trajectory│
                      │  · MetricsAccumulator × 4                   │
                      │  · ClassificationEngine (add-on)            │
                      └────────────────────┬───────────────────────┘
                                           ▼
                      ┌────────────────────────────────────────────┐
                      │  FastAPI  ·  REST + WebSocket broadcaster  │
                      │  ScenarioRunner: async tick loop, N clients│
                      └────────────────────┬───────────────────────┘
                                           ▼  one JSON payload / tick
                      ┌────────────────────────────────────────────┐
                      │  Next.js 14 dashboard  (Zustand + Plotly)  │
                      │  heatmap · beliefs · metrics · radar · AI  │
                      └────────────────────────────────────────────┘
```

### Fairness guarantee

A single `RFEnvironment` instance owns the hidden ground-truth trajectory. It is advanced once per tick by its own seeded RNG and **does not depend on which bands any strategy scans**, so all four strategies see the same world. Each strategy is given its **own observation RNG** (`seed + 1000 + i`), so sensor noise (`p_miss`, `p_fa`) is independent per strategy — one strategy's lucky false-alarm draw is never forced onto another.

### Partial observability

Only scanned bands yield an observation. Unscanned bands evolve by the belief filter's prediction step alone. The scheduler never reads `ground_truth_state`; only the metrics engine and the dashboard's optional "Instructor Mode" overlay do.

---

## 4. Technology Stack

### Backend (Python 3.10+, developed on 3.11.9)

| Package | Version constraint | Purpose |
| --- | --- | --- |
| `fastapi` | ≥ 0.110 | REST + WebSocket server |
| `uvicorn[standard]` | ≥ 0.29 | ASGI server |
| `numpy` | ≥ 1.26 | Vectorised belief / index computation |
| `scipy` | ≥ 1.12 | `scipy.signal.lombscargle` periodogram |
| `pydantic` | ≥ 2.6 | Request/response schemas with validation |
| `huggingface_hub` | ≥ 0.23 | Gated TSRD download (offline preprocessing only) |
| `pyarrow`, `pandas` | ≥ 15.0, ≥ 2.2 | PDW parquet parsing (offline only) |
| `anthropic` | ≥ 0.39 | AI Analyst (optional, lazy-imported) |
| `python-dotenv` | ≥ 1.0 | Loads `.env` before env-dependent modules import |
| `pytest`, `httpx` | ≥ 8.0, ≥ 0.27 | Tests |

### Frontend (Node 18+, developed on 22.16.0)

| Package | Version | Purpose |
| --- | --- | --- |
| `next` | ^14.2.35 | App Router, rewrites `/api/*` → backend |
| `react`, `react-dom` | ^18.3.1 | UI |
| `typescript` | ^5.4.5 | Strict typing of every payload (`lib/types.ts`) |
| `tailwindcss` | ^3.4.4 | Styling; custom palette (cyan / amber / crimson), glassmorphism |
| `plotly.js-dist-min`, `react-plotly.js` | ^2.34.0, ^2.6.0 | Spectrum waterfall heatmap, comparison chart |
| `framer-motion` | ^11.0.0 | Page/panel transitions |
| `zustand` | ^4.5.2 | Single dashboard store fed by the WebSocket |

---

## 5. Backend — Detailed Design

### 5.1 Environment layer (`backend/sim/`)

#### `base_environment.py` — `RFEnvironment` (abstract)

- `reset()`, `step()`, `ground_truth_state`, `band_info()`, `duration` (None if endless).
- `observe(bands, rng)` — shared noisy observation model:
  - P(o=1 | ON) = 1 − `p_miss`
  - P(o=1 | OFF) = `p_fa`
- `BandInfo` dataclass: `index`, `label`, `emitter_type`, `stats` — static metadata for the UI.

#### `synthetic_environment.py` — `SyntheticEnvironment`

Every band is a hidden 2-state Markov chain with one of four archetypes assigned by a configurable **emitter mix** (default 45 % markov / 25 % periodic / 20 % hopper / 10 % quiet):

| Archetype | Dynamics | Parameters drawn from |
| --- | --- | --- |
| `markov` | Gilbert-Elliott chain; initialised at stationary distribution π_ON = P01 / (P01 + P10) | `p01_range`, `p10_range` |
| `periodic` | ON for `dwell` slots once every `period` slots, ± `jitter`, random `phase` | `periodic_period_range`, `periodic_dwell_range` |
| `hopper` | Member of a hop group; exactly one band in the group is ON at any time; group advances by a random non-zero step every `hop_dwell` ticks | `hopper_hopset_range`, `hopper_dwell_range` |
| `quiet` | Always OFF | — |

Parameter ranges are loaded from `data/tsrd_defaults.json` (produced by the TSRD preprocessing script) with a hard-coded fallback, so Synthetic mode is statistically shaped by real-data-derived defaults. `reset()` re-seeds the RNG and rebuilds the bands, so a scenario replays identically.

#### `tsrd_environment.py` — `TSRDEnvironment`

Loads `data/cache/tsrd_sample_<k>.npz` (an int8 occupancy grid of shape `T_slots × N_bands`) and its `.json` metadata, and replays it slot by slot (looping by default). The raw TSRD data carries no sensor noise, so the environment adds the receiver's own `p_miss` / `p_fa` via the shared `observe()` — documented explicitly so it is not confused with the dataset's native fidelity. Band labels become centre frequencies in MHz when band edges are present. `list_cached_samples()` returns metadata for the setup screen, or an empty list (which disables Real-Data mode gracefully).

### 5.2 Algorithm layer (`backend/algo/`)

#### `belief_filter.py` — `BeliefFilter`

Per band, b_i(t) = P(state_i = ON | observations).

- **Predict** (all bands, every tick): b ← b·(1 − P10) + (1 − b)·P01
- **Update** (scanned bands only, exact Bayes with the normaliser written out):
  - e1 = P(o | ON), e0 = P(o | OFF) from the observation model
  - b_post = e1·b / (e1·b + e0·(1 − b))
- Beliefs are clipped to [1e-9, 1 − 1e-9] to avoid degenerate fixed points.

#### `thompson.py` — `ThompsonTransitionLearner`

Independent `Beta(1,1)` priors on P01_i and P10_i for every band. Posteriors are updated only from **consecutive-scan transitions** (a band observed on two adjacent ticks yields one noisy transition sample). `sample()` draws (P01, P10) from the posteriors each decision epoch — this sampling is the sole exploration mechanism. `counts()` exposes the number of observed transitions per band for the UCB bonus. The module docstring documents that observations are noisy proxies for the true state and why the posteriors still concentrate correctly.

#### `whittle_index.py` — `WhittleIndexEngine`

Implements the **closed-form Whittle index for the 2-state restless bandit** from Liu & Zhao (2010), Theorem 2, discounted criterion with β = 0.99. Both the positively-correlated (p11 ≥ p01) and negatively-correlated (p11 < p01) regimes are implemented (eqs. 35 and 36), including the crossing-time L(x, ω) recursion, capped at 5 000 iterations. The index is computed for **every band each tick** (not just recently-hit ones) — the key structural advantage over Greedy. `add_exploration_bonus()` adds c·√(log(t+1) / (n_i + 1)) because transition probabilities are learned, not known.

#### `periodicity.py` — `PeriodicityDetector`

Per band, a rolling `deque` (64) of hit timestamps. When ≥ 6 hits span ≥ 3 ticks:

1. **Lomb-Scargle periodogram** over 240 candidate periods in [3, min(200, span)] — chosen because it is correct for irregular, sparse sampling (we only see a band on the ticks we chose to scan it). A median-gap fallback handles very short histories.
2. **von Mises circular fit** on hit phases → circular mean phase and concentration κ (Fisher's approximation from the mean resultant length R). R doubles as the confidence in [0, 1].
3. **Prediction** of the next active tick.

Outputs: `index_boost(t)` — an additive boost of `0.5 · confidence · proximity` applied within 2 ticks before the predicted window (only when confidence ≥ 0.3); `predicted_next_active()` for the intercept-time-error metric; `summary()` for the UI radar.

#### `baselines.py`

- `Strategy` (ABC): `select(t) → M band indices`, optional `observe_feedback(...)`.
- `SequentialSweep` — fixed round-robin cursor advancing M bands per tick. The literal open-loop incumbent.
- `RandomStrategy` — M bands uniformly without replacement.
- `GreedyRecentHit` — re-scan the M most-recently-hit bands; fill spare capacity round-robin.

#### `reward.py`

reward(t) = R_hit · hits − C_dwell · M − C_miss_penalty · (truly-active bands not scanned). Defaults: 1.0 / 0.05 / 0.5. The miss-penalty term is ground-truth-only scoring, never visible to the scheduler.

#### `smart_scheduler.py` — `SmartScheduler`

Composes the above. Per tick:

```
select(t):
  (P01, P10) ← thompson.sample()
  b_pred     ← belief.predict(P01, P10)
  idx        ← whittle.index_array(b_pred, P01, P10)
  idx       += UCB bonus(t, thompson.counts())
  idx       += periodicity.index_boost(t)
  return argpartition top-M, ordered by index

observe_feedback(scanned, obs, t):
  belief.update(scanned, obs)
  thompson.update(scanned, obs)
  periodicity.record_hit(...) for hits; periodicity.update(t)
```

Flags `use_thompson` and `use_periodicity` allow ablation.

### 5.3 Metrics (`backend/metrics/metrics.py`) — `MetricsAccumulator`

All figures of merit are accumulated online against ground truth, one accumulator per strategy:

| Metric | Definition |
| --- | --- |
| **Pd** | intercepted active cells / all truly-active (band, tick) cells |
| **Pfa** | false alarms / scans of truly-idle bands |
| **Sensitivity** | detections / scans of truly-active bands (= 1 − empirical miss rate; distinct from Pd) |
| **Intercept rate** | total intercepts / ticks |
| **Avg reward** | running mean of the reward function |
| **% correct** | fraction of Top-M scanned bands that were truly active |
| **Intercept time error** | MAE between predicted and actual next OFF→ON onset (Smart only; from periodicity predictions) |

Rolling histories (`cumulative_intercepts`, `intercept_rate_history`, `reward_history`) feed the charts.

### 5.4 Simulation orchestrator (`backend/api/simulation.py`)

`ScenarioConfig` (dataclass) captures every knob: data source, N, M, seed, noise, emitter mix, TSRD sample id, reward weights, β, UCB c, and priority weights. `Simulation.step()`:

1. Advances the environment once.
2. Detects OFF→ON onsets and scores Smart's intercept-time error against its previous predictions.
3. For each strategy: `select` → `observe` (own RNG) → reward → metrics update → `observe_feedback`.
4. Refreshes Smart's next-active predictions.
5. Feeds the `ClassificationEngine` with **Smart's observations only** (operator-available data).
6. Emits the tick payload and appends a compact metrics-only history row for export.

### 5.5 Emitter behaviour classification (`backend/classification/`) — add-on

A presentation-layer classifier operating **exclusively on operator-available information** (Smart's scans/observations, the belief vector, and the periodicity output — never hidden ground truth).

- **`features.py` — `FeatureExtractor`**: rolling 160-tick buffers per band → `BandFeatures`: `evidence`, `duty_cycle`, `onset_rate`, `periodicity_strength`, `period`, `period_trend` (linear-fit slope of recent periods; negative = tightening), `bandwidth` (neighbour co-activity from beliefs), `hop_rate` (fraction of active observations that are fresh onsets), plus amplitude/pulse-width proxies from static hints.
- **`classifier.py` — `classify()`**: a five-rule, ordered, fully transparent decision table. Every result carries a `matched_rule` string.

| # | Label | Rule (abridged) |
| --- | --- | --- |
| 1 | Tightening-Pattern Emission | some_period ∧ period_trend < −0.05 ∧ duty < 0.6 |
| 2 | Scanning/Rotating-Pattern Emission | strong_period ∧ 0.03 ≤ duty ≤ 0.6 ∧ bandwidth < 0.5 |
| 3 | Frequency-Agile / Hopping Emission | duty < 0.35 ∧ hop_rate ≥ 0.6 ∧ ¬strong_period |
| 4 | Steady Comms-Like Emission | duty ≥ 0.6 ∧ onset_rate < 0.2 ∧ ¬some_period |
| 5 | Unclassified | fallback / evidence < 6 |

  Confidence scales with an evidence factor (saturating at 40 scans). A period ≤ 1.5 ticks is treated as "always on", not a scan cadence. Each label also maps to a clearly-marked **illustrative scene analogue** via `ANALOGUE_META` (see §6.4), serialised as `analogue_*` fields on every `Classification`.
- **`priority_score.py`**: Threat Priority Score = w_belief·belief + w_conf·confidence + w_urgency·urgency, where urgency = 1 − (ticks to predicted next window)/20, clipped to [0, 1]. Returns a term-by-term breakdown for the UI. Weights default to 0.5 / 0.2 / 0.3 and are live-adjustable. **The score only sorts the dashboard.**
- **`engine.py` — `ClassificationEngine`**: re-classifies every 12 ticks with hysteresis (a different, weaker label is rejected unless within 0.1 confidence) so chips do not flicker; recomputes priority every tick.

### 5.6 AI Analyst (`backend/ai_analyst/`) — add-on

- **`claude_client.py`**: thin wrapper around `anthropic.Anthropic`. Lazy import; reads `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` (default `claude-3-5-haiku-latest`) at instantiation. If the key is absent or any call fails, returns `AIResult(available=False, text="AI narration unavailable", error=...)` — the app never breaks.
- **`prompts.py`**: `SCOPE_SYSTEM` system prompt encoding the hard scope boundary (never select/recommend/name weapons; never compute trajectories/launch envelopes/engagement timing; decline and redirect), plus formatting rules for the UI's light-markdown renderer. Three prompt builders: `build_narrate_prompt` (1–2 sentences per band), `build_chat_prompt` (question + compact JSON snapshot), `build_summary_prompt` (2–3 paragraph run summary).
- Band narration is **cached per (band, label)** in `ScenarioRunner.narration_cache` and regenerated only when the label changes or `force=true`.

### 5.7 API server (`backend/api/main.py`)

- Loads `.env` from the repo root and `backend/.env` **before** importing env-dependent modules.
- `ScenarioRunner` holds one `Simulation`, a status (`paused | running | finished`), `ticks_per_sec` (default 8, max 200), a set of WebSocket clients, and an `asyncio` tick loop that broadcasts each payload and drops dead sockets.
- In-memory `SCENARIOS` registry keyed by a 12-hex-char id.
- CORS is wide-open (`*`) — appropriate for a local demo, not for production.

---

## 6. Frontend — Detailed Design

### 6.1 Pages (`frontend/app/`)

| Route | File | Purpose |
| --- | --- | --- |
| `/` | `page.tsx` | **Setup wizard** over an animated radar-sweep backdrop. Data-source toggle (Synthetic / Real Data · TSRD — disabled with "(cache empty)" when no samples exist). Synthetic sliders: N 8–64, M 1–⌊N/2⌋, seed 0–999, P(miss) 0–0.5, P(fa) 0–0.3, and the four-way emitter mix. Four quick presets: *Sparse & Quiet* (24/3), *Dense & Noisy* (32/4), *Heavy Frequency-Hopping* (28/3, 55 % hoppers), *Periodic Radar Field* (24/3, 60 % periodic). TSRD tab: sample picker (bands · slots · duration · emitters · sub-band MHz, with a STUB badge), plus P(miss), P(fa), M 1–12. A "Strategies compared" legend. "Launch Simulation →" POSTs the scenario and routes to `/dashboard?id=…&src=…`. |
| `/dashboard` | `dashboard/page.tsx` | **Mission control.** Reads the scenario id from the URL (redirects to `/` if absent), connects `WSClient`, auto-starts the scenario, and lays out all panels in a 1fr + 320 px grid. Quick-action buttons open the Run Report and AI Analyst. Optional 880 Hz Web Audio blip on each Smart hit (off by default). |

### 6.2 State (`frontend/lib/`)

- **`store.ts`** (Zustand): scenario metadata, connection state, a 200-tick rolling window of belief / truth / scanned vectors for the heatmap, a 600-tick window of per-strategy intercept-rate history for charts, latest periodicity and classification arrays, selected band, and UI toggles (`instructorMode`, `highContrast`, `soundOn`, `speed`). Status is driven only by explicit control actions so an in-flight tick cannot flip a paused indicator back to running.
- **`types.ts`**: complete TypeScript mirrors of every backend payload (`TickPayload`, `StrategyMetrics`, `Classification`, `BandDetail`, `PeriodicityEstimate`, …), strategy labels/colours, and the canonical `CLASSIFICATION_DISCLAIMER` string.
- **`ws-client.ts`**: `WSClient` with `onInit` / `onTick` / `onStatus` callbacks and **automatic exponential-backoff reconnect** (starts at 500 ms, ×1.8 per attempt, capped at 8 s; reset on a successful open) so the dashboard survives a backend restart without a page reload. Malformed frames are ignored.
- **`api.ts`**: typed fetch helpers for every REST endpoint. All calls are same-origin (`BASE = ""`) and reach the backend through the Next.js rewrite. `fetchTSRDSamples()` and `aiStatus()` fail soft (empty list / `unreachable`) so a missing backend degrades the UI instead of throwing.

### 6.3 Components (`frontend/components/`)

| Component | Function |
| --- | --- |
| `ScenarioControls` | Top bar: "← setup" back link, data-source badge (Synthetic / Real · TSRD), pulsing live/paused indicator (shows "connecting…" until the socket is up), tick counter, ▶ start / ❚❚ pause / ↺ reset, speed slider (1–30 ticks/s), toggles for **instructor** mode, high **contrast**, and hit **blip**, a one-click **CSV export** link, and the About button |
| `SpectrumHeatmap` | Plotly waterfall of Smart's belief vector over the last 200 ticks on a five-stop colourscale (near-black → slate → amber at 0.7 → cyan at 1.0). The current Top-M cells are drawn as hollow cyan squares on the newest column; Instructor Mode overlays ground truth as small red diamonds. Uses `datarevision` for incremental redraws |
| `TopMHighlightOverlay` | Badge strip of the M bands Smart is scanning this tick; badges that produced a hit glow and gain a ✓ |
| `BeliefBarPanel` | Live per-band belief bars **sorted by Threat Priority Score**. Each row: band label, emitter-type glyph (◈ static · ⟳ periodic · ⇄ hopper · ◉ TSRD-derived · quiet), animated bar coloured by belief (cyan > 0.66, amber > 0.40, else idle slate), behaviour chip with rule tooltip, numeric belief. Top-M rows are ringed and pulse. Click → band detail |
| `MetricsRow` / `MetricsCard` | The **Smart scheduler's** seven figures of merit (Pd, Pfa, Sensitivity, Intercept Rate, Avg Reward, % Correct, Time Error), each an animated counter with a hover tooltip defining the metric. Per-strategy comparison lives in `ComparisonChart` and the End-of-Run table |
| `ComparisonChart` | Intercept-rate curves for the four strategies on the shared trajectory; hosts `SceneAnaloguePanel` in its right half |
| `SceneAnaloguePanel` | **AI Scene Analogues** — recommended-next-scan card, analogue tally, top-5 ranked bands (see §6.4) |
| `PeriodicityRadar` | **Polar plot** of the von Mises fits: each periodic emitter with confidence ≥ 0.3 is a wedge whose angle is the expected active phase (phase_mean / period × 360°), whose radius is the confidence, and whose angular width shrinks as κ grows (spread = max(6°, 40° / (1 + κ))). Hover shows band, period, confidence |
| `BandDetailPopover` | Behaviour chip + analogue chip, AI narration (with instant local fallback), matched rule, priority breakdown, full feature vector for one band |
| `AiAnalystPanel` | Right-hand slide-over. Header shows AI status (`online · <model>` or `offline · <reason>`). **Priority-weight sliders** (belief / confidence / urgency, debounced 250 ms to the API). Chat with three starter prompts — two in-scope and one deliberate **out-of-scope probe** ("Which missile should intercept the top band?") so judges can watch the refusal live. Footer: "Powered by Claude · sensing & classification only" |
| `EndOfRunModal` | Full-width report: a **four-strategy metrics table** (intercept rate, avg reward, % correct, total intercepts), an embedded `ComparisonChart`, the AI-generated written summary, and the disclaimer with the note "not an operational assessment" |
| `FormattedAIText` | Dependency-free renderer for the AI's light markdown: blank-line paragraphs, `-`/`*`/`•` bullets, `**bold**`, `` `code` ``. Also detects inline "Key: value - Key: value" attribute runs and splits them into a bullet list so replies stay scannable |
| `ClassificationDisclaimer` | Persistent disclaimer in two variants: `icon` (an "i" button with hover/click tooltip) placed beside every label or score, and `footer` (a fixed bottom strip prefixed "⚠ Illustrative behaviour patterns only") on the dashboard |
| `AboutDrawer` | "How it works" slide-over for judges: the problem, the four algorithms in plain language, why it wins, and the Real-Data mode |
| `DatasetAttributionFooter` | TSRD citation, Apache-2.0 licence, dataset link, and a note that Stare-mode data provides ground truth while our receiver adds its own P_miss / P_fa. **Rendered only in Real-Data mode** |
| `Plot` | Dynamic import wrapper for `react-plotly.js` (SSR-safe) |

### 6.4 AI Scene Analogues (`SceneAnaloguePanel`)

A competition-demo panel that translates the behaviour-pattern labels into a one-glance "scene picture" for an operator and proposes where to look next. It is **attention-only**: it sorts and highlights; nothing downstream consumes it.

#### Backend mapping — `ANALOGUE_META` in `classification/classifier.py`

Each behaviour label is paired with a presentation analogue. The mapping is a static table; no additional inference is performed.

| Behaviour label | Analogue key | Chip | Glyph | Colour | Rationale shown in UI |
| --- | --- | --- | --- | --- | --- |
| Scanning/Rotating-Pattern Emission | `fighter` | Fighter-like | ✈ | `#22D3EE` | Regular scan cadence and moderate duty — resembles a rotating airborne search pattern in general EW literature |
| Frequency-Agile / Hopping Emission | `uav` | UAV-like | ⬡ | `#F5A623` | Low duty and high hop rate — resembles a frequency-agile small-platform emitter |
| Tightening-Pattern Emission | `missile` | Missile-like | ◆ | `#EF4444` | Detected period is shrinking — resembles a scan becoming more frequent (behaviour only, not a lock-on) |
| Steady Comms-Like Emission | `comms` | Datalink-like | ▣ | `#34D399` | High duty and stable frequency — resembles a continuous comms or datalink emission |
| Unclassified | `unknown` | Unknown | · | `#64748B` | Not enough clean evidence for an analogue |

`Classification.to_dict()` serialises these as `analogue_key`, `analogue_short`, `analogue_title`, `analogue_why`, `analogue_glyph`, `analogue_color`, so they ride along in every tick's `classification[]` array and in `GET /api/scenario/{id}/band/{band}` with no extra request.

#### Panel layout (right half of the Strategy Comparison card)

1. **Header** — "AI scene analogues", the disclaimer icon, and `t=<tick> · demo only`.
2. **Recommended next scan** card — the single highest-priority band the receiver is **not already dwelling on** this tick (falls back to the top band if every high-priority band is in Top-M). Shows the band label, analogue glyph + chip, priority score, and a status tag (`in Top-M this tick` / `not in current dwell`). Below it, a one-line **scan reason** built by `scanReason()` from operator-visible signals only:
   - current belief (always),
   - "high hop-rate agility" for UAV-like / Agile-Hopping,
   - "tightening period trend" for Missile-like / Tightening-Pattern,
   - "regular scan cadence" for Fighter-like,
   - "next window ~t=N" when urgency > 0.45 and a periodicity prediction exists, else "high intercept-ahead urgency".

   Clicking the card opens the band's detail popover. Until enough evidence exists the card reads "Waiting for enough scan evidence to propose a next look…".
3. **Tally chips** — one chip per analogue present in the scene (`✈ Fighter-like ×3`, `⬡ UAV-like ×2`, …), sorted by count, `unknown` excluded.
4. **Top-5 list** — bands with a known analogue, ranked by Threat Priority Score; each row shows band label, glyph, analogue chip, and the underlying behaviour label. Click → detail popover.
5. **Footer** — "Next-scan is attention-only (belief / hop rate / tightening). Not a fire recommendation."

#### Where else the analogue surfaces

- **`BandDetailPopover`** renders the analogue chip beside the behaviour chip, and its instant local narration (shown while Claude responds, or permanently if the AI is unavailable) names the analogue explicitly and ends with the standing "illustrative signal-behaviour analogue — not a real platform identification" sentence.
- **`BeliefBarPanel`** chips use the behaviour label; the analogue is one click away.

#### Why the recommendation is not a scheduler override

The Smart scheduler's Top-M is decided entirely by the Whittle index (§5.2). The "Recommended next scan" card is computed client-side from the priority score and deliberately prefers a band *outside* the current dwell so the operator sees a complementary suggestion, not a duplicate of what the receiver is already doing. It has no feedback path into `SmartScheduler.select()`.

### 6.5 Design system ("SIGINT ops room")

Defined in `tailwind.config.ts`, `globals.css`, and `layout.tsx`.

| Token | Value | Meaning |
| --- | --- | --- |
| `base` | `#0A0E17` | Page background (near-black navy) |
| `panel` | `#0F1524` | Panel fill |
| `cyan` | `#22D3EE` | Hits / active / Smart scheduler |
| `amber` | `#F5A623` | Uncertain / warning / Sequential sweep |
| `slateidle` | `#3B4A6B` | Idle belief |
| `crimson` | `#EF4444` | Reserved for confirmed high-priority (Tightening-Pattern) only |
| Random / Greedy | `#A78BFA` / `#F472B6` | Baseline line colours |

- **Typography:** Inter (sans) and JetBrains Mono (mono) loaded via `next/font/google` as CSS variables; all telemetry uses mono for tabular alignment.
- **Surfaces:** `.glass` / `.glass-strong` — translucent panels with 12–16 px backdrop blur and 1 px hairline borders; `.grid-bg` — a faint 40 px cyan grid texture behind everything; `.radar-sweep` — a conic-gradient sweep rotating every 6 s on the setup page.
- **Motion:** Tailwind keyframes `pulse-glow` (1.2 s cyan halo on Top-M and hits), `shimmer`, and `sweep`; Framer Motion for page/panel/drawer transitions and animated bars and counters.
- **Accessibility / projection:** `.high-contrast` mode (toggle in the top bar) raises panel opacity to 0.95 and brightens borders for projectors; the hit blip is opt-in; every disclaimer icon has an `aria-label`.
- **Page metadata:** title "Smart Scan Strategy for EW — ES Receiver Scheduler", description referencing SIH26055.

### 6.6 Wiring

`next.config.js` (with `reactStrictMode: true`) rewrites `/api/*` to `NEXT_PUBLIC_BACKEND_HTTP` (default `http://127.0.0.1:8000`); the dashboard opens its WebSocket to `NEXT_PUBLIC_BACKEND_WS` (default `ws://127.0.0.1:8000`, automatically `wss://` when served over HTTPS).

---

## 7. Data Pipeline — Turing Synthetic Radar Dataset

**Dataset:** Turing Synthetic Radar Dataset (TSRD), Alan Turing Institute — Gunn, Hosford, Jones, Zeitler, Groves, Nockles, *"The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving."* Apache-2.0. Gated on Hugging Face (~70 GB, ~4 billion pulses).

`backend/data/prepare_tsrd_cache.py` is a **one-time offline** script with two modes that share an identical binning pipeline:

```
python backend/data/prepare_tsrd_cache.py --real --num 5 --n-bands 24 --slot-us 2000   # gated HF download
python backend/data/prepare_tsrd_cache.py --stub --num 3 --n-bands 24 --slot-us 2000   # no network; realistic fake PDWs
```

Pipeline steps:

1. Download (or synthesise) a handful of **Stare-mode** pulse trains — Stare mode observes the full 0–18 GHz spectrum simultaneously and is the closest analogue to "ground truth per band per slot".
2. Parse 5-D PDWs: time of arrival, centre frequency, pulse width, angle of arrival, amplitude (column names are CLI-overridable).
3. Select the busiest 500 MHz sub-band; discretise into N bands.
4. Discretise time into `slot_us` slots.
5. Mark cell (slot, band) = 1 if any pulse's [ToA, ToA + PW] overlaps the slot → **occupancy grid**.
6. Extract per-emitter-cluster statistics — explicitly *not* a deinterleaving solution:
   - `_cluster_emitter_stats()`: per band with ≥ 4 pulses, dominant PRI = median inter-pulse gap (robust to missed pulses), `dominant_period_slots` = PRI / slot, duty cycle = Σ pulse width / span, centre frequency.
   - `_hop_stats()`: hop rate = fraction of consecutive pulses whose frequency jumps > 5 MHz; hop range = f_max − f_min.
7. Write `cache/tsrd_sample_<k>.npz` (compressed int8 grid) + `.json` (band edges, slot width, emitter stats, hop stats, `aoa_available`, `stub` flag), and aggregate 20th–80th-percentile ranges to `data/tsrd_defaults.json` for the synthetic generator.

CLI: `--real | --stub` (mutually exclusive, required), `--num` (default 2), `--n-bands` (24), `--slot-us` (2000), `--seed` (0).

**Real-mode download** (`load_real_pdws`) is deliberately defensive because the gated repo's layout is not publicly documented: it lists repo files, prefers Stare-mode test-split tabular files (`.parquet` / `.csv` / `.h5`), downloads only `--num` of them, and maps columns through alias lists (`toa` / `time_of_arrival` / `t`, `frequency` / `cf` / `centre_frequency`, …). Any failure raises and the driver falls back to `--stub` with a printed notice.

**Stub PDW generator** (`make_stub_pdws`) synthesises a realistic 5-D pulse stream over a 9000–9500 MHz slice and a 1 s window: four periodic emitters with clean PRIs (2–8 ms, 1 % jitter) and slowly drifting AoA to mimic a rotating antenna; two frequency hoppers (PRI 0.5–1.5 ms, hop-sets of 4–6 frequencies); and three steady/noise-like emitters with random ToA and frequency. It runs through the identical `build_occupancy_grid` pipeline, so stub and real caches are indistinguishable to the app apart from the `"stub": true` flag and the STUB badge in the UI. The dataset is never a hard runtime dependency; without a cache the app runs in Synthetic mode only.

**Current repository state:** three cached samples (`tsrd_sample_0..2`) are present and are **stub-generated** (`tsrd_defaults.json` reports `"source": "tsrd_stub"`). The "Real Data" figures in this report therefore validate the replay path and pipeline, not the gated dataset itself.

---

## 8. API Surface

### REST

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/scenario` | Create scenario → `{ scenario_id, name, n_bands, m, band_info, config }` |
| `GET` | `/api/scenario/{id}` | Config, status, tick, band info |
| `POST` | `/api/scenario/{id}/start` · `/pause` · `/reset` | Transport control |
| `POST` | `/api/scenario/{id}/speed` | `{ ticks_per_sec }` (0 < x ≤ 200) |
| `GET` | `/api/scenario/{id}/metrics/summary` | Cumulative per-strategy snapshot |
| `GET` | `/api/scenario/{id}/export?fmt=csv\|json` | Per-tick metrics history. CSV columns are `t` then `<strategy>.<metric>` for all four strategies (e.g. `smart.pd`, `sequential.intercept_rate`); served as an attachment `scenario_<id>.csv`. JSON returns `{ scenario_id, history[] }` |
| `GET` | `/api/data/tsrd/samples` | Cached TSRD samples for the picker |
| `GET` | `/api/health` | Liveness + scenario count |
| `GET` | `/api/scenario/{id}/band/{band}` | Band detail: classification, priority breakdown, features, periodicity |
| `POST` | `/api/scenario/{id}/priority-weights` | Set `w_belief`, `w_conf`, `w_urgency` (sort-only) |
| `GET` | `/api/ai/status` | `{ available, model, reason }` |
| `POST` | `/api/ai/narrate-band` | 1–2 sentence narration (cached per label; `force` bypasses) |
| `POST` | `/api/ai/chat` | Scope-bounded operator chat over a scenario snapshot |
| `POST` | `/api/ai/summarize-run` | End-of-run written summary |

### WebSocket — `WS /ws/scenario/{id}`

On connect the server sends an `init` message (n_bands, m, band_info, config, tick, status) and the latest tick if one exists. Thereafter, one JSON payload per tick:

```json
{
  "t": 412,
  "data_source": "synthetic",
  "strategies": {
    "smart":      { "scanned_bands": [...], "hits": [...], "metrics": {...}, "beliefs": [...], "index": [...] },
    "sequential": { "scanned_bands": [...], "hits": [...], "metrics": {...} },
    "random":     { ... },
    "greedy":     { ... }
  },
  "ground_truth": [0, 1, 0, ...],
  "periodicity":  [ { "band": 7, "period": 24.1, "confidence": 0.82, "next_active_tick": 430.0, ... } ],
  "classification": [ { "band": 7, "label": "...", "confidence": 0.71, "priority": 0.64, "matched_rule": "...", ... } ]
}
```

Inbound control commands (`start`, `pause`, `reset`, `speed`) are also accepted over the socket.

---

## 9. Testing and Scientific Validation

### 9.1 Unit tests — 31 tests, all passing

```
cd backend
python -m pytest -q
...............................  [100%]
```

| File | Tests | What is asserted |
| --- | :-: | --- |
| `test_belief_filter.py` | 3 | A **hand-computed 3-step toy example** (predict → Bayes update on o=1 → update on o=0 → prediction-only step) matches to 1e-6; a noiseless positive observation drives belief > 0.999; an unscanned band relaxes to the stationary distribution P01/(P01+P10) |
| `test_whittle_index.py` | 6 | Index is **monotonically non-decreasing in belief** over 501 points for four positively-correlated and three negatively-correlated (P01, P10) pairs; W(0)=0 and W(1)=1; no jump > 0.02 across the region boundaries p01 / ω_o / p11; the vectorised engine equals the scalar form; the index scales linearly with the reward weight |
| `test_thompson.py` | 3 | Posterior means converge to the true (P01, P10) within 0.03 / 0.05 after 20 000 noiseless transitions; successive samples under the uniform prior differ; `counts()` increments on observed transitions |
| `test_periodicity.py` | 3 | A clean period-20 hit train is recovered within ±2 ticks with confidence > 0.8; the index boost is positive one tick before the predicted window; a period is recovered from the busiest band of a **stub-TSRD occupancy grid** |
| `test_synthetic_environment.py` | 5 | Identical seeds give identical trajectories over 200 steps; empirical Markov transition frequencies match configured P01/P10 within 0.03 over 60 000 steps; periodic inter-window gaps match the configured period within ±3; exactly one band per hop group is ON at every step; the observation model yields 1−p_miss and p_fa within 0.03 over 40 000 draws |
| `test_tsrd_cache.py` | 4 | Occupancy-grid overlap logic on a two-pulse hand example (pulse spanning two slots, pulse in one slot, nothing else); stub pipeline produces a valid int8 grid with emitter stats; `list_cached_samples()` returns a list; a tiny fixture written to the cache dir loads, steps, loops, and exposes `band_info()` |
| `test_classification.py` | 7 | Hand-crafted steady / scanning / hopping / tightening bands each produce the expected label with sane features; < 6 scans → Unclassified; sooner predicted window → higher urgency and the priority breakdown carries all three terms; every classification exposes a non-empty `matched_rule` |
| **Total** | **31** | |

`conftest.py` puts `backend/` on `sys.path` so tests run without installation. No network or credentials are required; the suite is CI-safe.

### 9.2 Headless validation — `validate.py --ticks 3000` (seed 7, N = 24, M = 3, p_miss = 0.1, p_fa = 0.05)

**Synthetic (Dense)**

| Strategy | Intercept rate | Pd | Pfa | Sensitivity | Avg reward | % correct | Time error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Smart** | **1.229** | **0.225** | 0.048 | 0.906 | **−0.890** | **0.452** | 3.36 |
| Sequential | 0.478 | 0.088 | 0.054 | 0.906 | −2.001 | 0.176 | — |
| Random | 0.617 | 0.113 | 0.051 | 0.903 | −1.798 | 0.228 | — |
| Greedy | 0.526 | 0.097 | 0.050 | 0.903 | −1.939 | 0.194 | — |

Smart / Sequential intercept-rate ratio = **2.57×**

**Real Data replay (TSRD cache sample 0 — stub-generated)**

| Strategy | Intercept rate | Pd | Pfa | Sensitivity | Avg reward | % correct | Time error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Smart** | **1.789** | **0.222** | 0.043 | 0.902 | **−1.357** | **0.661** | 26.02 |
| Sequential | 0.893 | 0.111 | 0.050 | 0.895 | −2.690 | 0.332 | — |
| Random | 0.878 | 0.109 | 0.049 | 0.896 | −2.715 | 0.326 | — |
| Greedy | 0.965 | 0.120 | 0.047 | 0.898 | −2.588 | 0.358 | — |

Smart / Sequential intercept-rate ratio = **2.00×**

**Result: `VALIDATION: PASS`** — Smart beats every baseline on intercept rate in both modes.

Observations:

- **Sensitivity is ≈ 0.90 for all strategies** and **Pfa ≈ 0.05** — as expected, because these are receiver-level properties set by `p_miss` / `p_fa`, not by the scheduler. This confirms the metrics are correctly separated: the scheduler wins on *where it looks* (Pd, intercept rate, % correct), not by changing the sensor.
- **% correct** more than doubles (0.176 → 0.452 synthetic; 0.332 → 0.661 replay): Smart's Top-M choice is far more often pointed at a truly-active band.
- **Average reward** is negative for every strategy because the miss-penalty term counts all active bands not scanned (with M = 3 of 24, most active bands are unscanned by construction). Smart's reward is the least negative by a wide margin.
- **Intercept time error** exists only for Smart (the baselines make no predictions). The larger error on the replay cache reflects the stub's very short periods (5–10 slots) relative to the periodicity detector's 3-tick minimum and the looping replay boundary.

> Windows note: on some machines OpenBLAS fails to allocate its thread pool when spawned from a non-interactive shell. Setting `OPENBLAS_NUM_THREADS=1` before running `pytest` or `validate.py` resolves this with no effect on results.

---

## 10. Project Statistics

| Metric | Value |
| --- | --- |
| Python source (backend, excl. caches) | 3 261 lines across 38 files |
| TypeScript/TSX source (`app/`, `components/`, `lib/`) | 2 649 lines across 24 files |
| Backend packages | 7 (`sim`, `algo`, `metrics`, `api`, `data`, `classification`, `ai_analyst`) |
| Frontend components | 17 |
| REST endpoints | 16 · WebSocket endpoints: 1 |
| Unit tests | 31 (all passing) |
| Strategies compared | 4 (Smart + 3 baselines) |
| Figures of merit | 7 |
| Emitter archetypes (synthetic) | 4 |
| Behaviour-pattern labels | 5 |
| Runtime | Python 3.11.9 · Node 22.16.0 |

---

## 11. Configuration, Secrets and Deployment

### Environment variables (`.env.example`)

| Variable | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | No | Only for `prepare_tsrd_cache.py --real`. Read via `os.environ`; never written to disk. |
| `ANTHROPIC_API_KEY` | No | Enables the AI Analyst. Absent → panels show "AI narration unavailable". |
| `ANTHROPIC_MODEL` | No | Default `claude-3-5-haiku-latest`. |
| `NEXT_PUBLIC_BACKEND_HTTP` | No | Default `http://127.0.0.1:8000`. |
| `NEXT_PUBLIC_BACKEND_WS` | No | Default `ws://127.0.0.1:8000`. |

`.env`, `.env.local`, `*.local`, `backend/data/cache/*` (except `.gitkeep`), `.venv/`, `node_modules/`, `.next/`, `__pycache__/`, `.pytest_cache/`, and uvicorn/next log files are git-ignored.

### Running locally

```bash
# Backend
python -m venv .venv && .\.venv\Scripts\Activate.ps1          # or: source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/data/prepare_tsrd_cache.py --stub --num 3 --n-bands 24 --slot-us 2000
cd backend && python -m uvicorn api.main:app --port 8000

# Frontend
cd frontend && npm install && npm run dev                      # http://localhost:3000
```

The application runs **fully offline** in Synthetic mode. Real-Data mode activates automatically when cache files exist.

### Deployment posture

This is a demonstrator. Scenarios live in process memory (lost on restart), CORS is `*`, and there is no authentication. Hardening for multi-user or networked deployment would require a persistent scenario store, origin restrictions, and auth on the control endpoints.

---

## 12. Scope Boundaries and Responsible-Use Design

The classification and AI layers are deliberately constrained, and the constraint is enforced in code, prompts, and UI rather than by convention alone:

1. **No engagement logic exists anywhere in the codebase.** Nothing selects, recommends, names, or pairs a weapon, interceptor, or munition; nothing computes trajectories, guidance, launch envelopes, or engagement timing. The Threat Priority Score does exactly one thing — sort the belief panel.
2. **The AI system prompt (`SCOPE_SYSTEM`) instructs the model to decline** any out-of-scope request in one sentence and redirect to signal behaviour or scheduler performance. The AI Analyst panel ships with a deliberate out-of-scope starter prompt ("Which missile should intercept the top band?") so the refusal can be demonstrated live rather than asserted.
3. **Labels are behaviour patterns, not identities.** Every label, chip, analogue, and score is accompanied by the persistent disclaimer that these are illustrative analogues loosely inspired by general EW literature — not a validated IFF system and not real platform identification. The AI Scene Analogues panel is marked "demo only", its "Recommended next scan" card is labelled attention-only with an explicit "Not a fire recommendation" footer, and the analogue table never names a real airframe or munition.
4. **Explainability.** Every classification carries a `matched_rule` string; every priority carries a term-by-term breakdown; every AI response is grounded only in the JSON snapshot passed to it and is told not to invent numbers.
5. **The scheduler never sees ground truth.** Only the metrics engine and the opt-in Instructor Mode overlay read `ground_truth_state`.

---

## 13. Known Limitations

- **Real-Data mode currently runs on stub caches.** Building the genuine TSRD cache requires accepting the gated dataset's terms on Hugging Face and running `--real`; the pipeline is implemented but the checked-in samples are synthetic PDWs.
- **Whittle index computation is a Python loop over bands** (`index_array`). At N ≤ 128 this is negligible; at much larger N a vectorised or JIT implementation would be needed.
- **Thompson posteriors treat noisy observations as state transitions.** Documented in the module; posteriors still concentrate correctly under unbiased noise, but a fully rigorous treatment would marginalise over the observation model.
- **Intercept time error on short-period emitters is coarse** because the periodicity detector's minimum period is 3 ticks and the replay loops at the cache boundary.
- **Single-process, in-memory scenario store; open CORS; no auth** — appropriate for a hackathon demo, not production.
- **Angle of arrival is not modelled.** The TSRD AoA field is preserved in the cache metadata (`aoa_available`) but unused.
- **Frontend has no automated tests**; correctness relies on TypeScript's strict typing of the payload contract.
- **`frontend/images/DRDO logo.jpg` is not yet wired into the layout** — the asset exists but no component renders it.

---

## 14. Future Work

Scaffolded interfaces make these additive rather than disruptive:

- **Neural Whittle (WIBQL)** — a PyTorch module learning the index function end-to-end, with INT8 ONNX export and a CPU latency benchmark to substantiate edge deployment.
- **AoA spatial dimension (Phase 2)** — extend the belief state with an angle-of-arrival dimension using the TSRD AoA field, modelling spatially-scanning emitters in (frequency × angle × time).
- **Genuine TSRD cache** — run `prepare_tsrd_cache.py --real` with an HF token and re-validate against real Stare-mode pulse trains.
- **Ablation dashboard** — expose the `use_thompson` / `use_periodicity` flags in the UI to show each component's marginal contribution live.
- **Persistent scenarios and multi-user hardening** — database-backed scenario store, restricted CORS, authenticated control endpoints.
- **Frontend test suite** — component and store tests for the payload contract.

---

## 15. File Inventory

```
DRDO-SIH/
├── .env.example                      # documented env template (no secrets)
├── .gitignore
├── README.md                         # user-facing documentation
├── PROJECT_REPORT.md                 # this report
│
├── backend/
│   ├── pyproject.toml                # package metadata, pytest config
│   ├── requirements.txt
│   ├── validate.py                   # headless smart-vs-baselines validation
│   ├── algo/
│   │   ├── baselines.py              # Strategy ABC, SequentialSweep, Random, GreedyRecentHit
│   │   ├── belief_filter.py          # 2-state HMM forward filter
│   │   ├── periodicity.py            # Lomb-Scargle + von Mises intercept-ahead
│   │   ├── reward.py                 # R_hit / C_dwell / C_miss_penalty
│   │   ├── smart_scheduler.py        # composite Top-M policy
│   │   ├── thompson.py               # Beta posteriors on P01 / P10
│   │   └── whittle_index.py          # Liu & Zhao (2010) closed form, both regimes
│   ├── api/
│   │   ├── main.py                   # FastAPI REST + WebSocket, ScenarioRunner
│   │   ├── schemas.py                # Pydantic request/response models
│   │   └── simulation.py             # ScenarioConfig, Simulation orchestrator
│   ├── sim/
│   │   ├── base_environment.py       # RFEnvironment ABC, BandInfo, observe()
│   │   ├── synthetic_environment.py  # markov / periodic / hopper / quiet archetypes
│   │   └── tsrd_environment.py       # occupancy-grid replay, list_cached_samples()
│   ├── metrics/
│   │   └── metrics.py                # MetricsAccumulator (7 figures of merit)
│   ├── classification/
│   │   ├── features.py               # FeatureExtractor → BandFeatures
│   │   ├── classifier.py             # 5-rule decision table, labels, analogues, DISCLAIMER
│   │   ├── priority_score.py         # Threat Priority Score (sort-only)
│   │   └── engine.py                 # throttled, hysteretic ClassificationEngine
│   ├── ai_analyst/
│   │   ├── claude_client.py          # fail-safe Anthropic wrapper
│   │   └── prompts.py                # SCOPE_SYSTEM + 3 prompt builders
│   ├── data/
│   │   ├── prepare_tsrd_cache.py     # offline TSRD → occupancy grid pipeline (--real / --stub)
│   │   ├── tsrd_defaults.json        # aggregate ranges for the synthetic generator
│   │   └── cache/                    # tsrd_sample_{0,1,2}.{npz,json} (git-ignored; stub)
│   └── tests/                        # 31 pytest tests across 7 files
│
└── frontend/
    ├── package.json · tsconfig.json · next.config.js · tailwind.config.ts · postcss.config.js
    ├── app/
    │   ├── layout.tsx · globals.css
    │   ├── page.tsx                  # setup wizard
    │   └── dashboard/page.tsx        # mission-control dashboard
    ├── components/                   # 17 components (see §6.3)
    ├── lib/
    │   ├── api.ts · ws-client.ts · store.ts · types.ts
    ├── types/plotly-dist.d.ts        # module shim for plotly.js-dist-min
    └── images/DRDO logo.jpg          # branding asset (present; not yet referenced by any component)
```

---

*End of report.*
