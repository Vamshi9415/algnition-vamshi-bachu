# Compliance Check: AIgnition Brief + Hackathon Submission Guide

Checked against `AIgnition Project Brief 2026.pdf` and `hackathon-submission-guide.pdf`
(both in repo root), after the fixes in [audit_bugs_fixed.md](audit_bugs_fixed.md).
Submission deadline per the brief: **19 July 2026, 10:00 PM IST** — today is 2026-07-15.

## A. Submission Guide — hard requirements (Section 9 checklist)

| Requirement | Status | Notes |
|---|---|---|
| Repo on GitHub, public | ⚠️ Unverified | Can't check from here — confirm the GitHub repo (`Vamshi9415/algnition-vamshi-bachu` per README badge) is actually public before submitting. |
| `run.sh` at root, executable, runs end-to-end with one command | ✅ Fixed | Confirmed: `bash run.sh ./data/raw ./pickle/model.pkl ./output/predictions.csv` exits 0 and produces valid output on a clean pinned venv. |
| `run.sh` accepts `DATA_DIR MODEL_PATH OUTPUT_PATH` as **positional** args | ✅ Fixed | Previously only read env vars, silently ignoring the grader's positional invocation (see bugs doc #12). |
| `data/` folder exists, code reads from it dynamically | ⚠️ Path mismatch — see below | Code reads `data/raw/`, not `data/` directly. |
| Trained model committed under `pickle/` and loads cleanly | ❌ **Not committed** | See below — this is the single biggest open risk. |
| `requirements.txt` lists pinned versions | ✅ | All entries pinned; `openai==1.30.0` replaced with `google-genai==1.65.0`. |
| Output written to `OUTPUT_PATH` in announced format | ✅ | `date, channel, campaign_name, revenue_p10, revenue_p50, revenue_p90, confidence, pipeline_wmape, pipeline_picp, production_ready` — matches README. |
| Fresh output every run (no append) | ✅ | `df_out.to_csv(..., index=False)` overwrites. |
| No absolute paths | ✅ | All relative. |
| No interactive prompts | ✅ | |
| **No network calls at run time** | ⚠️ Caveat | Prophet's CmdStan backend isn't installed by `pip install prophet` alone and needs a network fetch to bootstrap — see below. Currently mitigated by a graceful fallback so this doesn't block the run, but it means Prophet never actually contributes to the forecast in a clean network-restricted environment. |
| Seeds set where randomness affects predictions | ⚠️ Not verified | LightGBM/Prophet aren't given explicit `random_state`/`seed` params anywhere in `config/forecasting.yaml` or the model constructors — re-running training could give slightly different numbers each time. Low risk (LightGBM is fairly stable) but worth adding for strict reproducibility. |
| Python version stated in README | ✅ | README says Python 3.11 is required (matches `requirements.txt` header comment). |

### ❌ Blocking: no trained `pickle/model.pkl` is committed

The guide is explicit: **"The model must be already trained and committed — we do
not retrain. The test run only generates features and predicts."** Before this audit,
`pickle/` contained only `.gitkeep` and a `model_stub.md` placeholder — no actual
`.pkl` file existed anywhere on disk or in git history. `run.sh` defaulted to
`SKIP_TRAIN=0`, meaning it would attempt to retrain from scratch on the grader's
held-out data every time — which conflicts with the guide's stated grading
assumption, and (until the fixes in this pass) crashed outright before producing any
model at all.

This audit trained a real model against the current sample data
(`pickle/model.pkl`, 8.3 MB, WMAPE=7.04%/PICP=75.23% — see
[audit_pipeline_results.md](audit_pipeline_results.md)) and verified `SKIP_TRAIN=1`
correctly loads and reuses it without retraining. **Before submitting:**
1. Decide whether to commit this trained `pickle/model.pkl` (delete it from
   `.gitignore`'s dead commented line if needed — it's currently *not* ignored, so a
   plain `git add pickle/model.pkl` will work) — the file must survive a `git clone`.
2. Given the guide replaces `data/` with held-out test data and does **not**
   retrain, whatever `pickle/model.pkl` you commit is what gets used to score
   every submission — retraining it against the final real sample data you intend to
   submit is worth doing last, right before submission.

### ⚠️ `data/` vs `data/raw/` path convention

The guide's default recommendation is `DATA_DIR=./data`, and it explicitly warns:
*"Replace the contents of your data/ folder with our held-out test data."* This
repo's actual convention — in the README, `run.sh`, and `src/cli/predict.py` — is
`data/raw/`. This isn't necessarily wrong (the guide says *"tell us if anything
differs"* when submitting the exact command), but it needs to be stated explicitly
in the submission form, or the grader may drop test data into `data/` (not
`data/raw/`) and get an empty-folder failure. **Recommend either**:
(a) restate the exact command as `./run.sh ./data/raw ./pickle/model.pkl
./output/predictions.csv` when submitting, or
(b) change the default `DATA_DIR` to `./data` and move the sample CSVs there, to
match the guide's default with zero ambiguity. (a) is lower-risk given how much else
already depends on `data/raw/`.

### ⚠️ Prophet / CmdStan network dependency

Confirmed in this audit: a fresh `pip install -r requirements.txt` does not produce a
working Prophet backend (`cmdstanpy` reports no CmdStan installation; building one
requires downloading CmdStan from GitHub). This is now handled gracefully (falls back
to LightGBM-only rather than crashing — see bugs doc #8), so it won't cause a `run.sh`
failure. But it does mean the ensemble is currently LightGBM-only unless the grading
environment happens to have CmdStan pre-installed. Worth a one-line callout in the
README so this isn't mistaken for a bug during review.

## B. Brief deliverables checklist

| Deliverable | Status | Notes |
|---|---|---|
| **1. Working prototype** | | |
| — ingest channel-level CSVs | ✅ | Google/Meta/Microsoft, auto-detected by filename + column signature. |
| — validate campaign consistency | ✅ | 12-check `DataValidator`, now actually passes on real data and writes its report (see bugs doc #10). |
| — accept future media budget inputs | ⚠️ Partial | `BudgetSimulator.simulate(forecast, spend_changes)` exists and works (verified via tests), but the live `/api/v1/simulate` FastAPI route (`src/api/routes/simulate.py`) is a **non-functional placeholder** that only echoes the request back — it never actually calls `BudgetSimulator`. No frontend UI component for budget simulation exists either (`frontend/src/components/` has no Budget/Simulate component). The capability exists in the backend library but isn't wired end-to-end. |
| — generate probabilistic revenue/ROAS forecasts | ✅ | P10/P50/P90 per campaign per day. |
| — channel-level / campaign-type / campaign-level outputs | ⚠️ Partial | Output is campaign-level; channel/campaign-type rollups aren't precomputed in `predictions.csv` but can be derived by grouping (channel is a column; campaign_type is not currently included in the output CSV, only in intermediate features). |
| — expected blended ROAS | ❌ Not in output | `predictions.csv` has revenue P10/50/90, not a blended ROAS figure or range. ROAS is computed internally as a feature but not surfaced as a forecast output. |
| — AI-assisted causal summaries | ✅ (conditionally) | `InsightGenerator` produces exec summary / risk analysis / budget recommendations via Gemini — but only if `GEMINI_API_KEY` is set in the environment (gracefully degrades to a placeholder string otherwise, which is correct behavior but means "AI-assisted causal summaries" won't actually appear unless a key is configured for the demo/grading run). |
| — **aggregate-period forecasts (30/60/90 days), not daily** | ❌ Gap | Brief constraint: *"Forecasts should be aggregate-period forecasts rather than daily forecasts."* Current output is one row per campaign **per day** across the horizon (8,160 rows for 109 campaigns × ~60/75 days). `config/settings.yaml` declares `horizons: [30, 60, 90]` but nothing aggregates the daily P10/50/90 series into a single period total. This is the most direct brief-constraint deviation found. |
| **2. Technical documentation** | ✅ | `docs/business_problem.md`, `docs/feature_catalog.md`, `docs/limitations.md`, `docs/statistical_validation.md` are all present and substantive; methodology, assumptions, and limitations are documented (some claims — e.g. quantile-sorting mitigation, Prophet fallback — were previously aspirational/unimplemented; now actually true after this pass). |
| **3. Architecture overview** | ✅ | `docs/architecture.md` covers frontend/backend/pipeline/LLM stack, though it still needs the GPT-4o→Gemini text swapped in a couple of spots (done) and doesn't mention the CmdStan/network caveat. |
| **4. Demo workflow** | ⚠️ Partial | Ingestion, forecast generation, and (conditional) AI insights are demonstrable end-to-end via `run.sh`. Budget simulation is not demoable through the actual API/UI (see above) — only through direct Python calls to `BudgetSimulator`. `scripts/demo_run.py` exists as a scripted walkthrough — worth running once more to confirm it still works after today's fixes. |

## C. Evaluation criteria — quick self-assessment

- **Technical soundness**: Forecasting methodology is real and now actually
  executes (quantile LightGBM + walk-forward CV + calibration/PICP checks). Biggest
  soundness caveat: the future-frame is a frozen single-day snapshot, which produced
  a degenerate $0 forecast for an entire channel in this run (see results doc §5) —
  worth fixing before judges see it, since it undermines "appropriate handling of
  uncertainty" if a reviewer notices a channel forecasting exactly zero.
- **Practical relevance**: Multi-channel ingestion and statistical rigor (Ljung-Box,
  PICP, bootstrap CI, SHAP) are genuinely present and running, which is a real
  differentiator — but budget simulation (an explicit brief ask) isn't wired to the
  UI/API, weakening "operational usefulness."
- **AI integration**: Present and functional (now on Gemini), but silently
  degrades to placeholder text without an API key — make sure a key is configured for
  the actual grading/demo run, or the "AI-assisted causal summaries" deliverable
  won't visibly work.
- **Product thinking**: Frontend exists and builds cleanly (`npm run build`
  succeeds), covers upload/forecast/AI-insights views, but has no budget-simulation
  screen despite the brief explicitly asking for one.
- **Engineering quality**: Was the weakest area found in this audit — 16 pytest
  failures, several fully broken code paths on the pipeline's main execution route,
  dead/duplicate modules (see below), and CI that excludes the very test files that
  were broken (`.github/workflows/ci.yml`'s unit-test job runs only
  `test_detector.py test_validator.py test_cleaner.py test_feature_store.py
  test_evaluation.py` — never `test_integration.py`, `test_pipeline_integration.py`,
  `test_canonical.py`, `test_uncertainty.py`, or `test_budget_simulator.py`, which
  were exactly the files carrying real bugs). CI's smoke test also uses synthetic
  UI-export-shaped data rather than the real committed CSVs, so it never exercised
  the column-mapping bug either. Recommend widening CI to run the full suite and to
  smoke-test against the actual `data/raw/*.csv` files before submitting, so this
  class of "passes CI, fails on real data" issue doesn't recur.

## D. Dead / duplicate code found (cosmetic, but affects "code quality" scoring)

Not fixed in this pass (out of scope — these are unused, so fixing them has no
functional benefit, but a reviewer grepping the codebase will find them confusing):

- `src/pipeline.py` — a full duplicate orchestrator (`AIgnitionPipeline`), shadowed
  and made unreachable by the `src/pipeline/` package of the same name. References a
  nonexistent `LLMInsightGenerator` class (the real class is `InsightGenerator`) and
  a nonexistent `FeatureStore` import path — would not even import if it weren't
  already dead.
- `src/api/routers/` — a duplicate of `src/api/routes/`, not wired into
  `src/api/main.py` (which imports from `routes/`, not `routers/`). Also references
  `LLMInsightGenerator` (doesn't exist) and `ForecastEnsemble` (doesn't exist — the
  real class is `EnsembleForecaster`), so it wouldn't import if invoked.
- Both dead files would raise `ImportError` immediately if anyone tried to actually
  use them — recommend deleting `src/pipeline.py` and `src/api/routers/` entirely
  rather than leaving confusing, broken duplicates for a code reviewer to trip over.

## E. One question worth resolving before submission

The brief's **Resources** section lists the intended dataset as *"GA4 session
source/medium conversion data"* and *"Shopify conversion data"* — not raw
Google/Meta/Microsoft Ads campaign-performance exports. This repo's entire ingestion
layer (canonical schema, channel detection, feature engineering) is built around
per-platform ad campaign CSVs (`campaign_name`, `spend`, `clicks`, `impressions`,
`conversions`, `revenue`), which is what's actually shipped in `data/raw/` and what
this audit tested against. It's possible the "AIgnition_dataset" download link
(inaccessible from here — it's an embedded link in the PDF) provides data in this
same shape, or it's possible it's structured as GA4/Shopify exports with a different
schema entirely. **Worth confirming directly against the actual dataset download**
before submission — if the real held-out test data is GA4/Shopify-shaped rather than
ad-platform-shaped, the ingestion/canonical-mapping layer (and today's channel-mapping
fixes) may need to be revisited for that schema instead.
