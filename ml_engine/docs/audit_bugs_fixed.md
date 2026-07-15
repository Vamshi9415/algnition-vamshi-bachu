# Bugs Found & Fixed During This Audit

Found by actually installing the pinned `requirements.txt` into a clean Python 3.11
venv and running `pytest` + `run.sh` against the CSVs already committed in this repo
— i.e. exactly what the hackathon's automated grader does. Every item below was
empirically reproduced (not just read from code) before being fixed, and re-verified
after the fix. Ordered roughly by how early in the pipeline they'd block execution.

## Severity: blocks every run, regardless of data

### 1. LLM client crashed the entire pipeline if no API key was set
`InsightGenerator.__init__` called `OpenAI(api_key=os.getenv("OPENAI_API_KEY"))`
unconditionally. The `openai` SDK raises `OpenAIError` immediately on construction
if the key is `None` — before any API call is attempted, before any data is even
touched. Since `ForecastPipeline.__init__` constructs `InsightGenerator` up front,
**this crashed 100% of runs on any machine without `OPENAI_API_KEY` set**, which is
exactly the judge's environment.
**Fix:** switched provider to Gemini (per explicit request) and made client
construction fail soft — no key → `self.client = None` → `_call()` returns a
`"[LLM unavailable — set GEMINI_API_KEY...]"` placeholder instead of crashing.
Files: `src/llm/insights.py`, `requirements.txt`, `config/llm.yaml`,
`config/settings.yaml`, `.env.example`, `docs/architecture.md`, `README.md`.

### 2. `CanonicalSchemaBuilder.build_from_many()` crashed on every real call
Expected `item["df"]` / `item["channel"]` (dict-style access), but the only real
caller, `CSVLoader.load_many()`, returns a list of `LoadedFile` **dataclass**
instances, which aren't subscriptable. Every invocation via the actual orchestrator
(`pipeline.run()`) raised `TypeError: 'LoadedFile' object is not subscriptable` —
**this alone meant the live pipeline could not process a single CSV, synthetic or
real, before this fix.**
**Fix:** use attribute access (`item.df`, `item.channel.value`).
File: `src/canonical/schema.py`.

## Severity: crashes on the repo's own committed sample data

### 3. Google/Meta canonical column mapping didn't match the real sample CSVs
`GOOGLE_MAP`/`META_MAP` in `src/canonical/schema.py` (and `config/channels.yaml`)
assumed Google/Meta Ads **UI-export** column names (`Date`, `Cost`, `Conv. value`,
`Reporting starts`, `Amount spent (USD)`, …). The CSVs actually committed in
`data/raw/` use **API-style** column names instead
(`segments_date`, `metrics_cost_micros`, `metrics_conversions_value` for Google;
`date_start`, `spend`, `conversion` for Meta). Result, measured empirically before
the fix:

| Channel | Rows | `date` null | `spend` null | `revenue` null |
|---|---|---|---|---|
| google | 19,272 | 100% | 100% | 100% |
| meta | 3,417 | 100% | 0%* | 100% |
| microsoft | 2,873 | 0% | 0% | 0% |

*(Meta's `spend`/`clicks`/`impressions` only survived by coincidence — those raw
column names happened to already equal the canonical target names.)*

`DataCleaner.clean()` drops any row with a null date, so **19,272 Google rows and
3,417 Meta rows were silently dropped** — leaving the pipeline to train on 2,873 Bing
rows only, defeating the entire cross-channel premise, and the combined-data
`DataValidator` check (>50% null on spend/revenue triggers a hard error) then failed
outright, so `run.sh` exited non-zero on the very data shipped in the repo.

Also found: Meta's `conversion` column is a single value column, not two (count +
value) as `META_MAP` assumed. Its magnitude/decimals (e.g. `183.00`, `163.20`,
`286.77`) and its ratio to spend (~8x, a plausible ROAS) confirm it's a **dollar
value**, not a conversion count — so it maps to `revenue`, not `conversions`. Also
found Google's `metrics_cost_micros` is denominated in micros (1e-6 currency units)
and needs `/1,000,000` before it's a usable spend figure.

**Fix:** extended `GOOGLE_MAP`/`META_MAP` to accept both UI-export and API-style
column names, added the micros→currency conversion, remapped Meta's `conversion` to
`revenue`. Updated `config/channels.yaml`'s `detection_columns` to support **either**
shape (list-of-alternative-sets), and updated `SourceDetector._detect_from_columns`
to check them. Result after fix: 0% nulls on date/spend/revenue across all 3
channels, 25,562/25,562 rows retained, validator passes with 0 errors.
Files: `src/canonical/schema.py`, `config/channels.yaml`, `src/ingestion/detector.py`.

### 4. ADF stationarity test crashed on any constant (all-zero/flat) revenue series
`StationarityTester.test()` only guarded on series length (`< 20` observations), not
on zero variance. `statsmodels.adfuller()` raises `ValueError: Invalid input, x is
constant` on a constant series. The orchestrator calls this **once per campaign** in
a loop with no try/except — with 109 real campaigns (including several
low-volume/paused ones with all-zero revenue in a stretch), this reliably crashed the
whole pipeline partway through, not just on synthetic edge cases.
**Fix:** added an explicit constant-series guard that returns a clean "undefined"
result instead of calling `adfuller`.
File: `src/evaluation/stationarity.py`.

### 5. Ensemble weights `KeyError` — config used different keys than the code
`config/forecasting.yaml`'s `ensemble.weights` used `{lightgbm: 0.5, prophet: 0.5}`,
but `EnsembleForecaster.combine()` looks up `self.weights["lgbm"]` (short name).
Crashed with `KeyError: 'lgbm'` the moment the ensemble step ran (i.e. after ~90
seconds of successful training on the real dataset — this bug only surfaces once you
get past bugs #1-4).
**Fix:** renamed the YAML keys to `lgbm`/`prophet` to match the code.
File: `config/forecasting.yaml`.

### 6. Walk-forward backtester passed config as `**kwargs`, but the model doesn't accept kwargs
`WalkForwardBacktester.run()` called `model_cls(**model_kwargs)`, but
`LGBMForecaster.__init__(self, config: dict = None)` takes a single positional dict,
not keyword arguments. Every fold failed silently (caught by a broad `except
Exception`, logged as `ERROR`, backtest continued with 0 folds) — meaning **walk-forward
CV, one of the README's headline "hackathon differentiator" stats, silently produced
no results on every run**, while looking like it succeeded because the surrounding
code didn't hard-fail.
**Fix:** `model_cls(model_kwargs or {})` — pass the dict positionally.
File: `src/evaluation/backtester.py`.

### 7. `_assert_no_leakage()` didn't actually assert anything
```python
for col in lag_cols[:5]:
    grp = df.groupby(...)[col].first()
    # pass silently — full leakage detection is done in tests
logger.info("Leakage check passed ...")
```
This computed a groupby result, discarded it, and unconditionally logged "passed" —
a no-op rubber stamp. The README lists leakage-checking as a named differentiator
(`docs/feature_catalog.md`'s "Leakage Checklist" claims features are "✅ Safe") but
the runtime check enforced nothing.
**Fix:** made it a real check — first row of every `_lag`/`_roll` column per
(channel, campaign) group must be `NaN`, or it raises. (`_roll_std_` columns are
correctly excluded — those are deliberately `fillna(0)`'d because std of one point is
undefined, not a leak.) Also verified empirically that the underlying `.shift()` lag
computation itself was **already correct** — no actual leakage existed, this was
purely a missing-safeguard issue, not an accuracy-corrupting bug.
File: `src/features/feature_store.py`.

### 8. Prophet crashes on any environment without a pre-built CmdStan backend
`pip install prophet` does **not** bundle a working CmdStan binary; `cmdstanpy`
reports "No CmdStan installation found" unless `install_cmdstan` is run separately
(a network download — which conflicts with the submission guide's "no network calls
at run time" rule). Confirmed this crashes `Prophet(...)` construction with
`AttributeError: 'Prophet' object has no attribute 'stan_backend'` in a fresh pinned
venv — i.e., this will very likely reproduce on the judge's machine too.
`ProphetForecaster.fit()` had no error handling, so this crashed the whole run.
**Fix:** wrapped the per-campaign fit in try/except; on failure, logs once and
continues, leaving `self.models` empty. `EnsembleForecaster.combine()` already
handled an empty/None Prophet prediction by falling back to LightGBM-only — that
fallback path just needed Prophet's own crash to not take the whole process down
first. This matches the documented (but previously unimplemented) mitigation in
`docs/limitations.md`: "Fit in parallel; fall back to LightGBM-only."
File: `src/forecasting/prophet_model.py`.

## Severity: silently produces wrong/missing output, doesn't crash

### 9. Quantile crossing was never actually corrected
`docs/limitations.md` claims P90<P50 crossing is mitigated via "`clip(lower=0)` and
sorted quantile post-processing" — but `UncertaintyEngine.enrich()` never sorted
anything; it only clipped negative values. Since LightGBM trains P10/P50/P90 as three
independent quantile models, crossing is a real possibility that was silently
uncorrected, which would corrupt PICP/pinball-loss validity if it occurred.
**Fix:** added `np.sort(...)` across `[revenue_p10, revenue_p50, revenue_p90]`
row-wise before computing interval width/confidence label.
File: `src/uncertainty/intervals.py`.

### 10. Validation report (`validation_report.json`/`.html`) was never written
`ValidationReport.to_json()`/`.to_html()` exist and work, but `orchestrator.run()`
never called them — only `.to_dict()` for the in-memory result. The artifact
`docs/architecture.md` advertises ("Validation → 12 checks → validation_report.json +
.html") was simply never produced by any real run.
**Fix:** call both after validation. This surfaced a **second**, previously-dormant
bug: `Path.write_text()` without `encoding="utf-8"` defaults to the platform
encoding — on Windows (cp1252), it can't encode the ⚡/✅/❌ characters in the HTML
template, so the very first real invocation crashed with `UnicodeEncodeError`. Fixed
by passing `encoding="utf-8"` explicitly (would also have silently failed in CI on a
non-UTF-8 locale).
Files: `src/pipeline/orchestrator.py`, `src/validation/validator.py`.

### 11. `SKIP_TRAIN=1` — the documented judge re-run path — was completely broken
Two independent bugs stacked:
- `src/cli/predict.py`'s skip-train branch did `bundle["pipeline"]`, but
  `ModelSerializer.save()` only ever stores `{lgbm, prophet, budget_sim, metadata}` —
  there is no `"pipeline"` key, ever. This raised `KeyError: 'pipeline'` immediately.
- Even with that fixed, `ForecastPipeline.run()` unconditionally called
  `self.lgbm.fit()` / `self.prophet.fit()` / `self.budget_sim.fit()` with **no way to
  skip training** even when a pre-trained bundle was loaded — the `skip_train` flag
  was plumbed through the CLI and `run.sh` but never actually reached the model-fit
  step.

This matters a lot: the hackathon submission guide's core contract is **"the model
must be already trained and committed — we do not retrain. The test run only
generates features and predicts."** That's exactly the `SKIP_TRAIN=1` path, and it
never worked.
**Fix:** added a `skip_train: bool` parameter to `ForecastPipeline.__init__`/`.run()`
that actually gates the fit calls; `predict.py` now constructs a fresh
`ForecastPipeline(skip_train=True)` and assigns the loaded `lgbm`/`prophet`/
`budget_sim` onto it, instead of expecting a nonexistent bundle key. Verified: a
second run with `SKIP_TRAIN=1` against the committed `pickle/model.pkl` now
reproduces the identical WMAPE/PICP (7.04%/75.23%) without re-fitting the top-level
model (only the independent walk-forward backtester still retrains fresh per-fold
models, which is expected/correct behavior for cross-validation).
Files: `src/pipeline/orchestrator.py`, `src/cli/predict.py`.

### 12. `run.sh` didn't accept the submission guide's required positional arguments
The guide's explicit contract: `./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>`, and
the grader calls it exactly that way. This repo's `run.sh` only read `DATA_DIR=...
MODEL_PATH=... bash run.sh` **environment variables** — positional args `$1`/`$2`/`$3`
were silently ignored, so a grader invocation would always fall back to the hardcoded
defaults regardless of the paths it was told to use.
**Fix:** positional args now take precedence, falling back to env vars, then to
defaults (`data/raw`, `pickle/model.pkl`, `output/predictions.csv`). Also changed the
`SKIP_TRAIN` default to auto-detect: skip training if `MODEL_PATH` already exists
(matches "we do not retrain" from the guide), unless explicitly overridden.
File: `run.sh`.

## Severity: edge-case correctness bugs found via the test suite

### 13. Bias classification always wrong when residual std is exactly 0
`ResidualDiagnostics.diagnose()`: `"Unbiased" if abs(mean) < std * 0.1 else ...`. When
predictions are exactly correct (residual std = 0), the threshold becomes `0 < 0`,
which is `False` — misclassifying a perfect, zero-mean residual as biased.
**Fix:** floor the threshold at `1e-9`.
File: `src/evaluation/residual_diagnostics.py`.

### 14. `UncertaintyEngine` was missing the `process()` method some callers expect
Dead code in `src/pipeline.py` (see below) and the original test suite called
`.process()`; only `.enrich()` existed.
**Fix:** added `process()` as a thin alias for `enrich()`.
File: `src/uncertainty/intervals.py`.

### 15. Test-suite-only bugs (not product bugs) fixed so the suite is a real signal
- `tests/test_canonical.py::test_roas_computed` tested ROAS computation on
  `CanonicalSchemaBuilder` output, but ROAS is deliberately computed later in
  `DataCleaner` — fixed the test to run the cleaner first.
- `tests/test_feature_store.py::test_no_future_leakage_in_lag_cols` used
  `groupby(...).first()`, which **silently skips NaNs** in pandas — so it was
  checking the second row, not the first, and would never have caught a real leak.
  Fixed to use `.nth(0)` (true positional first row).
- `tests/test_budget_simulator.py` called `sim.simulate(spend_changes_dict)` with one
  positional arg, but `BudgetSimulator.simulate(forecast_df, spend_changes)` needs
  both, and expects a `what_if_summary()` call afterward for the delta — fixed both
  tests to match the real (and correctly-used-elsewhere) signature.
- `tests/test_uncertainty.py` used column names `p10`/`p50`/`p90` and expected a
  `forecast_label` column; the real pipeline uses `revenue_p10`/`revenue_p50`/
  `revenue_p90` and `confidence_label` consistently everywhere else. Fixed the tests
  to match the real interface (rather than rename the production code and risk
  breaking every other caller).
- `tests/test_detector.py::test_detect_{google,meta}_from_columns` broke when I fixed
  #3 above (channels.yaml's detection columns changed to the API-style names) — see
  fix #3, added dual-format support so both the old tests and the real data pass.

## Net effect

Before this pass: `run.sh` crashed immediately (bug #1) even before touching data;
after working around that, it crashed on the repo's own committed sample CSVs (bug
#3) with a hard validation failure; after that, it crashed on canonical-schema
plumbing (#2), then stationarity (#4), then the ensemble (#5), then Prophet (#8);
walk-forward CV silently produced nothing (#6); the leakage guard and validation
report were no-ops (#7, #10); and the documented "no-retrain" judge workflow was
outright broken (#11). **16 pytest failures → 46/46 passing**, and `run.sh` now
completes in ~90s end-to-end on a clean pinned venv against the real data, producing
a genuinely-computed WMAPE=7.04%/PICP=75.23% and a valid `pickle/model.pkl` +
`output/predictions.csv`.
