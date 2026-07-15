# pickle/model.pkl

This directory holds the serialised model bundle produced by `run.sh`.

## How it is created

```bash
bash run.sh
# After this completes, pickle/model.pkl exists and can be inspected:
python -c "import pickle; b = pickle.load(open('pickle/model.pkl','rb')); print(b.keys())"
```

## Contents of the bundle

| Key | Type | Description |
|-----|------|-------------|
| `lgbm` | `LGBMForecaster` | Trained LightGBM P10/P50/P90 models |
| `prophet` | `ProphetForecaster` | Fitted Prophet models per campaign |
| `budget_sim` | `BudgetSimulator` | Fitted elasticity map |
| `metadata` | `dict` | Training date, data hash, WMAPE, PICP |

## Skip-training mode

If `pickle/model.pkl` already exists:

```bash
SKIP_TRAIN=1 DATA_DIR=data/raw OUTPUT_PATH=output/predictions.csv bash run.sh
```

This loads the bundle and skips all training — useful for judge re-runs on the same data.
