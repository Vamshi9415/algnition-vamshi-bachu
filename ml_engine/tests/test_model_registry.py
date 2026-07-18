from types import SimpleNamespace

from ml_engine.model_store.registry import registry_path
from ml_engine.model_store.serializer import ModelSerializer


def test_model_serializer_writes_registry_metadata(tmp_path):
    bundle_path = tmp_path / "model.pkl"
    lgbm = SimpleNamespace(feature_cols=["f1", "f2", "f3"])
    prophet = SimpleNamespace()
    budget_sim = SimpleNamespace()

    ModelSerializer.save(
        path=str(bundle_path),
        lgbm=lgbm,
        prophet=prophet,
        budget_sim=budget_sim,
        metadata={"wmape": 7.04, "trained": "2026-07-15", "source": "unit-test"},
    )

    registry_file = registry_path(bundle_path)
    assert registry_file.exists()

    loaded = ModelSerializer.load(str(bundle_path))
    assert loaded["metadata"]["wmape"] == 7.04
    assert loaded["metadata"]["features"] == 3
    assert loaded["metadata"]["source"] == "unit-test"