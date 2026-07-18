import pandas as pd

from ml_engine.llm import gemini
from ml_engine.llm.insights import InsightGenerator


def test_round_robin_key_manager_rotates_and_dedupes():
    manager = gemini.RoundRobinKeyManager(["k1", "k2", "k1", " "])

    assert manager.count == 2
    assert manager.current() == "k1"
    assert manager.next() == "k2"
    assert manager.mark_exhausted("k2") == "k1"


def test_round_robin_key_manager_reads_env_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_1", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1, k2\n, k3")

    manager = gemini.RoundRobinKeyManager()

    assert manager.count == 3
    assert manager.current() == "k1"
    assert manager.next() == "k2"
    assert manager.next() == "k3"


def test_round_robin_key_manager_ignores_empty_and_whitespace_entries(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_1", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", " KEY1, KEY2, ,\nKEY3 ,,  KEY2  ")

    manager = gemini.RoundRobinKeyManager()

    assert manager.count == 3
    assert manager.current() == "KEY1"
    assert manager.next() == "KEY2"
    assert manager.next() == "KEY3"
    assert manager.next() == "KEY1"


def test_round_robin_key_manager_falls_back_when_no_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_1", raising=False)

    try:
        gemini.RoundRobinKeyManager()
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_gemini_llm_raises_gracefully_when_all_keys_exhausted(monkeypatch):
    class QuotaError(Exception):
        status_code = 429

    class FakeModels:
        def generate_content(self, model, contents, config=None):
            raise QuotaError("quota exhausted")

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    monkeypatch.setattr(gemini, "GENAI_AVAILABLE", True)
    model = gemini.GeminiLLM(keys=["k1", "k2"])
    monkeypatch.setattr(model, "_client_for", lambda key: FakeClient())
    monkeypatch.setattr(gemini.GeminiLLM, "_is_quota_error", staticmethod(lambda exc: isinstance(exc, QuotaError)))

    try:
        model.generate("hello")
        raised = False
    except gemini.GeminiError as exc:
        raised = True
        assert "rate-limited" in str(exc)

    assert raised is True


def test_gemini_llm_retries_on_quota_error_and_rotates_keys(monkeypatch):
    class QuotaError(Exception):
        status_code = 429

    class FakeResponse:
        text = "success"
        candidates = []

    class FakeModels:
        def __init__(self, key: str):
            self.key = key

        def generate_content(self, model, contents, config=None):
            if self.key == "k1":
                raise QuotaError("quota exhausted")
            return FakeResponse()

    class FakeClient:
        def __init__(self, key: str):
            self.models = FakeModels(key)

    monkeypatch.setattr(gemini, "GENAI_AVAILABLE", True)
    model = gemini.GeminiLLM(keys=["k1", "k2"])
    monkeypatch.setattr(model, "_client_for", lambda key: FakeClient(key))
    monkeypatch.setattr(gemini.GeminiLLM, "_is_quota_error", staticmethod(lambda exc: isinstance(exc, QuotaError)))

    assert model.generate("hello") == "success"


def test_insight_generator_uses_direct_api(monkeypatch):
    class FakeGemini:
        def __init__(self, model: str):
            self.calls: list[tuple[str, str | None]] = []

        def generate(self, user_content: str, system_prompt: str | None = None) -> str:
            self.calls.append((user_content, system_prompt))
            return "ok"

    monkeypatch.setattr("ml_engine.llm.insights.GENAI_AVAILABLE", True)
    monkeypatch.setattr("ml_engine.llm.insights.GeminiLLM", lambda model: FakeGemini(model))

    generator = InsightGenerator()
    forecast = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "campaign_name": ["A", "B"],
            "channel": ["Search", "Social"],
            "revenue_p50": [100, 200],
            "revenue_p10": [90, 180],
            "revenue_p90": [110, 220],
        }
    )

    assert generator.executive_summary(forecast) == "ok"
    assert generator.budget_recommendation({"spend": 100}) == "ok"
