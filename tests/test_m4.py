"""Tests for Module 4: Evaluation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, EvalResult

def test_load_test_set():
    ts = load_test_set()
    assert len(ts) > 0 and "question" in ts[0] and "ground_truth" in ts[0]

def test_evaluate_returns_metrics(monkeypatch):
    import config
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    r = evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert k in r and isinstance(r[k], (int, float))

def test_failure_analysis_returns():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    assert len(f) == 1

def test_failure_has_diagnosis():
    results = [EvalResult("Q1", "A1", ["C1"], "GT1", 0.5, 0.6, 0.4, 0.3)]
    f = failure_analysis(results, bottom_n=1)
    if f:
        assert "diagnosis" in f[0] and "suggested_fix" in f[0]

def test_missing_key_does_not_invent_scores(monkeypatch):
    import config
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    result = evaluate_ragas(["Nghỉ phép?"], ["12 ngày"], [["12 ngày"]], ["12 ngày"])
    assert result["evaluation_status"] == "unavailable"
    assert result["per_question"] == []
    assert all(result[key] == 0 for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"))
