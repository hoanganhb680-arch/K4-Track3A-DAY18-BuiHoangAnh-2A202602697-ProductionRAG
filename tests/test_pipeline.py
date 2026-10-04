from types import SimpleNamespace
import json

from src.m4_eval import EvalResult
from src.pipeline import evaluate_saved_cases, run_query


def test_run_query_returns_unique_parent_context(monkeypatch):
    monkeypatch.setattr("src.pipeline.get_llm_client", lambda: None)
    child = SimpleNamespace(text="đoạn con", score=0.5, metadata={"parent_id": "0:parent_0"})
    search = SimpleNamespace(search=lambda query: [child], parent_texts={"0:parent_0": "đoạn cha"})
    reranker = SimpleNamespace(rerank=lambda query, docs, top_k: [child, child])
    answer, contexts = run_query("câu hỏi", search, reranker)
    assert answer == "đoạn cha"
    assert contexts == ["đoạn cha"]


def test_resume_eval_keeps_completed_questions(monkeypatch, tmp_path):
    report_path = tmp_path / "report.json"
    cases = [{"question": name, "answer": "a", "contexts": ["c"], "ground_truth": "g"} for name in ("q1", "q2")]
    report_path.write_text(json.dumps({"cases": cases, "query_latency_ms": {}}), encoding="utf-8")
    calls = []

    def fake_evaluate(questions, answers, contexts, truths):
        calls.extend(questions)
        return {"evaluation_status": "ragas", "per_question": [EvalResult(questions[0], answers[0], contexts[0], truths[0], 0.5, 0.6, 0.7, 0.8)]}

    monkeypatch.setattr("src.pipeline.evaluate_ragas", fake_evaluate)
    evaluate_saved_cases(str(report_path))
    evaluate_saved_cases(str(report_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert calls == ["q1", "q2"]
    assert report["aggregate"]["evaluation_status"] == "ragas"
    assert report["evaluated_questions"] == 2


def test_deepseek_client_takes_priority(monkeypatch):
    import config
    monkeypatch.setattr(config, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-test")
    client = config.get_llm_client()
    assert str(client.base_url) == "https://api.deepseek.com/v1/"
