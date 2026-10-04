from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from config import DEEPSEEK_API_KEY, EMBEDDING_MODEL, GEMINI_API_KEY, LLM_PROVIDER, OPENAI_API_KEY, RAGAS_MODEL, gemini_http_clients
    api_key = {"deepseek": DEEPSEEK_API_KEY, "gemini": GEMINI_API_KEY, "openai": OPENAI_API_KEY}[LLM_PROVIDER]
    if api_key:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from ragas.run_config import RunConfig
            from ragas.llms import LangchainLLMWrapper
            from datasets import Dataset
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_openai import ChatOpenAI

            llm_options = {"api_key": api_key, "model": RAGAS_MODEL, "timeout": 45, "max_retries": 1}
            if LLM_PROVIDER == "deepseek":
                llm_options["base_url"] = "https://api.deepseek.com/v1"
            elif LLM_PROVIDER == "gemini":
                llm_options["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
                llm_options["max_retries"] = 2
                llm_options["http_client"], llm_options["http_async_client"] = gemini_http_clients()
            llm = LangchainLLMWrapper(ChatOpenAI(**llm_options), bypass_n=LLM_PROVIDER != "openai")
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(max_workers=2, max_retries=1, timeout=90),
                raise_exceptions=True,
                batch_size=2,
            )
            df = result.to_pandas()
            per_question = [
                EvalResult(
                    question=str(row.get("question", row.get("user_input"))),
                    answer=str(row.get("answer", row.get("response"))),
                    contexts=list(row.get("contexts", row.get("retrieved_contexts"))),
                    ground_truth=str(row.get("ground_truth", row.get("reference"))),
                    faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                    context_precision=float(row.get("context_precision", 0.0) or 0.0),
                    context_recall=float(row.get("context_recall", 0.0) or 0.0),
                )
                for _, row in df.iterrows()
            ]

            f_score = float(df["faithfulness"].mean()) if "faithfulness" in df and not df["faithfulness"].isna().all() else 0.0
            ar_score = float(df["answer_relevancy"].mean()) if "answer_relevancy" in df and not df["answer_relevancy"].isna().all() else 0.0
            cp_score = float(df["context_precision"].mean()) if "context_precision" in df and not df["context_precision"].isna().all() else 0.0
            cr_score = float(df["context_recall"].mean()) if "context_recall" in df and not df["context_recall"].isna().all() else 0.0

            return {
                "faithfulness": round(f_score, 4),
                "answer_relevancy": round(ar_score, 4),
                "context_precision": round(cp_score, 4),
                "context_recall": round(cr_score, 4),
                "per_question": per_question,
                "evaluation_status": "ragas",
                "judge_model": RAGAS_MODEL,
            }
        except Exception as e:
            print(f"  ⚠️  RAGAS evaluation failed: {e}")
    else:
        print("  ⚠️  RAGAS requires DEEPSEEK_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY; scores unavailable.")
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [],
        "evaluation_status": "unavailable",
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    if not eval_results:
        return []

    scored_items = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg_score = sum(metrics.values()) / 4.0
        worst_metric = min(metrics, key=metrics.get)
        diag, fix = diagnostic_tree.get(worst_metric, ("Unknown error", "Review pipeline"))

        scored_items.append({
            "avg_score": avg_score,
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "diagnosis": diag,
            "suggested_fix": fix,
        })

    scored_items.sort(key=lambda x: x["avg_score"])
    return [
        {
            "question": x["question"],
            "worst_metric": x["worst_metric"],
            "score": x["score"],
            "diagnosis": x["diagnosis"],
            "suggested_fix": x["suggested_fix"],
            "answer": x["answer"],
            "ground_truth": x["ground_truth"],
        }
        for x in scored_items[:bottom_n]
    ]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k not in ("per_question", "cases", "query_latency_ms")},
        "num_questions": len(results.get("cases", results.get("per_question", []))),
        "evaluated_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(item) for item in results.get("per_question", [])],
        "cases": results.get("cases", []),
        "query_latency_ms": results.get("query_latency_ms", {}),
        "model": results.get("model"),
        "judge_model": results.get("judge_model"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
