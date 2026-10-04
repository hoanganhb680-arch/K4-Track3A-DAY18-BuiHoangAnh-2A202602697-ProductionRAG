from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import json, os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import EvalResult, load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import LLM_MODEL, RERANK_TOP_K, get_llm_client


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    parent_texts = {}
    for doc_index, doc in enumerate(docs):
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for parent in parents:
            parent_texts[f"{doc_index}:{parent.parent_id}"] = parent.text
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": f"{doc_index}:{child.parent_id}"}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    search.parent_texts = parent_texts
    print(f"  ✓ Indexed ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    selected = reranked if reranked else results[:3]
    parents = getattr(search, "parent_texts", {})
    contexts = list(dict.fromkeys(parents.get(r.metadata.get("parent_id"), r.text) for r in selected))

    client = get_llm_client() if contexts else None
    if client:
        try:
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model=LLM_MODEL, messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            raise
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    query_times = []

    for i, item in enumerate(test_set):
        query_start = time.perf_counter()
        answer, contexts = run_query(item["question"], search, reranker)
        query_times.append(round((time.perf_counter() - query_start) * 1000, 1))
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    results["query_latency_ms"] = {"total": round(sum(query_times), 1), "average": round(sum(query_times) / max(len(query_times), 1), 1)}
    results["cases"] = [
        {"question": question, "answer": answer, "contexts": context, "ground_truth": truth}
        for question, answer, context, truth in zip(questions, answers, all_contexts, ground_truths)
    ]
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    if results.get("evaluation_status") != "ragas":
        print("  RAGAS unavailable; scores are not measured.")
    else:
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            s = results.get(m, 0)
            print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


def evaluate_saved_cases(path: str = "reports/ragas_report.json"):
    """Resume RAGAS judging without repeating retrieval and generation."""
    from config import LLM_MODEL, RAGAS_MODEL

    with open(path, encoding="utf-8") as report_file:
        report = json.load(report_file)
    cases = report["cases"]
    if report.get("model") not in (None, LLM_MODEL):
        raise ValueError("The saved answers were generated with a different model")
    completed = [EvalResult(**item) for item in report.get("per_question", [])]
    if any(item.question != cases[index]["question"] for index, item in enumerate(completed)):
        raise ValueError("Saved evaluations do not match the question order")

    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    for index in range(len(completed), len(cases)):
        case = cases[index]
        for attempt in range(3):
            result = evaluate_ragas([case["question"]], [case["answer"]],
                                    [case["contexts"]], [case["ground_truth"]])
            if result["evaluation_status"] == "ragas":
                break
            if attempt < 2:
                time.sleep(60)
        else:
            raise RuntimeError(f"RAGAS failed for question {index + 1}; completed results remain in {path}")

        completed.extend(result["per_question"])
        results = {
            **{metric: round(sum(getattr(item, metric) for item in completed) / len(completed), 4) for metric in metrics},
            "per_question": completed,
            "evaluation_status": "ragas" if len(completed) == len(cases) else "partial",
            "cases": cases,
            "query_latency_ms": report.get("query_latency_ms", {}),
            "model": LLM_MODEL,
            "judge_model": RAGAS_MODEL,
        }
        save_report(results, failure_analysis(completed, bottom_n=5), path=path)
        print(f"  RAGAS {len(completed)}/{len(cases)} complete", flush=True)
    return completed


if __name__ == "__main__":
    if sys.argv[1:] == ["--resume-eval"]:
        evaluate_saved_cases()
    else:
        start = time.time()
        search, reranker = build_pipeline()
        evaluate_pipeline(search, reranker)
        print(f"\nTotal: {time.time() - start:.1f}s")
