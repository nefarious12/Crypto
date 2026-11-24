"""Project 5: evaluate the RAG pipeline against a labelled question set.

Metrics (per configuration):
  retrieval_hit   answerable questions where the retrieved chunks contain all expected facts
  answer_correct  answerable questions where the answer contains all expected facts
  false_abstain   answerable questions where the system wrongly said "not found"
  abstain_correct unanswerable questions where the system correctly said "not found"
  grounded        non-abstained answers whose numbers all appear in the retrieved context

It also sweeps chunk size and top_k so you can see which settings help.

Run: python -m genai_finance.evaluate
Educational only. Not investment advice.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .rag import DATA_DIR, Retriever, answer, build_chunks

ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"\d[\d,]*\.?\d*")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def numbers(s: str) -> set[str]:
    return {n.rstrip(".,") for n in NUM.findall(re.sub(r"\[\d+\]", "", s))}


def load_eval_set() -> list[dict]:
    return json.loads((DATA_DIR / "eval_set.json").read_text(encoding="utf-8"))


def evaluate(max_words=120, overlap=20, top_k=3, min_score=0.15, verbose=False) -> dict:
    retriever = Retriever(build_chunks([DATA_DIR / "meridian_fy2024.txt"], max_words, overlap))
    items = load_eval_set()
    answerable = [q for q in items if q["answerable"]]
    unanswerable = [q for q in items if not q["answerable"]]
    hit = correct = false_abstain = abstain_ok = grounded = answered = 0
    failures = []
    for q in items:
        r = answer(q["question"], retriever, top_k=top_k, min_score=min_score, use_llm=False)
        ctx = norm(" ".join(r["contexts"]))
        if q["answerable"]:
            facts = [norm(e) for e in q["expected"]]
            h = all(f in ctx for f in facts)
            c = all(f in norm(r["answer"]) for f in facts)
            hit += h
            correct += c
            false_abstain += r["abstained"]
            if not c:
                failures.append({"id": q["id"], "question": q["question"], "retrieved_ok": h, "abstained": r["abstained"]})
        else:
            abstain_ok += r["abstained"]
            if not r["abstained"]:
                failures.append({"id": q["id"], "question": q["question"], "retrieved_ok": False, "abstained": False})
        if not r["abstained"]:
            answered += 1
            grounded += numbers(r["answer"]) <= numbers(" ".join(r["contexts"]))
    metrics = {
        "max_words": max_words,
        "top_k": top_k,
        "chunks": len(retriever.chunks),
        "retrieval_hit": hit / len(answerable),
        "answer_correct": correct / len(answerable),
        "false_abstain": false_abstain / len(answerable),
        "abstain_correct": abstain_ok / len(unanswerable),
        "grounded": grounded / answered if answered else float("nan"),
    }
    if verbose:
        metrics["failures"] = failures
    return metrics


def sweep(sizes=(40, 120, 300), ks=(1, 3, 5)) -> list[dict]:
    return [evaluate(max_words=s, overlap=min(20, s // 4), top_k=k) for s in sizes for k in ks]


def save_outputs(rows: list[dict]) -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "eval_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    sizes = sorted({r["max_words"] for r in rows})
    ks = sorted({r["top_k"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    width = 0.25
    for j, k in enumerate(ks):
        vals = [next(r["answer_correct"] for r in rows if r["max_words"] == s and r["top_k"] == k) for s in sizes]
        bars = ax.bar(np.arange(len(sizes)) + (j - 1) * width, vals, width, label=f"top_k = {k}")
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8)
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([f"{s} words" for s in sizes])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Answer correct (answerable questions)")
    ax.set_xlabel("Chunk size")
    ax.set_title("RAG evaluation: chunk size and top_k change answer accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.01, 0.01, "Synthetic annual report, 21 answerable and 4 unanswerable questions. Educational only.", fontsize=7, color="gray")
    fig.tight_layout()
    (ROOT / "images").mkdir(exist_ok=True)
    fig.savefig(ROOT / "images" / "eval_sweep.png")


def fmt(rows: list[dict]) -> str:
    head = f"{'words':>5} {'top_k':>5} {'chunks':>6} {'retr_hit':>8} {'correct':>8} {'false_abs':>9} {'abs_ok':>7} {'grounded':>8}"
    lines = [head]
    for r in rows:
        lines.append(
            f"{r['max_words']:>5} {r['top_k']:>5} {r['chunks']:>6} {r['retrieval_hit']:>8.0%} {r['answer_correct']:>8.0%} "
            f"{r['false_abstain']:>9.0%} {r['abstain_correct']:>7.0%} {r['grounded']:>8.0%}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    rows = sweep()
    print(fmt(rows))
    save_outputs(rows)
    print("\nAbstention threshold trade-off (top_k=3): answering more vs. refusing unanswerable questions")
    print(f"{'words':>5} {'min_score':>9} {'correct':>8} {'false_abs':>9} {'abs_ok':>7}")
    for w in (40, 120, 300):
        for ms in (0.05, 0.10, 0.15):
            r = evaluate(max_words=w, overlap=min(20, w // 4), top_k=3, min_score=ms)
            print(f"{w:>5} {ms:>9.2f} {r['answer_correct']:>8.0%} {r['false_abstain']:>9.0%} {r['abstain_correct']:>7.0%}")
    detail = evaluate(verbose=True)
    print("\nDefault config (120 words, top_k=3) failures:")
    for f in detail["failures"]:
        print(" ", f)
