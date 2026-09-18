---
name: rag-eval
description: Evaluates RAG retrieval precision, Hit Rate, MRR, and answer faithfulness against acceptance criteria. Use this skill whenever testing RAG pipeline changes, validating chunking strategies, or running benchmark evaluations.
---

# RAG Evaluation & Benchmarking Skill

This workspace skill provides the standard operating procedure for evaluating and benchmarking the RAG pipeline against the metrics defined in `specs/`.

## Evaluation Procedure

1. **Verify Test Dataset**:
   Ensure sample documents exist in `data/raw/` and benchmark queries with ground-truth chunk IDs exist in `tests/fixtures/eval_dataset.json`.

2. **Execute Retrieval Benchmark**:
   Run the evaluation script to calculate Hit Rate @ K, Mean Reciprocal Rank (MRR), and latency:
   ```powershell
   python .agent/skills/rag-eval/scripts/eval_retrieval.py --k 3
   ```

3. **Check Metric Thresholds**:
   Compare results against the active specification (e.g. `specs/00-system-spec.md`):
   - **Hit Rate @ 3**: Target $\ge 85\%$
   - **MRR**: Target $\ge 0.75$
   - **P95 Latency**: Target $< 500\text{ ms}$ for local retrieval

4. **Document Results**:
   When completing a milestone or update, record the benchmark output into the current `walkthrough.md` artifact under the **Validation Results** section.
