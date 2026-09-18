# Project Rules: Spec-Driven Development & Antigravity Workflow

Welcome to the **Sales Assistant** project. This project strictly enforces **Spec-Driven Development (SDD)**.

---

## 1. Core Principles of Spec-Driven Development (SDD)

1. **Spec as the Single Source of Truth**:
   - Every feature, component, or pipeline design must originate from a formal specification located in `specs/`.
   - Never implement code in `src/` without referencing an existing, approved specification document.
2. **Contract-First Design**:
   - Define data schemas, chunking rules, embedding dimensionality, and retrieval APIs in the specification before implementation.
3. **Spec Updates on Pivot**:
   - If design details change during implementation or testing, update the corresponding `specs/*.md` document immediately.

---

## 2. Agent Interaction Protocol

When assisting with this project, the agent must adhere to the following workflow:

1. **Specification Review**:
   - When asked to add or change functionality, check `specs/` first.
   - If a spec is missing or ambiguous, propose or update a spec using `specs/templates/spec-template.md`.
   - Recommend the user run `/grill-me` to interview and harden specifications before planning implementation.
2. **Planning Mode**:
   - For any multi-file or architectural task, create an `implementation_plan.md` artifact outlining the phased tasks, dependencies, and test plan.
   - Await explicit user approval before modifying code.
3. **Test-Driven Verification**:
   - Code changes must be accompanied by unit or contract tests in `tests/`.
   - Verification results and evaluation metrics must be documented in a `walkthrough.md` artifact.
4. **Skills & Tool Usage**:
   - Use `.agent/skills/rag-eval` whenever evaluating retrieval quality, chunking effectiveness, or RAG output accuracy.
   - For broad documentation lookups, invoke the `research` subagent to preserve main chat context.

---

## 3. Project Structure Reference

- `specs/`: Markdown specifications defining requirements, schemas, and metrics.
- `.agent/skills/`: Workspace-specific agent skills and runbooks.
- `src/`: Production code organized by subsystem (`ingestion`, `retrieval`, `generation`).
- `tests/`: Verification suites and benchmarks testing against spec criteria.
- `data/`: Raw and sample documents used for RAG grounding.

## 4. Python Environment Rules
ALWAYS use the Python environment at `/home/ngthuan/anaconda3/envs/sales-assistant/`.
- Python: `/home/ngthuan/anaconda3/envs/sales-assistant/bin/python`
- Pip: `/home/ngthuan/anaconda3/envs/sales-assistant/bin/pip`
- Do NOT use any other Python installation

## 5. Shell Execution Rules
Always use bash -c '<command>' (or sh -c '<command>') for all shell executions to ensure the process terminates correctly and closes standard streams (sending EOF).
Example: Use bash -c 'pip list' instead of just pip list.
Avoid interactive shells. Ensure commands run non-interactively (e.g., redirect input with < /dev/null or use non-interactive flags like -y where applicable). If multiple steps or a wrapped session are needed, group them within bash -c '...' and ensure the command explicitly exits (e.g., bash -c '...; exit').
