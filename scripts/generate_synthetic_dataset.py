#!/usr/bin/env python3
"""Generate the Memory-LoRA training corpus via OpenRouter.

Produces two local parquet-backed artifacts under ``data/``:

  data/docs/documents.jsonl   -- one row per document (id, category, topic,
                                  cross-corpus split, list of (section, text))
  data/qna/qna.jsonl          -- one row per recall QA pair (doc_id, split,
                                  qna_split, question, prefix, target)

Four document categories, all seeded to contain specific, checkable facts
(numbers, names, claims) so recall is gradeable by exact-match, mirroring
Code2LoRA's assertion-completion targets:

  paper              -- the REAL Code2LoRA paper, chunked into sections
                        (not synthetic -- hand-authored below from the
                        paper we already read in full).
  coding_agent_harness -- synthetic docs about capabilities that help coding
                        agents / CLI tools (Claude Code, Codex, etc.) handle
                        large codebases: context injection, repo indexing,
                        diffing, static analysis, self-improvement loops.
  agile_pm           -- synthetic docs about agile project tracking: Jira
                        ticket lifecycles, sprints, story points, velocity,
                        epics, standups, retrospectives, burndown.
  general            -- broad diverse synthetic fact-sheets, needed so the
                        hypernetwork's document->LoRA mapping generalizes
                        (breadth requirement, same reason Code2LoRA needed
                        400+ repos rather than 1).

OpenRouter is OpenAI-API compatible -- plain ``openai`` client,
base_url=https://openrouter.ai/api/v1. Key read from OPENROUTER_API_KEY
(loaded from a local .env, never hardcoded/committed).

Usage:
    python scripts/generate_synthetic_dataset.py --limit 3     # smoke test
    python scripts/generate_synthetic_dataset.py --n-per-category 60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
from memory_lora.data_paths import DOCS_DIR, QNA_DIR, CACHE_DIR, ensure_dirs  # noqa: E402


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv(REPO_ROOT / ".env")

DEFAULT_GEN_MODEL = "qwen/qwen-2.5-7b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Cache-wrapped OpenRouter call
# ---------------------------------------------------------------------------

def _cache_key(model: str, messages: List[Dict[str, str]], **kwargs) -> str:
    payload = json.dumps({"model": model, "messages": messages, **kwargs},
                          sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class CachedClient:
    def __init__(self, client: OpenAI, cache_dir: Path):
        self.client = client
        self.cache_dir = cache_dir

    def chat(self, model: str, messages: List[Dict[str, str]],
             temperature: float = 0.9, max_tokens: int = 2000,
             retries: int = 4) -> str:
        key = _cache_key(model, messages, temperature=temperature,
                          max_tokens=max_tokens)
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text()
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                cache_file.write_text(text)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = 2 ** attempt
                print(f"  [warn] OpenRouter call failed ({e}); retry in {wait}s",
                      flush=True)
                time.sleep(wait)
        raise RuntimeError(f"OpenRouter call failed after {retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Topic seeds per category (kept diverse -> the hypernetwork sees breadth)
# ---------------------------------------------------------------------------

CODING_AGENT_TOPICS = [
    "repository-level context injection strategies for LLM coding agents",
    "incremental static analysis caching for large monorepos",
    "test-impact analysis to select which tests to rerun after a diff",
    "dependency graph indexing for cross-file code navigation",
    "safe automated refactoring patterns for renaming across a codebase",
    "context window budget management when an agent reads many files",
    "self-improvement loops where a coding agent revises its own tool use",
    "diffing and patch-application strategies for multi-file edits",
    "detecting and avoiding regressions when an agent edits shared utilities",
    "code search ranking heuristics for retrieval-augmented coding agents",
    "sandboxing and permission models for autonomous coding agents",
    "long-horizon planning for agents building complex multi-module programs",
    "memory architectures that let an agent recall project conventions",
    "strategies for agents to keep a mental model of a large codebase in sync",
    "CLI tool design patterns for developer-facing coding agents",
    "evaluating coding agent reliability on multi-step programming tasks",
    "handling build system and dependency resolution errors autonomously",
    "techniques for agents to summarize large pull requests for review",
    "version-control-aware agent workflows (branches, rebases, conflicts)",
    "strategies for agents to write and maintain their own regression tests",
]

AGILE_PM_TOPICS = [
    "Jira ticket lifecycle states and transition rules",
    "sprint planning and story point estimation techniques",
    "velocity tracking and forecasting sprint capacity",
    "epic and subtask hierarchy conventions in agile tracking tools",
    "daily standup meeting structure and anti-patterns",
    "sprint retrospective formats and action item follow-through",
    "burndown and burnup chart interpretation",
    "backlog grooming and prioritization frameworks (MoSCoW, WSJF)",
    "definition of done and acceptance criteria best practices",
    "kanban WIP limits and flow efficiency metrics",
    "cross-team dependency tracking in scaled agile (SAFe, LeSS)",
    "bug triage severity/priority labeling conventions",
    "release planning and versioning cadences",
    "stakeholder reporting cadences and status update formats",
    "agile ceremonies for distributed/remote teams",
]

PROJECT_STATUS_TOPICS = [
    "a web app team's sprint status: open tickets, in-progress work, recent commits",
    "a data pipeline team's current sprint: blocked tickets, recent diffs, on-call rotation",
    "a mobile app team's release cycle: feature tickets, QA status, code review queue",
    "an API service team's incident + sprint status: hotfix tickets, recent deploys",
    "an ML training infra team's sprint: experiment tickets, recent config diffs",
    "a platform team's migration project: tracked subtasks, rollout percentage, blockers",
    "a devtools team's backlog grooming outcome: prioritized tickets, recent PRs merged",
    "a security team's remediation sprint: CVE tickets, patch status, recent commits",
    "a frontend team's design-system rollout: component tickets, adoption tracking",
    "a backend team's database migration sprint: schema tickets, rollback plan, diffs",
]

PROJECT_STATUS_SYSTEM = (
    "You write a realistic internal project-status snapshot for a software "
    "team, combining a Jira-style ticket board with recent code activity. "
    "Invent a plausible project/repo name, then include: (1) 6-10 tickets, "
    "each with a ticket key (e.g. PROJ-1234), status (To Do/In Progress/In "
    "Review/Done/Blocked), assignee name, story points, and a one-line "
    "description; (2) a sprint summary (sprint number, dates, velocity, "
    "burndown status); (3) a 'recent changes' section describing 2-4 "
    "specific code changes (file names, what changed, why) as if summarizing "
    "recent commits/diffs; (4) any current blockers or risks. Every fact "
    "(ticket key, status, assignee, points, sprint number, file name) must "
    "be specific and consistent so it can be tested for recall. 400-700 "
    "words. No markdown headers, structured prose with clear labels."
)


def gen_project_status_document(client: CachedClient, model: str, topic: str) -> str:
    prompt = f"Write a project-status snapshot for: {topic}."
    return client.chat(
        model=model,
        messages=[
            {"role": "system", "content": PROJECT_STATUS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=1400,
    )


GENERAL_TOPICS = [
    "the history and mechanics of a fictional national park's geology",
    "the biology of a deep-sea bioluminescent organism",
    "the engineering of a suspension bridge's cable system",
    "the brewing process and quality control of specialty coffee",
    "the orbital mechanics of a hypothetical exoplanet system",
    "the supply chain logistics of a regional produce cooperative",
    "the architecture of a public transit signaling system",
    "the culinary traditions of a fictional coastal fishing village",
    "the manufacturing process of a specific alloy used in aerospace",
    "the ecology of a wetland restoration project",
    "the governance structure of a municipal water utility",
    "the training regimen of competitive long-distance cyclists",
    "the archival practices of a rare-book conservation lab",
    "the acoustics engineering of a concert hall renovation",
    "the logistics of a regional disaster-relief supply network",
    "the taxonomy and care requirements of a rare orchid genus",
    "the operations of a small-batch letterpress printing studio",
    "the hydrology of an urban stormwater management system",
    "the production pipeline of a stop-motion animation studio",
    "the maintenance schedule of a commercial wind turbine farm",
]

DOC_GEN_SYSTEM = (
    "You write dense, factual reference documents. Every document must contain "
    "15-30 SPECIFIC, VERIFIABLE facts: exact numbers, named entities, precise "
    "claims, thresholds, or procedures. Avoid vague generalities. Write "
    "300-800 words. Do not use markdown headers; write flowing prose "
    "paragraphs. Invent plausible specifics (names, numbers, dates) when the "
    "topic is fictional/hypothetical -- consistency within the document "
    "matters more than real-world accuracy."
)

QA_GEN_SYSTEM = (
    "You extract recall test questions from a reference document. Given the "
    "document, produce a JSON array of 15-25 objects, each with keys "
    "'question' and 'answer'. Each question must test recall of ONE specific "
    "fact stated in the document (a number, name, threshold, or precise "
    "claim). The answer must be SHORT (1-8 words, ideally a number, name, or "
    "short phrase) and must be copyable verbatim or near-verbatim from the "
    "document. Do not ask yes/no questions. Do not ask questions requiring "
    "reasoning beyond direct recall. Output ONLY the JSON array, no prose."
)


def gen_document(client: CachedClient, model: str, category: str, topic: str) -> str:
    prompt = (
        f"Write a reference document about: {topic}.\n"
        f"Category: {category}."
    )
    return client.chat(
        model=model,
        messages=[
            {"role": "system", "content": DOC_GEN_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=1200,
    )


def gen_qna(client: CachedClient, model: str, doc_text: str) -> List[Dict[str, str]]:
    raw = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": QA_GEN_SYSTEM},
            {"role": "user", "content": doc_text},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if q and a and len(a) < 200:
            out.append({"question": q, "answer": a})
    return out


# ---------------------------------------------------------------------------
# The Code2LoRA paper -- real document, hand-chunked into sections (from the
# paper we already read in full; not regenerated by an LLM).
# ---------------------------------------------------------------------------

def code2lora_paper_sections() -> List[Dict[str, str]]:
    return [
        {"name": "abstract", "text": (
            "Code2LoRA is a hypernetwork framework that generates repository-"
            "specific LoRA adapters, effectively injecting repository "
            "knowledge with zero inference-time token overhead. Code2LoRA "
            "supports two usage scenarios: Code2LoRA-Static converts a "
            "single repository snapshot into an adapter; Code2LoRA-Evo "
            "maintains an adapter backed by a GRU hidden state updated per "
            "code diff. The authors build RepoPeftBench, a benchmark of 604 "
            "Python repositories with two tracks: a static track with 40K "
            "training and 12K test assertion-completion tasks, and an "
            "evolution track with 215K commit-derived training and 87K "
            "commit-derived test tasks. On the static track, Code2LoRA-"
            "Static achieves 63.8% cross-repo and 66.2% in-repo exact match, "
            "matching the per-repository LoRA upper bound; on the evolution "
            "track, Code2LoRA-Evo achieves 60.3% cross-repo exact match, "
            "+5.2 percentage points over a single shared LoRA."
        )},
        {"name": "method_architecture", "text": (
            "Code2LoRA has three components: a shared repository encoder "
            "that maps repository-level context to dense embeddings, a "
            "hypernetwork that maps those embeddings to LoRA weights, and a "
            "base LLM that receives the generated adapter. Only the "
            "hypernetwork is trained. The repository encoder uses a frozen "
            "Qwen3-Embedding-0.6B model: each file is divided into 4096-"
            "token chunks with 512-token overlap, embedded, and mean-pooled "
            "to produce a file vector of dimension 1024. The repository "
            "embedding is the concatenation of a weighted mean and a max "
            "pool of file vectors, giving a 2048-dimensional vector. "
            "Code2LoRA-Static's hypernetwork has a 2-layer MLP trunk with "
            "GELU activation, hidden dimension 1024, followed by dedicated "
            "output heads per module type. LoRA matrices use rank r=16 and "
            "alpha=32, targeting seven module types (q_proj, k_proj, "
            "v_proj, o_proj, gate_proj, up_proj, down_proj) shared across "
            "all 28 transformer layers of the base model. Code2LoRA-Static "
            "has approximately 720 million trainable parameters. Code2LoRA-"
            "Evo adds a 1-layer GRU with hidden size 2048 that aggregates "
            "sequential diff embeddings into a hidden state, which "
            "substitutes for the static embedding in the same shared head; "
            "Code2LoRA-Evo has approximately 745 million trainable "
            "parameters, using truncated backpropagation through time with "
            "a window of K=16 steps."
        )},
        {"name": "benchmark_repopeftbench", "text": (
            "RepoPeftBench comprises 604 Python repositories drawn from "
            "GitHub: 512 in-distribution repositories (requiring at least "
            "300 stars) and a 92-repository temporal out-of-distribution "
            "holdout created strictly after the 2025-04-01 scrape cutoff. "
            "The in-distribution set is partitioned into cross-repo (103 "
            "held-out repositories: 51 validation, 52 test) and in-repo "
            "(409 training repositories) splits. The task is assertion "
            "completion: given a structured prefix from a test file "
            "(imports, enclosing class, helper methods, test body up to the "
            "assertion), the model predicts the expected value of the "
            "assertion. The static track draws 39,612 training and 11,636 "
            "test tasks from repository snapshots. The evolution track "
            "replays commit history, yielding 215,129 training and 86,793 "
            "test tasks derived from commits. Evaluation metrics are Exact "
            "Match (EM), Edit Similarity, and CodeBLEU. The base LLM used "
            "in all experiments is Qwen2.5-Coder-1.5B, loaded in bfloat16, "
            "trained on a single H100 80GB GPU using the TRL library."
        )},
        {"name": "results_static_track", "text": (
            "On RepoPeftBench's static track, Code2LoRA-Static achieves "
            "63.8% cross-repo exact match, 9.9 percentage points above the "
            "strongest baseline (full fine-tuning plus RAG, at 53.9%). "
            "Other baselines score lower: RAG with k=3 reaches 39.7% EM, "
            "Dependency-Resolved Context reaches 48.2% EM, full fine-tuning "
            "alone reaches 51.4% EM, and a single shared LoRA reaches 47.4% "
            "EM. On in-repo evaluation, Code2LoRA-Static reaches 66.2% EM, "
            "matching the per-repository LoRA upper bound of 64.0% EM "
            "without any per-repository training. A strengthened Text2LoRA "
            "baseline, matched on input modality and target-module "
            "coverage, reaches only 45.8% EM on cross-repo, isolating the "
            "Text2LoRA hypernetwork head itself as the bottleneck."
        )},
        {"name": "results_evolution_track", "text": (
            "On the evolution track, which evaluates on commit-derived "
            "prefixes, Code2LoRA-Evo is the strongest method on both "
            "splits: 60.3% cross-repo EM and 64.5% in-repo EM, a gain of "
            "5.2 percentage points over a single shared LoRA on cross-repo. "
            "Code2LoRA-Evo's in-repo EM of 64.5% exceeds the per-repository "
            "LoRA upper bound of 64.2% without any per-repository training. "
            "Code2LoRA-Static, evaluated on the same commit-derived inputs "
            "as a within-framework reference, drops to 55.7% cross-repo EM "
            "and 60.6% in-repo EM, markedly below its static-track "
            "performance, showing that snapshot-based adaptation goes "
            "stale as a repository accumulates commits. On the 92-"
            "repository temporal out-of-distribution holdout, Code2LoRA-"
            "Evo achieves the highest exact match at 74.1%, ahead of "
            "Code2LoRA-Static at 72.2% and a single shared LoRA at 72.3%."
        )},
        {"name": "efficiency", "text": (
            "Code2LoRA-Static and Code2LoRA-Evo generate a repository-"
            "specific adapter in under 10 milliseconds per repository with "
            "zero extra inference tokens, versus approximately 1,500 extra "
            "tokens per query for RAG with k=3, and approximately 500 to "
            "2,000 extra tokens per query for Dependency-Resolved Context. "
            "Full fine-tuning requires about 4 hours of training and adds "
            "3.1 gigabytes of storage per repository; per-repository LoRA "
            "requires about 5 minutes of training and 32 megabytes of "
            "storage per repository. In contrast, Code2LoRA-Static's "
            "hypernetwork adds a fixed 679 megabytes of storage shared "
            "across all repositories, and Code2LoRA-Evo adds 65 megabytes, "
            "independent of how many repositories are served."
        )},
        {"name": "limitations", "text": (
            "The Code2LoRA paper's limitations section notes the evaluation "
            "is restricted to Python repositories, a single frozen backbone "
            "(Qwen2.5-Coder-1.5B), and one downstream task (assertion "
            "completion). The reported 74.1% out-of-distribution exact "
            "match may be partially inflated because assertion targets in "
            "the post-cutoff OOD repositories are systematically shorter "
            "(median 7 characters) than in the cross-repo and in-repo test "
            "sets (median 12-13 characters). The LoRA-generation "
            "hypernetwork dominates the trainable parameter count -- "
            "approximately 720 million for Code2LoRA-Static and 745 "
            "million for Code2LoRA-Evo -- so the evolution-track finding "
            "is most directly supported at the 1.5-billion-parameter "
            "backbone scale."
        )},
    ]


# ---------------------------------------------------------------------------
# Splits + orchestration
# ---------------------------------------------------------------------------

def assign_cross_corpus_split(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.8:
        return "train"
    if r < 0.9:
        return "cr_val"
    return "cr_test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    ap.add_argument("--n-per-category", type=int, default=60,
                     help="Docs to generate per synthetic category "
                          "(coding_agent_harness, agile_pm, general).")
    ap.add_argument("--limit", type=int, default=0,
                     help="If set, overrides --n-per-category to a small "
                          "number for a cheap smoke test.")
    ap.add_argument("--only-categories", nargs="+", default=[],
                     help="Restrict generation to these categories (e.g. "
                          "--only-categories project_status), instead of "
                          "regenerating the whole corpus. Also skips the "
                          "paper doc when set.")
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    ensure_dirs()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (expected in .env)")

    client = CachedClient(
        OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key),
        cache_dir=CACHE_DIR,
    )
    rng = random.Random(args.seed)

    n_per_cat = args.limit if args.limit else args.n_per_category

    docs_path = DOCS_DIR / "documents.jsonl"
    qna_path = QNA_DIR / "qna.jsonl"
    docs_f = docs_path.open("a")
    qna_f = qna_path.open("a")

    doc_counter = 0
    qna_counter = 0

    def emit_doc(doc_id: str, category: str, topic: str,
                 sections: List[Dict[str, str]], split: str) -> None:
        nonlocal doc_counter, qna_counter
        docs_f.write(json.dumps({
            "doc_id": doc_id, "doc_version": "v1", "category": category,
            "topic": topic, "split": split, "sections": sections,
        }) + "\n")
        doc_counter += 1

        full_text = "\n\n".join(s["text"] for s in sections)
        qnas = gen_qna(client, args.gen_model, full_text)
        n_qna = len(qnas)
        for i, qa in enumerate(qnas):
            qna_split = "train" if rng.random() < 0.8 else "held_out"
            qna_f.write(json.dumps({
                "doc_id": doc_id, "doc_version": "v1", "split": split,
                "qna_split": qna_split,
                "question": qa["question"],
                "prefix": f"Q: {qa['question']}\nA:",
                "target": " " + qa["answer"],
            }) + "\n")
            qna_counter += 1
        print(f"  [{category}] {doc_id} ({topic[:50]}...) -> {n_qna} QAs, split={split}",
              flush=True)

    # 1. The real Code2LoRA paper -- always included, always in train split
    # (we WANT it memorized, not held out for cross-corpus generalization
    # testing -- that's the whole point of this build).
    if not args.skip_paper and not args.only_categories:
        print("=== paper (real, hand-authored sections) ===", flush=True)
        emit_doc("code2lora_paper", "paper", "Code2LoRA paper",
                 code2lora_paper_sections(), split="train")

    # 2. Synthetic categories
    topic_lists = {
        "coding_agent_harness": CODING_AGENT_TOPICS,
        "agile_pm": AGILE_PM_TOPICS,
        "project_status": PROJECT_STATUS_TOPICS,
        "general": GENERAL_TOPICS,
    }
    if args.only_categories:
        topic_lists = {k: v for k, v in topic_lists.items() if k in args.only_categories}
    for category, topics in topic_lists.items():
        print(f"=== {category} ({n_per_cat} docs) ===", flush=True)
        for i in range(n_per_cat):
            topic = topics[i % len(topics)]
            if i >= len(topics):
                topic = f"{topic} (variant {i // len(topics) + 1}, different specifics)"
            doc_id = f"{category}_{i:04d}"
            split = assign_cross_corpus_split(rng)
            try:
                if category == "project_status":
                    text = gen_project_status_document(client, args.gen_model, topic)
                else:
                    text = gen_document(client, args.gen_model, category, topic)
            except RuntimeError as e:
                print(f"  [error] skipping {doc_id}: {e}", flush=True)
                continue
            sections = [{"name": "body", "text": text}]
            emit_doc(doc_id, category, topic, sections, split)

    docs_f.close()
    qna_f.close()
    print(f"\nDone. {doc_counter} documents, {qna_counter} QA pairs written to:")
    print(f"  {docs_path}")
    print(f"  {qna_path}")


if __name__ == "__main__":
    main()
