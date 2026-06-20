"""
Orca Seed Taxonomy — training domain definitions.

Every domain has: weight (% of total), system prompt, subtopics, and prompt templates.
The parallel pipeline distributes work across these domains automatically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

# ─────────────────────────────────────────────────────────────────────────────
#  Domain definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Domain:
    name: str
    weight: int  # relative weight for sampling
    system: str  # injected into seed generation prompt
    subtopics: list[str]
    templates: list[str]
    multi_turn: bool = False  # generate multi-turn conversations
    turns: int = 2            # number of turns if multi_turn


# Shared quality rules for all domains
_QUALITY = """\
Rules:
- Orca voice: direct, precise, zero sycophancy
- NEVER start with "Great question", "Certainly", "Of course", "Sure!", "Absolutely"
- NEVER end with "I hope this helps" or "Let me know if..."
- Get to the point immediately
- Use code examples when they clarify faster than prose
- If it's a short answer, keep it short
- Output ONLY valid JSON, no extra text
"""

_SCHEMA_QA = '{"question": "...", "answer": "..."}'
_SCHEMA_MULTI = '{"conversation": [{"role": "human", "content": "..."}, {"role": "gpt", "content": "..."}, ...]}'


PYTHON = Domain(
    name="python",
    weight=18,
    system=f"""\
Generate a realistic Python developer Q&A pair in Orca voice.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}""",
    subtopics=[
        "async/await and asyncio event loops",
        "Python decorators and metaclasses",
        "generators and itertools",
        "context managers and __enter__/__exit__",
        "type hints and mypy",
        "dataclasses vs NamedTuple vs Pydantic",
        "list comprehensions vs generators for memory",
        "Python GIL and threading vs multiprocessing",
        "pathlib and file operations",
        "functools: lru_cache, partial, reduce",
        "exception handling best practices",
        "Python packaging with pyproject.toml",
        "virtual environments and dependency management",
        "Python performance profiling with cProfile",
        "slots and memory optimization",
        "walrus operator and match statements",
        "dict merging operators",
        "unittest vs pytest",
        "mocking with unittest.mock",
        "regex in Python with re module",
        "struct module and binary data",
        "subprocess and os.system",
        "logging best practices",
        "argparse vs typer vs click",
        "Python closures and scoping (LEGB)",
    ],
    templates=[
        "A developer asks about {subtopic}. Write a realistic question and Orca-style answer.",
        "Debug this scenario: developer misunderstands {subtopic}. Show the mistake and the fix.",
        "Best practice question about {subtopic} from a developer switching from another language.",
    ],
)

ALGORITHMS = Domain(
    name="algorithms",
    weight=12,
    system=f"""\
Generate a computer science Q&A about algorithms and data structures.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}""",
    subtopics=[
        "time and space complexity analysis",
        "binary search and its variants",
        "merge sort vs quicksort tradeoffs",
        "hash maps: collision resolution strategies",
        "BFS vs DFS and when to use each",
        "dynamic programming with memoization",
        "sliding window technique",
        "two pointer technique",
        "heap operations and priority queues",
        "trie data structure",
        "union-find / disjoint set",
        "topological sort",
        "Dijkstra's algorithm",
        "recursion to iteration conversion",
        "bit manipulation tricks",
        "prefix sums and difference arrays",
        "monotonic stack/queue",
        "segment trees",
    ],
    templates=[
        "Explain {subtopic} with a concrete Python example.",
        "When would you choose {subtopic} over alternatives? Give a real scenario.",
        "A developer is confused about {subtopic}. Give a precise, direct explanation.",
    ],
)

DEBUGGING = Domain(
    name="debugging",
    weight=10,
    system=f"""\
Generate a realistic debugging Q&A. Show a bug and its root cause fix.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}
The question should include a short code snippet with a subtle bug.
The answer should: identify root cause (not just symptom), fix it, explain why.""",
    subtopics=[
        "off-by-one errors in loops",
        "mutable default arguments in Python",
        "race conditions in async code",
        "memory leaks in Python",
        "incorrect variable scope (closures)",
        "list mutation during iteration",
        "integer overflow in non-Python languages",
        "floating point comparison bugs",
        "SQL N+1 query problem",
        "incorrect use of == vs is in Python",
        "unhashable type errors",
        "recursion without base case",
        "key errors in dict access",
        "index out of bounds",
        "type coercion bugs in JavaScript",
        "timezone-naive datetime bugs",
        "encoding/decoding issues (utf-8 vs bytes)",
        "thread-unsafe singleton patterns",
    ],
    templates=[
        "A developer has a bug related to {subtopic}. Show buggy code and the fix.",
        "Subtle production bug caused by {subtopic}. Real scenario, real fix.",
    ],
)

BASH_LINUX = Domain(
    name="bash_linux",
    weight=8,
    system=f"""\
Generate a Linux/bash Q&A for developers.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}""",
    subtopics=[
        "pipe chaining and xargs",
        "awk for log parsing",
        "sed for in-place file editing",
        "find with complex filters",
        "grep -P and extended regex",
        "bash process substitution",
        "trap and signal handling",
        "cron job syntax",
        "systemd service units",
        "tmux workflow",
        "ssh config and key management",
        "rsync for backups",
        "lsof and netstat for debugging",
        "strace for system calls",
        "jq for JSON processing",
        "curl with headers and auth",
        "disk usage: du, df, ncdu",
        "htop, top, ps for process management",
        "environment variables and .bashrc vs .bash_profile",
        "set -euo pipefail in scripts",
    ],
    templates=[
        "A developer needs to {subtopic}. Give the command with explanation.",
        "One-liner for {subtopic}. Explain each part.",
        "Common mistake with {subtopic} and the correct approach.",
    ],
)

SQL = Domain(
    name="sql",
    weight=8,
    system=f"""\
Generate a SQL / database Q&A for developers.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}""",
    subtopics=[
        "window functions: ROW_NUMBER, RANK, DENSE_RANK",
        "CTEs vs subqueries performance",
        "JOIN types: INNER, LEFT, CROSS, FULL OUTER",
        "index types: B-tree, hash, partial",
        "EXPLAIN ANALYZE output reading",
        "GROUP BY with HAVING vs WHERE",
        "upsert with ON CONFLICT",
        "JSON operations in PostgreSQL",
        "connection pooling with PgBouncer",
        "ACID properties explained",
        "deadlock detection and prevention",
        "database normalization: 1NF, 2NF, 3NF",
        "pagination: OFFSET vs cursor-based",
        "text search with tsvector/tsquery",
        "partitioning large tables",
        "vacuum and autovacuum in PostgreSQL",
        "transactions and savepoints",
        "stored procedures vs application logic",
    ],
    templates=[
        "A developer asks about {subtopic} with a realistic schema. Show the query.",
        "Performance problem related to {subtopic}. Diagnose and fix.",
        "What's wrong with this approach to {subtopic}? Correct it.",
    ],
)

SYSTEMS_DESIGN = Domain(
    name="systems_design",
    weight=8,
    system=f"""\
Generate a system design Q&A for senior engineers.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}
Answers should be architectural and opinionated. State tradeoffs explicitly.""",
    subtopics=[
        "designing a URL shortener",
        "rate limiting strategies: token bucket vs leaky bucket",
        "caching layers: L1/L2/CDN tradeoffs",
        "event-driven architecture vs REST",
        "database sharding strategies",
        "CAP theorem in real systems",
        "service mesh: Istio vs Envoy",
        "message queues: Kafka vs RabbitMQ vs SQS",
        "distributed locking with Redis",
        "API gateway patterns",
        "circuit breaker pattern",
        "CQRS and event sourcing",
        "blue-green vs canary deployments",
        "distributed tracing with OpenTelemetry",
        "designing for idempotency",
        "bulk vs streaming data processing",
        "multi-region active-active architecture",
    ],
    templates=[
        "System design question about {subtopic}. Include constraints and tradeoffs.",
        "When would you use {subtopic} and when would you not? Be specific.",
        "What breaks at scale with {subtopic}?",
    ],
)

DOCKER_K8S = Domain(
    name="docker_k8s",
    weight=6,
    system=f"""\
Generate a Docker/Kubernetes Q&A for developers.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}""",
    subtopics=[
        "multi-stage Docker builds",
        "Docker layer caching optimization",
        "Docker compose networking",
        "Kubernetes Pod vs Deployment vs StatefulSet",
        "Kubernetes services: ClusterIP, NodePort, LoadBalancer",
        "ConfigMap vs Secret",
        "resource limits and requests",
        "horizontal pod autoscaling",
        "Kubernetes RBAC",
        "ingress controllers",
        "PersistentVolume and PersistentVolumeClaim",
        "init containers and sidecars",
        "health checks: liveness vs readiness probes",
        "rolling updates and rollback",
        "kubectl debugging workflow",
    ],
    templates=[
        "A developer is debugging {subtopic}. Show the issue and fix.",
        "Best practice for {subtopic} with a concrete example.",
        "Common mistake with {subtopic}.",
    ],
)

GIT = Domain(
    name="git",
    weight=5,
    system=f"""\
Generate a git workflow Q&A.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}""",
    subtopics=[
        "rebase vs merge: when to use each",
        "interactive rebase for cleaning history",
        "git bisect for finding regressions",
        "git stash and stash pop",
        "cherry-pick use cases",
        "git reflog for recovering lost commits",
        "branch naming conventions",
        "git hooks: pre-commit, post-merge",
        "git worktree",
        "squashing commits",
        "git blame and archaeology",
        "resolving merge conflicts",
        "git tags and semantic versioning",
        "monorepo strategies",
    ],
    templates=[
        "A developer is stuck on {subtopic}. Show the exact commands.",
        "Explain {subtopic} with a before/after example.",
    ],
)

API_DESIGN = Domain(
    name="api_design",
    weight=5,
    system=f"""\
Generate a REST/API design Q&A.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}""",
    subtopics=[
        "REST vs GraphQL vs gRPC tradeoffs",
        "versioning: URL vs header vs query param",
        "HTTP status code usage (200 vs 201 vs 204)",
        "pagination: offset, cursor, keyset",
        "authentication: JWT vs session vs API key",
        "rate limiting response headers",
        "idempotent POST vs PUT",
        "bulk endpoints design",
        "webhook design and delivery guarantees",
        "OpenAPI/Swagger spec-first development",
        "error response formats",
        "CORS configuration",
        "long-polling vs SSE vs WebSockets",
    ],
    templates=[
        "API design decision about {subtopic}. Give the tradeoffs and recommendation.",
        "What's wrong with this API design for {subtopic}? Fix it.",
    ],
)

SECURITY = Domain(
    name="security",
    weight=5,
    system=f"""\
Generate a developer security Q&A (defensive, educational).
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}
Focus on: prevention, detection, secure patterns. Educational only.""",
    subtopics=[
        "SQL injection prevention with parameterized queries",
        "XSS: stored vs reflected vs DOM-based",
        "CSRF protection patterns",
        "password hashing: bcrypt vs argon2",
        "JWT security pitfalls (alg:none, weak secrets)",
        "secrets management: vault vs env vars",
        "dependency vulnerability scanning",
        "content security policy headers",
        "SSRF prevention",
        "timing attacks on comparison functions",
        "directory traversal prevention",
        "OAuth 2.0 flow security",
        "HTTPS and certificate pinning",
        "input validation at boundaries",
    ],
    templates=[
        "Security vulnerability related to {subtopic}. Show the vulnerable pattern and the fix.",
        "How to properly implement {subtopic}? Show code.",
    ],
)

TOOL_USE = Domain(
    name="tool_use",
    weight=8,
    system=f"""\
Generate a realistic multi-turn conversation where an AI assistant uses tools.
The assistant is ORCA — it searches the web or runs code when needed.

Tool format:
  [searching web for "query"]
  → result: brief summary of what was found

  [running code]
  → output: result

Output ONLY JSON: {_SCHEMA_MULTI}
Conversation should have 3-5 turns. Show tool use naturally — not on every turn, only when useful.
{_QUALITY}""",
    subtopics=[
        "user asks for current Python version and docs link",
        "user wants to verify a math calculation",
        "user asks to check if a library exists",
        "user asks for performance benchmark comparison",
        "user wants to parse and analyze some data",
        "user asks for a code snippet that the assistant then runs to verify it works",
        "user asks about latest AI model releases",
        "user wants to debug a regex by testing it",
        "user asks for real-time pricing or API rate info",
        "user needs to verify a complex formula with code",
    ],
    templates=[
        "Scenario: {subtopic}. Generate a realistic 3-5 turn conversation showing tool use.",
    ],
    multi_turn=True,
    turns=4,
)

MULTI_TURN = Domain(
    name="multi_turn",
    weight=10,
    system=f"""\
Generate a realistic multi-turn developer conversation.
The AI is ORCA — direct, intelligent, never sycophantic.

Output ONLY JSON: {_SCHEMA_MULTI}
Conversation must have 4-6 turns. Show:
- The user refining their question
- The AI asking one clarifying question when genuinely needed
- The AI showing memory of earlier context
- No repetition, no padding
{_QUALITY}""",
    subtopics=[
        "debugging a FastAPI application",
        "designing a database schema for a SaaS app",
        "setting up a CI/CD pipeline",
        "optimizing a slow Python script",
        "choosing between two architectural approaches",
        "learning async programming in Python",
        "building a CLI tool",
        "setting up Docker for a Python app",
        "implementing authentication in a web app",
        "refactoring legacy code",
        "understanding a complex error message",
        "planning a microservices migration",
    ],
    templates=[
        "Developer scenario: {subtopic}. Generate a realistic 4-6 turn conversation.",
    ],
    multi_turn=True,
    turns=5,
)

ORCA_VOICE = Domain(
    name="orca_voice",
    weight=5,
    system=f"""\
Generate a Q&A that perfectly demonstrates Orca's voice and personality.
Orca is: precise, surgical, confident, never sycophantic, gets to the point immediately.

Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}
The answer should demonstrate Orca at its best — like a brilliant senior engineer
who has seen everything and wastes no words. Dry humor is fine when it fits.
Pick any technical topic for the question.""",
    subtopics=[
        "a common beginner mistake",
        "a subtle performance issue",
        "a misconception about how something works",
        "the right tool for a specific job",
        "why a popular approach is actually bad",
        "explaining a complex concept simply",
        "giving an opinion on a tech debate",
        "diagnosing an error from limited info",
        "comparing two approaches honestly",
        "explaining why something failed",
    ],
    templates=[
        "Topic: {subtopic}. Generate a Q&A that shows Orca's voice perfectly.",
    ],
)

REASONING = Domain(
    name="reasoning",
    weight=5,
    system=f"""\
Generate a logical reasoning or problem-solving Q&A.
Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}
Show clear, structured reasoning. State assumptions. Show the work when it matters.""",
    subtopics=[
        "Fermi estimation problem",
        "logical puzzle with a trick",
        "probability calculation",
        "systems thinking problem",
        "trade-off analysis",
        "root cause analysis scenario",
        "capacity planning calculation",
        "cost-benefit analysis",
        "first-principles reasoning about a tech claim",
    ],
    templates=[
        "A developer asks a reasoning question about {subtopic}. Show structured thinking.",
    ],
)


STARTUP_STRATEGY = Domain(
    name="startup_strategy",
    weight=6,
    system=f"""\
Generate a Q&A about startup strategy, entrepreneurship, or building a business.
Orca speaks as a direct, experienced co-founder — no corporate fluff, no platitudes.

Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}
The answer should be specific and actionable, not generic advice.
Think: what would a YC partner or experienced operator actually say?""",
    subtopics=[
        "pricing strategy for a new SaaS product",
        "how to find your first 10 customers",
        "when to raise funding vs bootstrap",
        "how to validate an idea without building it",
        "choosing a co-founder",
        "competitive moat for an AI product",
        "how to pitch to pre-seed investors",
        "revenue model options for developer tools",
        "when to pivot vs stay the course",
        "building a product with no marketing budget",
        "how to get into Y Combinator",
        "positioning against established competitors",
        "go-to-market for a B2B product",
        "unit economics and what matters early",
    ],
    templates=[
        "A founder asks about {subtopic}. Give a direct, experienced answer.",
        "Question about {subtopic} from an early-stage entrepreneur.",
    ],
)

INVESTOR_PITCH = Domain(
    name="investor_pitch",
    weight=5,
    system=f"""\
Generate a Q&A about investor relations, fundraising, or pitch preparation.
Orca knows what investors actually look for and gives honest, specific advice.

Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}
Answers should reference real frameworks (TAM/SAM/SOM, traction metrics, story arc)
without being generic. Be specific about what works and what kills deals.""",
    subtopics=[
        "what makes a great elevator pitch",
        "how to calculate and present TAM",
        "what investors look for in a founding team",
        "how to structure a pre-seed pitch deck",
        "what traction metrics matter at each stage",
        "how to handle 'what's your moat?' question",
        "angel investors vs venture capitalists",
        "how to find and approach investors cold",
        "common reasons investors pass on deals",
        "how to negotiate a term sheet",
        "what does a lead investor actually do",
        "SAFE vs priced round at pre-seed",
        "how to tell your company story compellingly",
    ],
    templates=[
        "A first-time founder asks: {subtopic}. Give honest investor-perspective advice.",
        "Investor Q&A about {subtopic}.",
    ],
)

REVENUE_GENERATION = Domain(
    name="revenue_generation",
    weight=5,
    system=f"""\
Generate a Q&A about generating revenue, monetization, or building profitable products.
Orca thinks like an operator who has shipped products that make money.

Output ONLY JSON: {_SCHEMA_QA}
{_QUALITY}
Focus on practical, specific tactics — not abstract principles.
Include numbers, frameworks, or concrete examples where possible.""",
    subtopics=[
        "converting free users to paid",
        "pricing psychology for software products",
        "building a sales process from zero",
        "outbound vs inbound customer acquisition",
        "reducing churn for SaaS products",
        "upselling and expanding existing accounts",
        "affiliate and partnership revenue models",
        "API monetization strategies",
        "building recurring revenue vs one-time sales",
        "automating lead generation",
        "content marketing as a revenue channel",
        "productizing a service or consulting business",
    ],
    templates=[
        "A founder asks how to {subtopic}. Give specific, actionable advice.",
        "Revenue question: {subtopic}. What actually works?",
    ],
)


# Registry — all domains in order
ALL_DOMAINS: list[Domain] = [
    PYTHON,
    ALGORITHMS,
    DEBUGGING,
    BASH_LINUX,
    SQL,
    SYSTEMS_DESIGN,
    DOCKER_K8S,
    GIT,
    API_DESIGN,
    SECURITY,
    TOOL_USE,
    MULTI_TURN,
    ORCA_VOICE,
    REASONING,
    STARTUP_STRATEGY,
    INVESTOR_PITCH,
    REVENUE_GENERATION,
]

DOMAIN_MAP: dict[str, Domain] = {d.name: d for d in ALL_DOMAINS}

TOTAL_WEIGHT = sum(d.weight for d in ALL_DOMAINS)


def get_domain(name: str) -> Domain:
    if name not in DOMAIN_MAP:
        raise ValueError(f"Unknown domain '{name}'. Available: {list(DOMAIN_MAP)}")
    return DOMAIN_MAP[name]


def sample_domains(n: int, names: list[str] | None = None) -> list[tuple[Domain, int]]:
    """
    Distribute n examples across domains by weight.
    Returns list of (domain, count) pairs.
    """
    domains = [DOMAIN_MAP[name] for name in names] if names else ALL_DOMAINS
    total_w = sum(d.weight for d in domains)
    counts = []
    allocated = 0
    for i, d in enumerate(domains):
        if i == len(domains) - 1:
            count = n - allocated
        else:
            count = max(1, round(n * d.weight / total_w))
        counts.append((d, count))
        allocated += count
    return counts


def build_prompt(domain: Domain) -> tuple[str, str]:
    """Return (system, user) prompt for one generation."""
    subtopic = random.choice(domain.subtopics)
    template = random.choice(domain.templates)
    user = template.format(subtopic=subtopic)
    return domain.system, user
