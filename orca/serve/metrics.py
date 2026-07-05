"""
Production monitoring — real gap this closes: zero visibility into
errors/latency/uptime once this runs somewhere real. No dashboard, no
alerting, no way to know if the service is degrading before a user
complains.

HONEST SCOPE:
  - In-memory, single-instance. Metrics don't survive a restart and don't
    aggregate across multiple API instances if you're running more than
    one (see orca/serve/session_store.py / rate limiting for the pattern
    this project uses when something DOES need to be cluster-shared via
    Redis — metrics don't get that treatment here because pushing every
    request's timing through Redis on the hot path would add real latency
    for a feature whose whole point is observability, not correctness).
    A real multi-instance production deployment should scrape /metrics from
    EACH instance independently (standard Prometheus practice — this is
    not a limitation unique to this implementation, it's how Prometheus
    scraping is supposed to work) rather than expecting one aggregated
    number.
  - Latency percentiles are computed from a bounded rolling window (last
    2000 samples per endpoint), not a true histogram — fine for "is this
    endpoint getting slower," not for feeding into billing/SLA-grade
    reporting.
  - The /metrics endpoint deliberately has NO auth (standard Prometheus
    scrape convention — the scraper is usually inside a private network,
    not exposed to the internet). If you expose this port publicly, put a
    firewall/reverse-proxy rule in front of it — a public /metrics endpoint
    leaks operational detail (request volumes, error rates, endpoint
    names) that shouldn't be internet-visible.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

_lock = threading.Lock()
_START_TIME = time.time()

_request_count: dict[str, int] = defaultdict(int)
_status_count: dict[str, int] = defaultdict(int)          # keyed "endpoint:status_code"
_error_count: dict[str, int] = defaultdict(int)            # 5xx responses, keyed by endpoint
_latency_samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))

MAX_ENDPOINTS_TRACKED = 500  # defensive cap — see record_request()


def record_request(endpoint: str, status_code: int, duration_ms: float) -> None:
    """Called once per request by the metrics middleware. Never raises —
    a metrics recording failure must not break the request it's measuring."""
    try:
        with _lock:
            # Defensive cap: if endpoint labels somehow become unbounded
            # (e.g. a bug elsewhere putting raw IDs into the path label
            # instead of a normalized route template), stop growing new
            # buckets rather than leaking memory indefinitely. Existing
            # tracked endpoints keep updating normally.
            if endpoint not in _request_count and len(_request_count) >= MAX_ENDPOINTS_TRACKED:
                endpoint = "_overflow_"

            _request_count[endpoint] += 1
            _status_count[f"{endpoint}:{status_code}"] += 1
            if status_code >= 500:
                _error_count[endpoint] += 1
            _latency_samples[endpoint].append(duration_ms)
    except Exception:
        pass


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return round(ordered[idx], 1)


def get_metrics_snapshot() -> dict:
    """JSON-friendly summary — used by GET /api/admin/metrics."""
    with _lock:
        endpoints = {}
        for endpoint, count in _request_count.items():
            samples = list(_latency_samples[endpoint])
            endpoints[endpoint] = {
                "requests": count,
                "errors": _error_count.get(endpoint, 0),
                "error_rate": round(_error_count.get(endpoint, 0) / count, 4) if count else 0.0,
                "latency_p50_ms": _percentile(samples, 0.50),
                "latency_p95_ms": _percentile(samples, 0.95),
                "latency_p99_ms": _percentile(samples, 0.99),
            }

        total_requests = sum(_request_count.values())
        total_errors = sum(_error_count.values())

        return {
            "uptime_sec": round(time.time() - _START_TIME),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_error_rate": round(total_errors / total_requests, 4) if total_requests else 0.0,
            "endpoints": endpoints,
        }


def get_prometheus_text() -> str:
    """Prometheus exposition format — GET /metrics. No auth (standard scrape convention, see module docstring)."""
    lines = [
        "# HELP orca_uptime_seconds Time since the server process started.",
        "# TYPE orca_uptime_seconds counter",
        f"orca_uptime_seconds {round(time.time() - _START_TIME)}",
        "",
        "# HELP orca_requests_total Total requests handled, by endpoint.",
        "# TYPE orca_requests_total counter",
    ]
    with _lock:
        for endpoint, count in _request_count.items():
            safe_label = endpoint.replace('"', "'")
            lines.append(f'orca_requests_total{{endpoint="{safe_label}"}} {count}')

        lines += [
            "",
            "# HELP orca_errors_total Total 5xx responses, by endpoint.",
            "# TYPE orca_errors_total counter",
        ]
        for endpoint, count in _error_count.items():
            safe_label = endpoint.replace('"', "'")
            lines.append(f'orca_errors_total{{endpoint="{safe_label}"}} {count}')

        lines += [
            "",
            "# HELP orca_latency_p95_ms 95th percentile request latency in milliseconds, by endpoint.",
            "# TYPE orca_latency_p95_ms gauge",
        ]
        for endpoint, samples in _latency_samples.items():
            safe_label = endpoint.replace('"', "'")
            p95 = _percentile(list(samples), 0.95)
            lines.append(f'orca_latency_p95_ms{{endpoint="{safe_label}"}} {p95}')

    return "\n".join(lines) + "\n"


def reset() -> None:
    """Test-only — clears all recorded metrics."""
    with _lock:
        _request_count.clear()
        _status_count.clear()
        _error_count.clear()
        _latency_samples.clear()
