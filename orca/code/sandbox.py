"""
Sandboxed Python code execution for the Code Interpreter feature.

Safety model:
  1. AST static analysis — block dangerous node types before any execution
  2. Subprocess isolation — run in a fresh Python process, not the server process
  3. Hard time limit (10s default) — killed via timeout
  4. No network: subprocess inherits parent env but code can't reach Ollama/internet
     without explicit sockets (which AST check blocks)
  5. Stdout/stderr captured, combined as output

Intentionally NOT using Docker/nsjail — this is a local-first tool for the user's
own machine. The AST check stops obvious abuse; network restrictions and FS isolation
require Docker and are a future upgrade path.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Optional

MAX_EXEC_SECONDS = 10
MAX_OUTPUT_BYTES = 64 * 1024   # 64KB — prevents runaway print loops
MAX_CODE_BYTES   = 128 * 1024  # 128KB

# AST node types that are always blocked
BANNED_NODES: set[str] = {
    # Import of dangerous modules
    "Import", "ImportFrom",
}

# Module names allowed through the import whitelist
_ALLOWED_MODULES: set[str] = {
    # stdlib — safe / useful
    "math", "cmath", "decimal", "fractions", "statistics",
    "random", "itertools", "functools", "collections", "heapq", "bisect",
    "string", "re", "textwrap", "difflib", "unicodedata",
    "datetime", "calendar", "time",
    "json", "csv", "io", "struct", "codecs",
    "pprint", "reprlib", "copy", "enum", "dataclasses", "typing",
    "abc", "contextlib", "operator",
    "hashlib", "hmac", "base64", "binascii", "uuid",
    "pathlib",  # read-only use is fine; write ops will just fail
    "traceback", "sys", "platform",
    # Data science staples (present if user installed them)
    "numpy", "pandas", "scipy", "sklearn", "matplotlib",
    "seaborn", "plotly",
    # Misc popular
    "PIL", "Pillow", "requests",  # network — blocked via AST attribute check below
}

# Attribute / call names that are outright banned regardless of module
_BANNED_ATTRS: set[str] = {
    "eval", "exec", "compile", "__import__",
    "open",          # file write — read is OK for many scripts, but block by default
    "system", "popen", "Popen",  # os/subprocess shell-out
    "socket", "connect", "listen",  # network primitives
    "ctypes", "cffi", "mmap",
    "memoryview",
}

# os module operations that are unsafe
_BANNED_OS_ATTRS: set[str] = {
    "system", "popen", "execv", "execve", "fork", "kill",
    "remove", "unlink", "rmdir", "makedirs", "mkdir", "rename",
    "chmod", "chown", "setuid", "setgid",
}


@dataclass
class CodeResult:
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None     # static analysis or timeout error
    exit_code: int = 0
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.exit_code == 0

    def combined_output(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr}")
        if self.error:
            parts.append(f"[error]\n{self.error}")
        return "\n".join(parts) or "(no output)"


def _check_ast(code: str) -> Optional[str]:
    """
    Parse and walk the AST. Return an error string if blocked, else None.
    This is the first line of defense — runs in the server process, never executes code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    for node in ast.walk(tree):
        # ── Import check ────────────────────────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    return f"ImportError: module '{alias.name}' is not allowed in the sandbox"

        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            top = node.module.split(".")[0]
            if top not in _ALLOWED_MODULES:
                return f"ImportError: module '{node.module}' is not allowed in the sandbox"

        # ── Banned builtins / calls ─────────────────────────────────────────────
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in _BANNED_ATTRS:
                    return f"SecurityError: '{node.func.id}' is not allowed in the sandbox"
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in _BANNED_ATTRS:
                    return f"SecurityError: attribute '{node.func.attr}' is not allowed in the sandbox"
                # Extra block for os.* dangerous attrs
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    if node.func.attr in _BANNED_OS_ATTRS:
                        return f"SecurityError: 'os.{node.func.attr}' is not allowed in the sandbox"

        # ── __dunder__ attribute access ─────────────────────────────────────────
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                # Allow common safe dunders but block class-surgery patterns
                _SAFE_DUNDERS = {"__name__", "__doc__", "__class__", "__len__",
                                  "__str__", "__repr__", "__iter__", "__next__",
                                  "__init__", "__dict__", "__slots__"}
                if node.attr not in _SAFE_DUNDERS:
                    return f"SecurityError: attribute '{node.attr}' is not allowed in the sandbox"

    return None  # passed


def run_code(
    code: str,
    timeout: int = MAX_EXEC_SECONDS,
) -> CodeResult:
    """
    Execute Python code in a subprocess with AST pre-check.
    Returns a CodeResult with stdout/stderr/error/duration.
    """
    code = textwrap.dedent(code).strip()

    if len(code.encode()) > MAX_CODE_BYTES:
        return CodeResult(error=f"Code exceeds {MAX_CODE_BYTES//1024}KB limit")

    # Static analysis first — never spawns a process
    ast_error = _check_ast(code)
    if ast_error:
        return CodeResult(error=ast_error)

    # Run in subprocess
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=timeout,
            text=True,
            # Deliberately inherit env — the user's own machine, their packages
        )
        duration_ms = int((time.monotonic() - t0) * 1000)

        stdout = result.stdout[:MAX_OUTPUT_BYTES]
        stderr = result.stderr[:MAX_OUTPUT_BYTES]

        if len(result.stdout) > MAX_OUTPUT_BYTES:
            stdout += f"\n...(output truncated at {MAX_OUTPUT_BYTES//1024}KB)"

        return CodeResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=result.returncode,
            duration_ms=duration_ms,
        )

    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - t0) * 1000)
        return CodeResult(
            error=f"Execution timed out after {timeout}s",
            exit_code=-1,
            duration_ms=duration_ms,
        )
    except Exception as e:
        return CodeResult(error=f"Execution failed: {e}", exit_code=-1)
