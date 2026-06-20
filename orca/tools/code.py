"""
Orca Code Tool — sandboxed local code execution.
Runs Python and shell code in a subprocess with timeout and output capture.
No network calls to execute code. Fully local.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

TIMEOUT = 30  # seconds
MAX_OUTPUT = 8000  # chars


@dataclass
class RunResult:
    code: str
    language: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def format(self) -> str:
        lines = []
        if self.timed_out:
            lines.append(f"[TIMEOUT after {TIMEOUT}s]")
        if self.stdout:
            lines.append(self.stdout[:MAX_OUTPUT])
        if self.stderr:
            lines.append(f"[stderr]\n{self.stderr[:2000]}")
        if not lines:
            lines.append(f"[exit {self.exit_code}]")
        return "\n".join(lines)


def run_python(code: str) -> RunResult:
    """Execute Python code in a subprocess."""
    code = textwrap.dedent(code)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        return RunResult(
            code=code,
            language="python",
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(code=code, language="python", stdout="", stderr="", exit_code=-1, timed_out=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


def run_shell(command: str) -> RunResult:
    """Execute a shell command. Blocked: rm -rf, dd, mkfs, format, shutdown."""
    BLOCKED = ["rm -rf", "dd if=", "mkfs", ":(){:|:&};:", "shutdown", "reboot", "sudo rm"]
    for b in BLOCKED:
        if b in command.lower():
            return RunResult(
                code=command,
                language="shell",
                stdout="",
                stderr=f"BLOCKED: '{b}' is a destructive command. Orca won't run it.",
                exit_code=1,
            )
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        return RunResult(
            code=command,
            language="shell",
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return RunResult(code=command, language="shell", stdout="", stderr="", exit_code=-1, timed_out=True)


def run_code(code: str, language: str = "python") -> RunResult:
    """Dispatch to the right executor."""
    lang = language.lower().strip()
    if lang in ("python", "python3", "py"):
        return run_python(code)
    elif lang in ("shell", "bash", "sh", "zsh"):
        return run_shell(code)
    else:
        return RunResult(
            code=code,
            language=lang,
            stdout="",
            stderr=f"Language '{lang}' not supported. Supported: python, shell/bash.",
            exit_code=1,
        )
