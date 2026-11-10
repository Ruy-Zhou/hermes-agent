"""Restricted backend: shell commands in a subprocess with a clamped cwd, an
allowlisted environment and a hard timeout.

This is intentionally a *shallow* sandbox. There is no chroot, namespace,
seccomp or user separation: the path guard inspects command text and clamps the
cwd the wrapper reports back, and the env guard builds a fresh environment for
each child. That keeps accidental escapes and host credentials out of child
processes; it is not a kernel-enforced security boundary.

The snapshot machinery is deliberately disabled. A login-shell profile would
reintroduce the very host environment variables the allowlist exists to drop.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Iterable

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.environments.base import BaseEnvironment
from tools.environments.local import _find_bash

logger = logging.getLogger(__name__)

# Values are read from the Hermes process environment; nothing else crosses into
# the child. PATH is included because a shell with no PATH cannot run anything.
DEFAULT_RESTRICTED_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM")

# Names that let a value execute code or redirect module/shell loading. Even an
# explicit allowlist entry must not reintroduce them.
_FORBIDDEN_ENV_NAMES = frozenset({
    "BASH_ENV", "BASHOPTS", "CDPATH", "DYLD_INSERT_LIBRARIES", "ENV", "GLOBIGNORE",
    "IFS", "JAVA_TOOL_OPTIONS", "LD_LIBRARY_PATH", "LD_PRELOAD", "NODE_OPTIONS",
    "PERL5OPT", "PS4", "PYTHONPATH", "PYTHONSTARTUP", "SHELLOPTS",
})

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IS_WINDOWS = os.name == "nt"

# Find the start of a path-shaped fragment inside a shell token. The lookbehind
# keeps URL-ish and arithmetic text ("http://x", "1/2", "s/a/b/g") out of the
# guard while still catching "2>/etc/passwd" and "--out=/etc/passwd".
_PATH_START_RE = re.compile(r"(?<![A-Za-z0-9_:./\\])(?:[A-Za-z]:[\\/]|/|~/|\.{1,2}/)")


def _normalise_allowlist(names: Iterable[str] | str) -> tuple[str, ...]:
    """Validate the configured names; drop blanks, duplicates and unsafe names."""
    if isinstance(names, str):
        names = names.replace(",", " ").split()
    result: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        if not _ENV_NAME_RE.match(name):
            raise ValueError(f"invalid restricted env allowlist name: {name!r}")
        if name.upper() in _FORBIDDEN_ENV_NAMES:
            logger.warning("restricted backend ignoring unsafe env allowlist entry %s", name)
            continue
        if name not in result:
            result.append(name)
    return tuple(result)


def _is_within(root: str, candidate: str) -> bool:
    """True when *candidate* is *root* or below it (case-insensitive on Windows)."""
    try:
        norm_root = os.path.normcase(root)
        return os.path.commonpath([norm_root, os.path.normcase(candidate)]) == norm_root
    except (ValueError, OSError):
        return False


def _path_candidates(command: str) -> list[str]:
    """Path-shaped fragments in a shell command, including bare ``..`` tokens.

    shlex strips quotes so ``cat '/etc/passwd'`` is seen; a malformed command
    falls back to whitespace splitting, which is stricter, never more permissive.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    candidates: list[str] = []
    for token in tokens:
        if token == ".." or ".." in re.split(r"[/\\]", token):
            candidates.append(token)
        match = _PATH_START_RE.search(token)
        if match:
            candidates.append(token[match.start():])
    return candidates


def find_path_violation(command: str, cwd: str, root: str) -> str | None:
    """Reject command text that names a path outside *root*; else return None.

    Paths are resolved against *cwd* and through symlinks, so ``../`` escapes,
    absolute host paths and links pointing out of the root are all caught. This
    is a static check: dynamically constructed paths (``$VAR``, ``$(...)``) are
    not evaluated here — the cwd clamp in ``_update_cwd`` is the second layer.
    """
    for text in _path_candidates(command):
        text = text.rstrip("'\"")
        if text == "~" or text.startswith("~/"):
            text = root + text[1:]
        elif text.startswith("~"):
            return f"home expansion {text!r} is not allowed"
        candidate = text if os.path.isabs(text) else os.path.join(cwd, text)
        resolved = os.path.realpath(os.path.normpath(candidate))
        if not _is_within(root, resolved):
            return f"path {text!r} escapes restricted root {root!r}"
    return None


class RestrictedEnvironment(BaseEnvironment):
    """Run each command as a fresh local bash subprocess inside one directory."""

    is_local = True
    # File tools must go through execute(): reading host paths directly would
    # bypass the same path guard the shell is subject to.
    uses_host_filesystem = False
    _passwordless_sudo_probe_supported = False

    def __init__(self, cwd: str = "", timeout: int = 60, *, root: str = "",
                 env_allowlist: Iterable[str] | str = ()):
        self._allowlist = _normalise_allowlist(env_allowlist) or DEFAULT_RESTRICTED_ENV_ALLOWLIST
        raw_root = os.path.expanduser(root or cwd or os.getcwd())
        # as_posix keeps cwd usable as a Git Bash `cd` target on Windows; normcase
        # in _is_within still compares the two spellings correctly.
        self.root = Path(os.path.realpath(os.path.abspath(raw_root))).as_posix()
        if not os.path.isdir(self.root):
            raise ValueError(f"restricted root {self.root!r} is not an existing directory")
        # BaseEnvironment.__init__ calls get_temp_dir(), which reads self.root.
        super().__init__(cwd=self.root, timeout=timeout, persist_env=False)
        Path(self._temp_dir).mkdir(parents=True, exist_ok=True)
        # No login shell, no snapshot: sourcing a profile would read host env.
        self._snapshot_ready = False
        self._prefer_nonlogin = True

    @property
    def _temp_dir(self) -> str:
        return (Path(self.root) / ".hermes-restricted").as_posix()

    def get_temp_dir(self) -> str:
        """Session artifacts stay inside the root, never in the host /tmp."""
        return self._temp_dir

    def init_session(self):
        """Intentionally empty; see the module docstring."""

    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        from tools.environments.local import _windows_to_msys_path
        return BaseEnvironment._quote_cwd_for_cd(_windows_to_msys_path(cwd))

    def _clamp_cwd(self, cwd: str) -> str:
        resolved = os.path.realpath(os.path.abspath(os.path.expanduser(cwd or self.root)))
        if not _is_within(self.root, resolved):
            raise ValueError(f"cwd {resolved!r} escapes restricted root {self.root!r}")
        return Path(resolved).as_posix()

    def _clean_env(self) -> dict[str, str]:
        """Build the child environment from scratch — never os.environ.copy()."""
        cleaned = {name: os.environ[name] for name in self._allowlist if name in os.environ}
        cleaned.setdefault("PATH", os.defpath)
        # These are set by the backend, not passed through from the host: keeping
        # ~ / temp writes / $PWD inside the root is what makes the cwd guard hold.
        cleaned["HOME"] = self.root
        cleaned["TMPDIR"] = self._temp_dir
        cleaned["PWD"] = self.root
        cleaned["TERM"] = cleaned.get("TERM", "dumb")
        return cleaned

    def execute(self, command: str, cwd: str = "", **kwargs) -> dict:
        """Guard command text and the requested cwd before the shared execute flow."""
        try:
            target_cwd = self._clamp_cwd(cwd or self.cwd)
        except ValueError as exc:
            return {"output": f"[restricted] {exc}", "returncode": 126}
        violation = find_path_violation(command, target_cwd, self.root)
        if violation:
            return {"output": f"[restricted] {violation}", "returncode": 126}
        return super().execute(command, cwd=target_cwd, **kwargs)

    def _update_cwd(self, result: dict):
        """A `cd` that slipped past the static guard must not move the next call."""
        previous = self.cwd
        super()._update_cwd(result)
        if _IS_WINDOWS and self.cwd != previous:
            # The wrapper reports Git Bash's /d/... spelling; translate it back so
            # a legitimate `cd subdir` is not mistaken for an escape.
            from tools.environments.local import _msys_to_windows_path
            if normalized := _msys_to_windows_path(self.cwd):
                self.cwd = Path(normalized).as_posix()
                if "cwd" in result:
                    result["cwd"] = self.cwd
        if _is_within(self.root, os.path.realpath(self.cwd)):
            return
        self.cwd = previous
        result.pop("cwd_observed", None)
        result.pop("cwd", None)
        result["output"] = (result.get("output") or "") + (
            f"\n[restricted] cwd outside {self.root}; staying in {previous}")

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        # login is ignored on purpose: `bash -l` sources host profiles.
        args = [_find_bash(), "-c", cmd_string]
        proc = subprocess.Popen(
            args, text=True, env=self._clean_env(), encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            # Own session so a timeout can kill the whole tree, not just bash.
            start_new_session=True, cwd=self._clamp_cwd(self.cwd),
            **({"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}))
        if stdin_data is not None:
            self._pipe_stdin(proc, stdin_data)
        return proc

    @staticmethod
    def _pipe_stdin(proc: subprocess.Popen, stdin_data: str) -> None:
        errors: list[OSError] = []

        def write() -> None:
            try:
                proc.stdin.write(stdin_data)
                proc.stdin.close()
            except OSError as exc:
                errors.append(exc)

        # spawn_context_thread keeps the caller's profile scope across the hop.
        from agent.memory_provider import spawn_context_thread
        thread = spawn_context_thread(write, name="restricted-stdin")
        thread.start()
        proc._hermes_stdin_thread = thread
        proc._hermes_stdin_errors = errors

    def _kill_process(self, proc) -> None:
        """SIGKILL the child's whole process group (or the Windows process tree)."""
        if not _IS_WINDOWS:
            with contextlib.suppress(OSError, ProcessLookupError):
                pgid = os.getpgid(proc.pid)
                if pgid != os.getpgrp():
                    os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — _IS_WINDOWS branch
        else:
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=windows_hide_flags(), timeout=5, check=False)
        with contextlib.suppress(OSError):
            proc.kill()

    def _force_kill_process(self, proc) -> None:
        self._kill_process(proc)

    def cleanup(self):
        for path in (self._snapshot_path, self._cwd_file):
            with contextlib.suppress(OSError):
                os.unlink(path)
