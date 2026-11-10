"""Behaviour-contract tests for the ``restricted`` terminal backend.

The backend is a shallow sandbox, so these tests pin the four guarantees it
does make (subprocess execution, timeout kill, cwd confinement, env scrubbing)
and the factory/config wiring that lets ``terminal.backend: restricted`` be
selected like any other backend.
"""

import os
import shutil

import pytest

from tools.environments import restricted as r
from tools.environments.restricted import (RestrictedEnvironment, _is_within,
                                           _normalise_allowlist, find_path_violation)


def _bash_binary():
    found = shutil.which("bash")
    if found:
        return found
    try:
        import pm.shell
        return pm.shell.bash()
    except Exception:
        return None


BASH = _bash_binary()
needs_bash = pytest.mark.skipif(not BASH, reason="a bash executable is required for subprocess tests")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "inside.txt").write_text("inside-marker", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside-marker", encoding="utf-8")
    monkeypatch.setenv("HERMES_RESTRICTED_HOST_VALUE", "must-not-leak")
    monkeypatch.setenv("HERMES_RESTRICTED_ALLOWED_VALUE", "allowed-value")
    monkeypatch.setattr(r, "_find_bash", lambda: BASH)
    env = RestrictedEnvironment(cwd=str(root), timeout=5, root=str(root),
                                env_allowlist=("PATH", "HERMES_RESTRICTED_ALLOWED_VALUE"))
    yield env, root, tmp_path
    env.cleanup()


# --- Path guard (pure) -------------------------------------------------------

def test_path_guard_rejects_absolute_and_parent_escapes(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for command in ("cat /etc/passwd", "cd ..", "cat ../outside.txt",
                    "cat sub/../../outside.txt", "echo hi > /etc/out", "cat /root/.ssh/id_rsa"):
        assert find_path_violation(command, str(root), str(root)), command


def test_path_guard_allows_paths_inside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for command in ("cat inside.txt", f"cat {root / 'inside.txt'}", "echo http://example.com",
                    "python -c 'print(1/2)'", "cat ~/inside.txt", "mkdir -p sub && cd sub"):
        assert find_path_violation(command, str(root), str(root)) is None, command


def test_path_guard_follows_symlinks_out_of_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")
    assert find_path_violation("cat link/secret.txt", str(root), str(root))


def test_path_guard_catches_absolute_path_in_assignment_and_quote(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert find_path_violation("tool --out=/etc/passwd", str(root), str(root))
    assert find_path_violation("cat '/etc/passwd'", str(root), str(root))


# --- Env allowlist (pure) ----------------------------------------------------

def test_allowlist_drops_unsafe_names_and_invalid_identifiers():
    names = _normalise_allowlist(["PATH", "LD_PRELOAD", "PYTHONPATH", "PATH", " safe "])
    assert names == ("PATH", "safe")
    with pytest.raises(ValueError):
        _normalise_allowlist(["BAD-NAME"])


def test_clean_env_only_contains_allowlisted_host_values(sandbox):
    env, _, _ = sandbox
    cleaned = env._clean_env()
    assert cleaned["HERMES_RESTRICTED_ALLOWED_VALUE"] == "allowed-value"
    assert "HERMES_RESTRICTED_HOST_VALUE" not in cleaned
    assert cleaned["HOME"] == env.root
    assert cleaned["TMPDIR"] == env.get_temp_dir()
    assert cleaned["PWD"] == env.root
    assert _is_within(env.root, os.path.realpath(cleaned["HOME"]))


# --- Subprocess execution ----------------------------------------------------

@needs_bash
def test_execute_runs_command_and_sees_files_inside_root(sandbox):
    env, _, _ = sandbox
    result = env.execute("cat inside.txt")
    assert result["returncode"] == 0, result
    assert "inside-marker" in result["output"]


@needs_bash
def test_execute_does_not_leak_host_environment(sandbox):
    env, _, _ = sandbox
    result = env.execute("env")
    assert result["returncode"] == 0, result
    assert "allowed-value" in result["output"]
    assert "must-not-leak" not in result["output"]
    assert "HERMES_RESTRICTED_HOST_VALUE" not in result["output"]


@needs_bash
@pytest.mark.parametrize("command", ["cat ../outside.txt", "cat /etc/passwd", "cd ..",
                                     "cat ../../etc/passwd", "echo hi > ../escaped-file"])
def test_execute_refuses_commands_that_escape_root(sandbox, command):
    env, _, tmp_path = sandbox
    result = env.execute(command)
    assert result["returncode"] == 126, result
    assert "[restricted]" in result["output"]
    assert "outside-marker" not in result["output"]
    assert not (tmp_path / "escaped-file").exists()


@needs_bash
def test_execute_refuses_an_outside_cwd_argument(sandbox):
    env, _, tmp_path = sandbox
    result = env.execute("pwd", cwd=str(tmp_path))
    assert result["returncode"] == 126, result
    assert "escapes restricted root" in result["output"]


@needs_bash
def test_execute_kills_command_that_exceeds_timeout(sandbox):
    env, _, _ = sandbox
    result = env.execute("sleep 30", timeout=1)
    assert result["returncode"] == 124, result
    assert result.get("hermes_timed_out") is True
    assert "timed out" in result["output"]


@needs_bash
def test_execute_pipes_stdin_to_the_child(sandbox):
    env, _, _ = sandbox
    result = env.execute("cat", stdin_data="stdin-marker")
    assert result["returncode"] == 0, result
    assert "stdin-marker" in result["output"]


@needs_bash
def test_cwd_observed_by_wrapper_stays_inside_root(sandbox):
    env, _, _ = sandbox
    result = env.execute("mkdir -p sub && cd sub")
    assert result["returncode"] == 0, result
    assert _is_within(env.root, os.path.realpath(env.cwd))
    # A subsequent command still runs inside the root, not wherever `cd` landed.
    assert env.execute("cat ../inside.txt")["returncode"] == 0


def test_no_login_shell_snapshot_is_used(sandbox):
    env, _, _ = sandbox
    assert env._snapshot_ready is False
    assert env._prefer_nonlogin is True
    assert env.uses_host_filesystem is False


# --- Backend selection / config wiring --------------------------------------

def test_factory_builds_restricted_environment(tmp_path):
    from tools import terminal_tool_backends as backends
    root = tmp_path / "root"
    root.mkdir()
    env = backends._create_environment(
        env_type="restricted", image="", cwd=str(root), timeout=7,
        restricted_config={"root": str(root), "env_allowlist": ["PATH"]})
    try:
        assert isinstance(env, RestrictedEnvironment)
        assert env.timeout == 7
        assert _is_within(env.root, os.path.realpath(str(root)))
    finally:
        env.cleanup()


def test_restricted_shaper_falls_back_to_default_allowlist():
    from tools import terminal_tool_backends as backends
    shaped = backends._restricted_config_from_config({"restricted_root": "", "restricted_env_allowlist": []})
    assert shaped["root"] == ""
    assert shaped["env_allowlist"] == r.DEFAULT_RESTRICTED_ENV_ALLOWLIST


def test_requirements_reject_a_missing_root_and_accept_an_empty_one(tmp_path):
    from tools import terminal_tool_backends as backends
    pre = backends._BACKEND_SPECS["restricted"]["pre"]
    assert pre({"restricted_root": ""}) is True
    assert pre({"restricted_root": str(tmp_path)}) is True
    assert pre({"restricted_root": str(tmp_path / "does-not-exist")}) is False


def test_env_config_exposes_restricted_keys(monkeypatch, tmp_path):
    from tools import terminal_tool
    monkeypatch.setenv("TERMINAL_ENV", "restricted")
    monkeypatch.setenv("TERMINAL_RESTRICTED_ROOT", str(tmp_path))
    monkeypatch.setenv("TERMINAL_RESTRICTED_ENV_ALLOWLIST", '["PATH","LANG"]')
    config = terminal_tool._get_env_config()
    assert config["env_type"] == "restricted"
    assert config["restricted_root"] == str(tmp_path)
    assert config["restricted_env_allowlist"] == ["PATH", "LANG"]
    assert terminal_tool._backend_default_cwd("restricted") != "/root"
