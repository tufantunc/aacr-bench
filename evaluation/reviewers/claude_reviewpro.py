"""review-pro reviewer: Claude Code harness + review-pro skill instead of /code-review.

This is arm A2 of the pre-registered comparison in
https://github.com/tufantunc/review-pro/tree/main/studies/2026-08-aacr-bench-comparison

Design constraint from the registration: A2 must differ from A1 (reviewers/claude.py)
**only in the reviewer**. Everything else — env resolution, MCP finding contract,
StopFailure hook, result envelope parsing, token accounting, repo prep/cleanup — is
imported from reviewers.claude rather than reimplemented, so the two arms cannot
drift apart through copy-paste.

What differs, and only this:
  1. review-pro's core (skills + subagents) is installed into an isolated agent home
     per run, so the harness discovers the review-pro skill without touching the
     operator's real ~/.claude.
  2. The prompt asks for the review-pro skill on the same base...head target A1 gets,
     instead of the official /code-review slash command.

Registration rules encoded here:
  - Every finding synthesis emits is reported, including Low/Nitpick. No severity
    floor, no post-hoc pruning: anti-overreporting is the product's job at synthesis
    time, not this adapter's job afterwards.
  - Findings go through the same MCP contract as A1 (file, line, summary,
    failure_scenario), so the judge sees both arms in identical shape.

Config (env):
  ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL  (shared with A1)
  REVIEW_PRO_CORE_DIR   required — path to a review-pro checkout's core/ directory
  REVIEW_PRO_STACK_PACK  optional — stack pack name to install per instance
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import config
from repo_utils import clean_worktree, diff_stat, prepare_repo
from schema import ReviewInstance

# A1's plumbing, reused verbatim so the arms stay comparable.
from reviewers.claude import (
    build_findings_system_prompt,
    build_hook_settings,
    build_mcp_config,
    ensure_claude_installed,
    resolve_claude_env,
    _write_temp_settings,
    _check_and_log_retry_exhausted,
    save_result as _save_result_claude,
)

REVIEW_PRO_CORE_DIR_VAR = "REVIEW_PRO_CORE_DIR"
REVIEW_PRO_STACK_PACK_VAR = "REVIEW_PRO_STACK_PACK"

_REVIEWER_ID = "review-pro/skill"


def log(message: str) -> None:
    print(message, flush=True)


def resolve_review_pro_core() -> Path:
    """Locate the pinned review-pro core/ directory, failing early if absent."""
    raw = os.environ.get(REVIEW_PRO_CORE_DIR_VAR)
    if not raw:
        raise SystemExit(
            f"Missing env var: {REVIEW_PRO_CORE_DIR_VAR}. Point it at the core/ "
            "directory of a pinned review-pro checkout."
        )
    core = Path(raw).expanduser().resolve()
    for required in ("skills", "agents"):
        if not (core / required).is_dir():
            raise SystemExit(f"{REVIEW_PRO_CORE_DIR_VAR}={core} has no {required}/ directory.")
    return core


def install_review_pro_home(core_dir: Path) -> Path:
    """Create an isolated CLAUDE_CONFIG_DIR containing review-pro's skills+agents.

    Mirrors what `review-pro init --target claude-code` does (skills/ and agents/
    under the tool home) without invoking the CLI, so the pin is the checkout rather
    than whatever npm would resolve.
    """
    home = Path(tempfile.mkdtemp(prefix="review_pro_home_"))
    shutil.copytree(core_dir / "skills", home / "skills")
    (home / "agents").mkdir(parents=True, exist_ok=True)
    for agent_file in sorted((core_dir / "agents").glob("*.md")):
        shutil.copy2(agent_file, home / "agents" / agent_file.name)
    skill_count = len(list((home / "skills").iterdir()))
    agent_count = len(list((home / "agents").iterdir()))
    log(f"review-pro home: {home} ({skill_count} skills, {agent_count} agents)")
    return home


def install_stack_pack(repo_path: Path, core_dir: Path) -> Optional[str]:
    """Copy one stack pack into the repo's .review-pro/, if configured.

    Registration note: the stack pack is part of normal review-pro usage, so it is
    allowed — but it is recorded in the result envelope so the run is reproducible
    and the choice is visible.
    """
    pack = os.environ.get(REVIEW_PRO_STACK_PACK_VAR)
    if not pack:
        return None
    source = core_dir.parent / "stacks" / pack
    if not source.is_dir():
        raise SystemExit(f"{REVIEW_PRO_STACK_PACK_VAR}={pack} not found at {source}")
    destination = repo_path / ".review-pro" / pack
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    log(f"stack pack installed: {pack}")
    return pack


def build_review_prompt(base_commit: str, head_commit: str) -> str:
    """Ask for review-pro on the same base...head range A1 reviews.

    Deliberately close to A1's target semantics: same range, no extra hints about
    what to look for, no repository-specific guidance. The only instruction beyond
    "use review-pro" is to report everything, which mirrors the registration's
    no-severity-floor rule.
    """
    return (
        f"Use the review-pro skill to review the changes in {base_commit}...{head_commit} "
        "in the repository at the current working directory.\n\n"
        "Report EVERY finding the synthesis stage produces, at every severity "
        "including Low and Nitpick — do not filter, rank, or summarise them away."
    )


def run_review_pro(
    repo_path: Path,
    base_commit: str,
    head_commit: str,
    instance_id: str,
    results_dir: Path,
    claude_env: Dict[str, str],
    review_pro_home: Path,
    timeout_minutes: int,
) -> subprocess.CompletedProcess:
    """Invoke Claude Code headlessly with review-pro as the reviewer."""
    mcp_config = build_mcp_config(results_dir, instance_id)
    findings_prompt = build_findings_system_prompt()
    settings_path = _write_temp_settings(
        build_hook_settings(results_dir, instance_id),
        prefix="review_pro_settings_",
    )
    prompt = build_review_prompt(base_commit, head_commit)

    command = [
        "claude",
        "-p",
        prompt,
        "--add-dir", str(repo_path),
        "--permission-mode", "acceptEdits",
        "--output-format", "json",
        "--mcp-config", json.dumps(mcp_config),
        "--allowed-tools", config.MCP_REPORT_TOOL,
        "--append-system-prompt", findings_prompt,
        "--settings", settings_path,
    ]

    environment = os.environ.copy()
    environment.update(claude_env)
    # Isolate skill/agent discovery to the pinned review-pro core.
    environment["CLAUDE_CONFIG_DIR"] = str(review_pro_home)

    process_timeout = timeout_minutes * 60 + 60
    log(f"$ claude -p '<review-pro {base_commit[:8]}...{head_commit[:8]}>'  (cwd={repo_path})")
    try:
        return subprocess.run(
            command,
            cwd=str(repo_path),
            env=environment,
            capture_output=True,
            text=True,
            timeout=process_timeout,
        )
    except subprocess.TimeoutExpired as expired:
        log(f"TIMEOUT after {process_timeout}s — keeping findings reported so far")
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=expired.stdout or "",
            stderr=(expired.stderr or "") + f"\nTimeoutExpired after {process_timeout}s",
        )
    finally:
        try:
            os.unlink(settings_path)
        except OSError:
            pass


def save_result(
    results_dir: Path,
    instance: ReviewInstance,
    review_process: subprocess.CompletedProcess,
    started_at: str,
    duration_seconds: float,
    stack_pack: Optional[str],
) -> Path:
    """Delegate to A1's writer, then stamp the arm-specific provenance fields.

    Reusing A1's writer means the findings collapse (MCP partial -> envelope,
    fallback parsing, token accounting) is identical across arms; only the labels
    differ.
    """
    out_path = _save_result_claude(
        results_dir=results_dir,
        instance=instance,
        review_process=review_process,
        started_at=started_at,
        duration_seconds=duration_seconds,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["reviewer"] = _REVIEWER_ID
    payload["review_pro"] = {
        "core_dir": os.environ.get(REVIEW_PRO_CORE_DIR_VAR),
        "stack_pack": stack_pack,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def review_instance(
    instance: ReviewInstance,
    repo_dir: Path,
    results_dir: Path,
    claude_env: Dict[str, str],
    timeout_minutes: int = 30,
    preview: bool = False,
    review_pro_home: Optional[Path] = None,
) -> Dict[str, Any]:
    """Review one instance with review-pro. Signature mirrors reviewers.claude."""
    log(f"=== [review-pro] Processing {instance.instance_id} ===")

    core_dir = resolve_review_pro_core()
    repo_path = prepare_repo(
        repo_dir=repo_dir,
        clone_url=instance.resolved_clone_url,
        repo_full_name=instance.repo,
        base_commit=instance.base_commit,
        head_commit=instance.head_commit,
    )

    stat = diff_stat(repo_path, instance.base_commit, instance.head_commit)
    log(f"diff stat: {stat}")

    if preview:
        log("Preview mode: skipping review-pro call.")
        return {"instance_id": instance.instance_id, "status": "preview", "diff_stat": stat}

    results_dir.mkdir(parents=True, exist_ok=True)
    stale_partial = config.partial_path(results_dir, instance.instance_id).resolve()
    if stale_partial.exists():
        stale_partial.unlink()

    owns_home = review_pro_home is None
    home = install_review_pro_home(core_dir) if owns_home else review_pro_home
    stack_pack = install_stack_pack(repo_path, core_dir)

    try:
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.monotonic()
        review_process = run_review_pro(
            repo_path=repo_path,
            base_commit=instance.base_commit,
            head_commit=instance.head_commit,
            instance_id=instance.instance_id,
            results_dir=results_dir,
            claude_env=claude_env,
            review_pro_home=home,
            timeout_minutes=timeout_minutes,
        )
        duration_seconds = time.monotonic() - start_time

        out_path = save_result(
            results_dir=results_dir,
            instance=instance,
            review_process=review_process,
            started_at=started_at,
            duration_seconds=duration_seconds,
            stack_pack=stack_pack,
        )
    finally:
        if owns_home:
            shutil.rmtree(home, ignore_errors=True)

    # Drop review-pro's .review-pro/ and anything else left in the worktree.
    clean_worktree(repo_path)

    retry_exhausted = _check_and_log_retry_exhausted(out_path, results_dir)
    status = "ok" if review_process.returncode == 0 else "failed"
    log(
        f"--- [review-pro] {instance.instance_id} {status} "
        f"(exit={review_process.returncode}) -> {out_path}"
    )
    return {
        "instance_id": instance.instance_id,
        "status": status,
        "exit_code": review_process.returncode,
        "result_path": str(out_path),
        "retry_exhausted": retry_exhausted,
    }
