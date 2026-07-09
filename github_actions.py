"""GitHub REST API actions and client handoff packaging. Extracted from actions.py."""

import base64
import datetime
import json
import os
import re
import urllib.parse
import urllib.request
import zipfile

from config import GITHUB_API_BASE_URL, HANDOFF_EXCLUDE_PATTERNS
from actions import _resolve_path


# ---------------------------------------------------------------------------
# GitHub PR creation
# ---------------------------------------------------------------------------

def open_pr(
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str = "main",
) -> dict:
    """Open a GitHub pull request via the GitHub REST API.

    Args:
        repo:  Repository in ``owner/repo`` format.
        title: PR title.
        body:  PR body / description.
        head:  Head branch (branch to merge from).
        base:  Base branch (default: ``"main"``).

    Returns:
        dict with keys ``pr_url`` (str) and ``pr_number`` (int).

    Raises:
        RuntimeError: On any HTTP error — message includes status code and
            response body so the brain loop can report or replan.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable not set — cannot open PR"
        )

    url = f"{GITHUB_API_BASE_URL}/repos/{repo}/pulls"
    payload = json.dumps({
        "title": title,
        "body":  body,
        "head":  head,
        "base":  base,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization":        f"Bearer {token}",
            "Accept":               "application/vnd.github+json",
            "Content-Type":         "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_body = resp.read().decode("utf-8")
    except Exception as exc:
        status = getattr(exc, "code", "unknown")
        try:
            err_body = exc.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            err_body = str(exc)
        raise RuntimeError(
            f"GitHub API error {status} opening PR on {repo!r}: {err_body}"
        ) from exc

    data = json.loads(response_body)
    return {
        "pr_url":    data["html_url"],
        "pr_number": data["number"],
    }


# ---------------------------------------------------------------------------
# GitHub API helper + extended actions
# ---------------------------------------------------------------------------

def _github_api(method: str, path: str, payload: dict | None = None) -> dict | list:
    """Authenticated GitHub REST API call.

    Args:
        method:  HTTP verb — GET, POST, PUT, PATCH, DELETE.
        path:    API path starting with '/', e.g. '/user/repos'.
        payload: Optional JSON-serialisable request body.

    Returns:
        Parsed JSON response (dict or list). Empty dict for 204 No Content.

    Raises:
        RuntimeError: GITHUB_TOKEN missing or HTTP error — includes status + body.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError("GITHUB_TOKEN environment variable not set")

    url = f"{GITHUB_API_BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization":        f"Bearer {token}",
            "Accept":               "application/vnd.github+json",
            "Content-Type":         "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}
    except Exception as exc:
        status = getattr(exc, "code", "unknown")
        try:
            err_body = exc.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            err_body = str(exc)
        raise RuntimeError(
            f"GitHub API error {status} {method} {path}: {err_body}"
        ) from exc


def create_repo(
    name: str,
    private: bool = True,
    description: str = "",
    auto_init: bool = True,
) -> dict:
    """Create a new GitHub repository under the authenticated user.

    Args:
        name:        Repository name (spaces are replaced with hyphens by GitHub).
        private:     True → private repo (default). False → public.
        description: Optional short description.
        auto_init:   Initialise with an empty README commit (default True).

    Returns:
        dict with ``repo_url``, ``full_name``, ``clone_url``.
    """
    try:
        data = _github_api("POST", "/user/repos", {
            "name":        name,
            "private":     private,
            "description": description,
            "auto_init":   auto_init,
        })
    except RuntimeError as exc:
        # GitHub returns HTTP 422 when a repo with that name already exists.
        # Treat as success — fetch and return the existing repo rather than
        # raising an error (which would make Genie retry with a new name).
        msg = str(exc)
        if "422" in msg and "already exist" in msg.lower():
            try:
                me = _github_api("GET", "/user")
                login = me.get("login", "")  # type: ignore[union-attr]
                # GitHub silently replaces spaces with hyphens in repo names.
                safe_name = name.replace(" ", "-")
                data = _github_api("GET", f"/repos/{login}/{safe_name}")
                return {
                    "repo_url":       data["html_url"],    # type: ignore[index]
                    "full_name":      data["full_name"],   # type: ignore[index]
                    "clone_url":      data["clone_url"],   # type: ignore[index]
                    "already_existed": True,
                }
            except Exception:
                pass  # fall through to re-raise original error
        raise
    return {
        "repo_url":   data["html_url"],    # type: ignore[index]
        "full_name":  data["full_name"],   # type: ignore[index]
        "clone_url":  data["clone_url"],   # type: ignore[index]
    }


def delete_repo(repo: str) -> dict:
    """Permanently delete a GitHub repository.

    Args:
        repo: ``owner/repo`` format.

    Returns:
        ``{"status": "deleted", "repo": repo}``.
    """
    _github_api("DELETE", f"/repos/{repo}")
    return {"status": "deleted", "repo": repo}


def list_repos(user: str = "") -> list:
    """List repositories for the authenticated user (or another user).

    Args:
        user: GitHub username. Omit (or empty) to list your own repos.

    Returns:
        List of dicts with ``full_name``, ``private``, ``url``.
    """
    path = f"/users/{user}/repos?per_page=100" if user else "/user/repos?per_page=100"
    data = _github_api("GET", path)
    return [
        {"full_name": r["full_name"], "private": r["private"], "url": r["html_url"]}
        for r in data  # type: ignore[union-attr]
    ]


def fork_repo(repo: str) -> dict:
    """Fork a repository to the authenticated user's account.

    Args:
        repo: ``owner/repo`` format.

    Returns:
        dict with ``repo_url`` and ``full_name`` of the new fork.
    """
    data = _github_api("POST", f"/repos/{repo}/forks", {})
    return {
        "repo_url":  data["html_url"],   # type: ignore[index]
        "full_name": data["full_name"],  # type: ignore[index]
    }


def list_branches(repo: str) -> list:
    """List all branches in a repository.

    Args:
        repo: ``owner/repo`` format.

    Returns:
        List of branch name strings.
    """
    data = _github_api("GET", f"/repos/{repo}/branches?per_page=100")
    return [b["name"] for b in data]  # type: ignore[union-attr]


def create_branch(repo: str, branch: str, from_ref: str = "main") -> dict:
    """Create a new branch off an existing ref.

    Args:
        repo:     ``owner/repo`` format.
        branch:   Name for the new branch.
        from_ref: Existing branch/tag/SHA to branch from (default ``"main"``).

    Returns:
        ``{"status": "created", "branch": branch, "from_ref": from_ref}``.
    """
    ref_data = _github_api("GET", f"/repos/{repo}/git/ref/heads/{from_ref}")
    sha = ref_data["object"]["sha"]  # type: ignore[index]
    _github_api("POST", f"/repos/{repo}/git/refs", {
        "ref": f"refs/heads/{branch}",
        "sha": sha,
    })
    return {"status": "created", "branch": branch, "from_ref": from_ref}


def delete_branch(repo: str, branch: str) -> dict:
    """Delete a branch from a repository.

    Args:
        repo:   ``owner/repo`` format.
        branch: Branch name to delete.

    Returns:
        ``{"status": "deleted", "branch": branch}``.
    """
    _github_api("DELETE", f"/repos/{repo}/git/refs/heads/{branch}")
    return {"status": "deleted", "branch": branch}


def list_prs(repo: str, state: str = "open") -> list:
    """List pull requests for a repository.

    Args:
        repo:  ``owner/repo`` format.
        state: ``"open"``, ``"closed"``, or ``"all"`` (default ``"open"``).

    Returns:
        List of dicts with ``number``, ``title``, ``state``, ``url``.
    """
    data = _github_api("GET", f"/repos/{repo}/pulls?state={state}&per_page=50")
    return [
        {"number": p["number"], "title": p["title"], "state": p["state"], "url": p["html_url"]}
        for p in data  # type: ignore[union-attr]
    ]


def merge_pr(repo: str, pr_number: int, merge_method: str = "merge") -> dict:
    """Merge an open pull request.

    Args:
        repo:         ``owner/repo`` format.
        pr_number:    PR number to merge.
        merge_method: ``"merge"``, ``"squash"``, or ``"rebase"`` (default ``"merge"``).

    Returns:
        dict with ``sha``, ``merged`` (bool), ``message``.
    """
    data = _github_api("PUT", f"/repos/{repo}/pulls/{pr_number}/merge", {
        "merge_method": merge_method,
    })
    return {
        "sha":     data.get("sha", ""),        # type: ignore[union-attr]
        "merged":  data.get("merged", False),  # type: ignore[union-attr]
        "message": data.get("message", ""),    # type: ignore[union-attr]
    }


def close_pr(repo: str, pr_number: int) -> dict:
    """Close (without merging) an open pull request.

    Args:
        repo:      ``owner/repo`` format.
        pr_number: PR number to close.

    Returns:
        ``{"status": "closed", "pr_number": pr_number}``.
    """
    _github_api("PATCH", f"/repos/{repo}/pulls/{pr_number}", {"state": "closed"})
    return {"status": "closed", "pr_number": pr_number}


def create_issue(
    repo: str,
    title: str,
    body: str = "",
    labels: list | None = None,
) -> dict:
    """Create a new issue in a repository.

    Args:
        repo:   ``owner/repo`` format.
        title:  Issue title.
        body:   Issue body / description (default empty).
        labels: Optional list of label name strings.

    Returns:
        dict with ``issue_number`` and ``url``.
    """
    payload: dict = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    data = _github_api("POST", f"/repos/{repo}/issues", payload)
    return {"issue_number": data["number"], "url": data["html_url"]}  # type: ignore[index]


def close_issue(repo: str, issue_number: int) -> dict:
    """Close an open issue.

    Args:
        repo:         ``owner/repo`` format.
        issue_number: Issue number to close.

    Returns:
        ``{"status": "closed", "issue_number": issue_number}``.
    """
    _github_api("PATCH", f"/repos/{repo}/issues/{issue_number}", {
        "state": "closed",
    })
    return {"status": "closed", "issue_number": issue_number}


def list_issues(repo: str, state: str = "open") -> list:
    """List issues for a repository (excludes pull requests).

    Args:
        repo:  ``owner/repo`` format.
        state: ``"open"``, ``"closed"``, or ``"all"`` (default ``"open"``).

    Returns:
        List of dicts with ``number``, ``title``, ``state``, ``url``.
    """
    # GitHub /issues includes PRs; filter them out with is:issue in q
    data = _github_api("GET", f"/repos/{repo}/issues?state={state}&per_page=50")
    return [
        {"number": i["number"], "title": i["title"], "state": i["state"], "url": i["html_url"]}
        for i in data  # type: ignore[union-attr]
        if not i.get("pull_request")
    ]


def create_release(
    repo: str,
    tag: str,
    name: str,
    body: str = "",
    draft: bool = False,
    prerelease: bool = False,
) -> dict:
    """Create a GitHub release (and tag) for a repository.

    Args:
        repo:       ``owner/repo`` format.
        tag:        Git tag name (created if it doesn't exist).
        name:       Release title.
        body:       Release notes (default empty).
        draft:      True → save as draft (default False).
        prerelease: True → mark as pre-release (default False).

    Returns:
        dict with ``release_url`` and ``id``.
    """
    data = _github_api("POST", f"/repos/{repo}/releases", {
        "tag_name":   tag,
        "name":       name,
        "body":       body,
        "draft":      draft,
        "prerelease": prerelease,
    })
    return {"release_url": data["html_url"], "id": data["id"]}  # type: ignore[index]


def add_collaborator(repo: str, username: str, permission: str = "push") -> dict:
    """Invite a user as a collaborator on a repository.

    Args:
        repo:       ``owner/repo`` format.
        username:   GitHub username to invite.
        permission: ``"pull"``, ``"push"`` (default), ``"maintain"``, ``"triage"``,
                    or ``"admin"``.

    Returns:
        ``{"status": "invited", "username": username, "permission": permission}``.
    """
    _github_api("PUT", f"/repos/{repo}/collaborators/{username}", {
        "permission": permission,
    })
    return {"status": "invited", "username": username, "permission": permission}


def get_file_contents(repo: str, path: str, ref: str = "main") -> dict:
    """Read a file from a GitHub repository.

    Args:
        repo: ``owner/repo`` format.
        path: File path within the repo (e.g. ``"README.md"``).
        ref:  Branch, tag, or commit SHA (default ``"main"``).

    Returns:
        dict with ``content`` (decoded text), ``sha`` (blob SHA for updates), ``url``.
    """
    data = _github_api("GET", f"/repos/{repo}/contents/{path}?ref={ref}")
    content_b64 = data.get("content", "")  # type: ignore[union-attr]
    decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
    return {
        "content": decoded,
        "sha":     data.get("sha", ""),       # type: ignore[union-attr]
        "url":     data.get("html_url", ""),  # type: ignore[union-attr]
    }


def put_file(
    repo: str,
    path: str,
    content: str,
    message: str,
    sha: str = "",
    branch: str = "main",
) -> dict:
    """Create or update a file in a GitHub repository.

    Args:
        repo:    ``owner/repo`` format.
        path:    File path within the repo (e.g. ``"src/main.py"``).
        content: Full file content as a plain string.
        message: Commit message.
        sha:     Blob SHA of the existing file — required when updating an
                 existing file (get it from ``get_file_contents``).
                 Omit (or pass ``""``) when creating a new file.
        branch:  Target branch (default ``"main"``).

    Returns:
        dict with ``url`` (file HTML URL) and ``sha`` (new blob SHA).
    """
    payload: dict = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch":  branch,
    }
    if sha:
        payload["sha"] = sha
    data = _github_api("PUT", f"/repos/{repo}/contents/{path}", payload)
    return {
        "url": data["content"]["html_url"],  # type: ignore[index]
        "sha": data["content"]["sha"],       # type: ignore[index]
    }


# ---------------------------------------------------------------------------
# GitHub — repo management extras
# ---------------------------------------------------------------------------

def get_authenticated_user() -> dict:
    """Return the profile of the authenticated user.

    Returns:
        dict with ``login``, ``name``, ``email``, ``url``.
    """
    data = _github_api("GET", "/user")
    return {
        "login": data.get("login", ""),   # type: ignore[union-attr]
        "name":  data.get("name", ""),    # type: ignore[union-attr]
        "email": data.get("email", ""),   # type: ignore[union-attr]
        "url":   data.get("html_url", ""),# type: ignore[union-attr]
    }


def update_repo(
    repo: str,
    name: str = "",
    description: str = "",
    private: bool | None = None,
    homepage: str = "",
    has_issues: bool | None = None,
    has_wiki: bool | None = None,
    archived: bool | None = None,
) -> dict:
    """Update repository settings.

    Only the fields you pass are changed; omit a field to leave it unchanged.

    Args:
        repo:        ``owner/repo`` format.
        name:        New name (renames the repo).
        description: New description.
        private:     True → private, False → public.
        homepage:    New homepage URL.
        has_issues:  Enable/disable the Issues tab.
        has_wiki:    Enable/disable the Wiki tab.
        archived:    True → archive the repo (cannot be undone via API).

    Returns:
        dict with ``full_name``, ``repo_url``, ``private``.
    """
    payload: dict = {}
    if name:           payload["name"]        = name
    if description:    payload["description"] = description
    if private is not None:  payload["private"]     = private
    if homepage:       payload["homepage"]    = homepage
    if has_issues is not None: payload["has_issues"] = has_issues
    if has_wiki   is not None: payload["has_wiki"]   = has_wiki
    if archived   is not None: payload["archived"]   = archived
    data = _github_api("PATCH", f"/repos/{repo}", payload)
    return {
        "full_name": data.get("full_name", ""),   # type: ignore[union-attr]
        "repo_url":  data.get("html_url", ""),    # type: ignore[union-attr]
        "private":   data.get("private", False),  # type: ignore[union-attr]
    }


def set_repo_topics(repo: str, topics: list) -> dict:
    """Replace the full list of topics on a repository.

    Args:
        repo:   ``owner/repo`` format.
        topics: List of topic strings (lowercase, no spaces).

    Returns:
        ``{"topics": [...]}``.
    """
    data = _github_api("PUT", f"/repos/{repo}/topics", {"names": topics})
    return {"topics": data.get("names", [])}  # type: ignore[union-attr]


def search_repos(query: str, sort: str = "best-match", per_page: int = 10) -> list:
    """Search GitHub repositories.

    Args:
        query:    GitHub search query string (same syntax as the search bar).
        sort:     ``"best-match"``, ``"stars"``, ``"forks"``, ``"updated"``
                  (default ``"best-match"``).
        per_page: Max results to return (default 10, max 30).

    Returns:
        List of dicts with ``full_name``, ``description``, ``stars``, ``url``.
    """
    q = urllib.parse.quote(query)
    path = f"/search/repositories?q={q}&sort={sort}&per_page={min(per_page, 30)}"
    data = _github_api("GET", path)
    return [
        {
            "full_name":   r["full_name"],
            "description": r.get("description", ""),
            "stars":       r.get("stargazers_count", 0),
            "url":         r["html_url"],
        }
        for r in data.get("items", [])  # type: ignore[union-attr]
    ]


# ---------------------------------------------------------------------------
# GitHub — labels
# ---------------------------------------------------------------------------

def list_labels(repo: str) -> list:
    """List all labels in a repository.

    Args:
        repo: ``owner/repo`` format.

    Returns:
        List of dicts with ``name``, ``color``, ``description``.
    """
    data = _github_api("GET", f"/repos/{repo}/labels?per_page=100")
    return [
        {"name": l["name"], "color": l["color"], "description": l.get("description", "")}
        for l in data  # type: ignore[union-attr]
    ]


def create_label(
    repo: str,
    name: str,
    color: str,
    description: str = "",
) -> dict:
    """Create a new label in a repository.

    Args:
        repo:        ``owner/repo`` format.
        name:        Label name.
        color:       Hex colour without ``#`` (e.g. ``"e11d48"``).
        description: Optional description.

    Returns:
        dict with ``name``, ``color``, ``url``.
    """
    data = _github_api("POST", f"/repos/{repo}/labels", {
        "name":        name,
        "color":       color.lstrip("#"),
        "description": description,
    })
    return {
        "name":  data["name"],      # type: ignore[index]
        "color": data["color"],     # type: ignore[index]
        "url":   data["url"],       # type: ignore[index]
    }


# ---------------------------------------------------------------------------
# GitHub — branch protection
# ---------------------------------------------------------------------------

def protect_branch(
    repo: str,
    branch: str,
    required_approvals: int = 1,
    dismiss_stale_reviews: bool = False,
    require_code_owner_reviews: bool = False,
    require_status_checks: list | None = None,
) -> dict:
    """Set branch protection rules.

    Args:
        repo:                        ``owner/repo`` format.
        branch:                      Branch to protect.
        required_approvals:          Number of required approving reviews (0 to disable).
        dismiss_stale_reviews:       Dismiss approvals when new commits are pushed.
        require_code_owner_reviews:  Require review from code owners.
        require_status_checks:       List of status-check context strings that must pass.

    Returns:
        ``{"status": "protected", "branch": branch}``.
    """
    payload: dict = {
        "required_status_checks": (
            {"strict": True, "contexts": require_status_checks}
            if require_status_checks else None
        ),
        "enforce_admins": False,
        "required_pull_request_reviews": {
            "required_approving_review_count": required_approvals,
            "dismiss_stale_reviews":            dismiss_stale_reviews,
            "require_code_owner_reviews":       require_code_owner_reviews,
        } if required_approvals > 0 else None,
        "restrictions": None,
    }
    _github_api("PUT", f"/repos/{repo}/branches/{branch}/protection", payload)
    return {"status": "protected", "branch": branch}


# ---------------------------------------------------------------------------
# GitHub — webhooks
# ---------------------------------------------------------------------------

def list_webhooks(repo: str) -> list:
    """List webhooks for a repository.

    Args:
        repo: ``owner/repo`` format.

    Returns:
        List of dicts with ``id``, ``url``, ``events``, ``active``.
    """
    data = _github_api("GET", f"/repos/{repo}/hooks")
    return [
        {
            "id":     h["id"],
            "url":    h["config"].get("url", ""),
            "events": h.get("events", []),
            "active": h.get("active", False),
        }
        for h in data  # type: ignore[union-attr]
    ]


def create_webhook(
    repo: str,
    url: str,
    events: list | None = None,
    secret: str = "",
    active: bool = True,
) -> dict:
    """Create a repository webhook.

    Args:
        repo:   ``owner/repo`` format.
        url:    Payload delivery URL.
        events: List of event strings (default ``["push"]``).
        secret: Optional secret used to sign payloads.
        active: Whether the webhook should be active immediately (default True).

    Returns:
        dict with ``id`` and ``url``.
    """
    config: dict = {"url": url, "content_type": "json"}
    if secret:
        config["secret"] = secret
    data = _github_api("POST", f"/repos/{repo}/hooks", {
        "name":   "web",
        "config": config,
        "events": events or ["push"],
        "active": active,
    })
    return {"id": data["id"], "url": data["config"]["url"]}  # type: ignore[index]


def delete_webhook(repo: str, hook_id: int) -> dict:
    """Delete a repository webhook.

    Args:
        repo:    ``owner/repo`` format.
        hook_id: Webhook ID (from ``list_webhooks``).

    Returns:
        ``{"status": "deleted", "hook_id": hook_id}``.
    """
    _github_api("DELETE", f"/repos/{repo}/hooks/{hook_id}")
    return {"status": "deleted", "hook_id": hook_id}


# ---------------------------------------------------------------------------
# GitHub — Actions / workflows
# ---------------------------------------------------------------------------

def list_workflows(repo: str) -> list:
    """List GitHub Actions workflows in a repository.

    Args:
        repo: ``owner/repo`` format.

    Returns:
        List of dicts with ``id``, ``name``, ``path``, ``state``.
    """
    data = _github_api("GET", f"/repos/{repo}/actions/workflows")
    return [
        {"id": w["id"], "name": w["name"], "path": w["path"], "state": w["state"]}
        for w in data.get("workflows", [])  # type: ignore[union-attr]
    ]


def trigger_workflow(
    repo: str,
    workflow: str,
    ref: str = "main",
    inputs: dict | None = None,
) -> dict:
    """Trigger a GitHub Actions workflow dispatch event.

    Args:
        repo:     ``owner/repo`` format.
        workflow: Workflow file name (e.g. ``"ci.yml"``) or numeric workflow ID.
        ref:      Branch or tag to run on (default ``"main"``).
        inputs:   Optional dict of workflow_dispatch input values.

    Returns:
        ``{"status": "triggered", "workflow": workflow, "ref": ref}``.
    """
    _github_api("POST", f"/repos/{repo}/actions/workflows/{workflow}/dispatches", {
        "ref":    ref,
        "inputs": inputs or {},
    })
    return {"status": "triggered", "workflow": workflow, "ref": ref}


def list_workflow_runs(
    repo: str,
    workflow: str = "",
    status: str = "",
    per_page: int = 10,
) -> list:
    """List GitHub Actions workflow runs.

    Args:
        repo:     ``owner/repo`` format.
        workflow: Workflow filename to filter by (e.g. ``"ci.yml"``). Omit for all.
        status:   Filter by status: ``"completed"``, ``"in_progress"``,
                  ``"queued"``, ``"success"``, ``"failure"`` etc.
        per_page: Max results (default 10).

    Returns:
        List of dicts with ``id``, ``name``, ``status``, ``conclusion``, ``url``, ``branch``.
    """
    if workflow:
        path = f"/repos/{repo}/actions/workflows/{workflow}/runs?per_page={per_page}"
    else:
        path = f"/repos/{repo}/actions/runs?per_page={per_page}"
    if status:
        path += f"&status={status}"
    data = _github_api("GET", path)
    return [
        {
            "id":         r["id"],
            "name":       r.get("name", ""),
            "status":     r.get("status", ""),
            "conclusion": r.get("conclusion", ""),
            "branch":     r.get("head_branch", ""),
            "url":        r.get("html_url", ""),
        }
        for r in data.get("workflow_runs", [])  # type: ignore[union-attr]
    ]


# ---------------------------------------------------------------------------
# GitHub — gists
# ---------------------------------------------------------------------------

def create_gist(
    description: str,
    files: dict,
    public: bool = False,
) -> dict:
    """Create a GitHub Gist.

    Args:
        description: Short description of the gist.
        files:       Dict mapping filename → content string,
                     e.g. ``{"hello.py": "print('hello')"}``.
        public:      True → public gist (default False → secret).

    Returns:
        dict with ``gist_url`` and ``id``.
    """
    data = _github_api("POST", "/gists", {
        "description": description,
        "public":      public,
        "files":       {k: {"content": v} for k, v in files.items()},
    })
    return {"gist_url": data["html_url"], "id": data["id"]}  # type: ignore[index]


def list_gists(user: str = "") -> list:
    """List gists for the authenticated user (or another user).

    Args:
        user: GitHub username. Omit for your own gists.

    Returns:
        List of dicts with ``id``, ``description``, ``url``, ``public``.
    """
    path = f"/users/{user}/gists?per_page=30" if user else "/gists?per_page=30"
    data = _github_api("GET", path)
    return [
        {
            "id":          g["id"],
            "description": g.get("description", ""),
            "url":         g["html_url"],
            "public":      g.get("public", False),
        }
        for g in data  # type: ignore[union-attr]
    ]


# ---------------------------------------------------------------------------
# GitHub — stars
# ---------------------------------------------------------------------------

def star_repo(repo: str) -> dict:
    """Star a repository.

    Args:
        repo: ``owner/repo`` format.

    Returns:
        ``{"status": "starred", "repo": repo}``.
    """
    _github_api("PUT", f"/user/starred/{repo}")
    return {"status": "starred", "repo": repo}


def unstar_repo(repo: str) -> dict:
    """Unstar a repository.

    Args:
        repo: ``owner/repo`` format.

    Returns:
        ``{"status": "unstarred", "repo": repo}``.
    """
    _github_api("DELETE", f"/user/starred/{repo}")
    return {"status": "unstarred", "repo": repo}


# ---------------------------------------------------------------------------
# GitHub — organisation
# ---------------------------------------------------------------------------

def create_org_repo(
    org: str,
    name: str,
    private: bool = True,
    description: str = "",
    auto_init: bool = True,
) -> dict:
    """Create a repository under a GitHub organisation.

    Args:
        org:         Organisation login name.
        name:        Repository name.
        private:     True → private (default). False → public.
        description: Optional description.
        auto_init:   Initialise with an empty README (default True).

    Returns:
        dict with ``repo_url``, ``full_name``, ``clone_url``.
    """
    data = _github_api("POST", f"/orgs/{org}/repos", {
        "name":        name,
        "private":     private,
        "description": description,
        "auto_init":   auto_init,
    })
    return {
        "repo_url":  data["html_url"],   # type: ignore[index]
        "full_name": data["full_name"],  # type: ignore[index]
        "clone_url": data["clone_url"],  # type: ignore[index]
    }


def list_org_members(org: str) -> list:
    """List public members of a GitHub organisation.

    Args:
        org: Organisation login name.

    Returns:
        List of dicts with ``login`` and ``url``.
    """
    data = _github_api("GET", f"/orgs/{org}/members?per_page=100")
    return [
        {"login": m["login"], "url": m["html_url"]}
        for m in data  # type: ignore[union-attr]
    ]


def list_teams(org: str) -> list:
    """List teams in a GitHub organisation.

    Args:
        org: Organisation login name.

    Returns:
        List of dicts with ``id``, ``name``, ``slug``.
    """
    data = _github_api("GET", f"/orgs/{org}/teams?per_page=100")
    return [
        {"id": t["id"], "name": t["name"], "slug": t["slug"]}
        for t in data  # type: ignore[union-attr]
    ]


# ---------------------------------------------------------------------------
# GitHub — packages
# ---------------------------------------------------------------------------

def list_packages(package_type: str = "container", user: str = "") -> list:
    """List packages for the authenticated user (or an org).

    Args:
        package_type: One of ``"npm"``, ``"maven"``, ``"rubygems"``, ``"docker"``,
                      ``"nuget"``, ``"container"`` (default ``"container"``).
        user:         Username/org to query. Omit for the authenticated user.

    Returns:
        List of dicts with ``id``, ``name``, ``url``.
    """
    if user:
        path = f"/users/{user}/packages?package_type={package_type}&per_page=50"
    else:
        path = f"/user/packages?package_type={package_type}&per_page=50"
    data = _github_api("GET", path)
    return [
        {"id": p["id"], "name": p["name"], "url": p.get("url", "")}
        for p in data  # type: ignore[union-attr]
    ]


def delete_package_version(
    package_type: str,
    package_name: str,
    version_id: int,
) -> dict:
    """Delete a specific version of a user-owned package.

    Args:
        package_type: Package type (e.g. ``"container"``).
        package_name: Package name.
        version_id:   Version ID (from ``list_packages`` or the API).

    Returns:
        ``{"status": "deleted", "version_id": version_id}``.
    """
    _github_api("DELETE", f"/user/packages/{package_type}/{package_name}/versions/{version_id}")
    return {"status": "deleted", "version_id": version_id}


# ---------------------------------------------------------------------------
# GitHub — notifications
# ---------------------------------------------------------------------------

def list_notifications(all_: bool = False) -> list:
    """List notifications for the authenticated user.

    Args:
        all_: True → include already-read notifications (default False).

    Returns:
        List of dicts with ``id``, ``title``, ``type``, ``repo``, ``url``.
    """
    path = f"/notifications?all={'true' if all_ else 'false'}&per_page=30"
    data = _github_api("GET", path)
    return [
        {
            "id":    n["id"],
            "title": n["subject"].get("title", ""),
            "type":  n["subject"].get("type", ""),
            "repo":  n["repository"]["full_name"],
            "url":   n["subject"].get("url", ""),
        }
        for n in data  # type: ignore[union-attr]
    ]


def mark_notifications_read(repo: str = "") -> dict:
    """Mark notifications as read.

    Args:
        repo: ``owner/repo`` — mark only notifications for this repo.
              Omit to mark ALL notifications as read.

    Returns:
        ``{"status": "marked_read"}``.
    """
    if repo:
        _github_api("PUT", f"/repos/{repo}/notifications", {})
    else:
        _github_api("PUT", "/notifications", {})
    return {"status": "marked_read"}


# ---------------------------------------------------------------------------
# GitHub — audit log
# ---------------------------------------------------------------------------

def get_audit_log(
    org: str,
    phrase: str = "",
    per_page: int = 25,
) -> list:
    """Fetch the audit log for a GitHub organisation.

    Args:
        org:      Organisation login name.
        phrase:   Optional search filter (same syntax as the audit log UI).
        per_page: Max events to return (default 25).

    Returns:
        List of raw audit log event dicts (``action``, ``actor``, ``created_at``, etc.).
    """
    path = f"/orgs/{org}/audit-log?per_page={per_page}"
    if phrase:
        path += f"&phrase={urllib.parse.quote(phrase)}"
    data = _github_api("GET", path)
    return list(data)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Handoff package
# ---------------------------------------------------------------------------

_HANDOFF_SECRET_PATTERNS = [
    re.compile(r'-----BEGIN .{0,20}PRIVATE KEY'),
    re.compile(r'(?i)(api_key|token|secret|password)\s*[=:]\s*\S{8,}'),
    re.compile(r'AKIA[0-9A-Z]{16}'),
]

_HANDOFF_MD_TEMPLATE = """\
# Handoff Package

## What Was Delivered
{task_summary}

## Setup
1. Unzip this package into your working directory.
2. Install dependencies: `pip install -r requirements.txt` (or equivalent).
3. Copy `.env.example` to `.env` and fill in required values.

## Endpoints
{endpoints_block}

## Notes
- All secrets and credentials have been excluded from this package.
- Generated by Genie on {iso_date}.
"""


def _handoff_excluded(rel_path: str) -> bool:
    """Return True if *rel_path* matches any HANDOFF_EXCLUDE_PATTERNS entry."""
    for pattern in HANDOFF_EXCLUDE_PATTERNS:
        stripped = pattern.lstrip("*")
        if stripped in rel_path or rel_path.endswith(stripped):
            return True
    return False


def assemble_handoff(
    repo_path: str,
    task_summary: str,
    output_dir: str = "",
    endpoints: list[str] | None = None,
) -> dict:
    """Package *repo_path* into a handoff zip with a HANDOFF.md manifest.

    Args:
        repo_path:    Path to the repository / project directory to package.
        task_summary: Free-text description of what was delivered.
        output_dir:   Destination directory for the zip and HANDOFF.md.
                      Defaults to ``<repo_path>_handoff`` (sibling directory).
        endpoints:    Optional list of live endpoint URLs to document.

    Returns:
        ``{"package_path": str, "handoff_md_path": str, "secrets_clean": True}``

    Raises:
        RuntimeError: If any included file contains secrets patterns.
        OSError:      If the zip file cannot be written.
    """
    repo_path = _resolve_path(repo_path)

    if not output_dir:
        output_dir = repo_path.rstrip("/") + "_handoff"
    os.makedirs(output_dir, exist_ok=True)

    # Collect all files and filter exclusions
    included: list[tuple[str, str]] = []  # (abs_path, rel_path)
    for dirpath, _dirnames, filenames in os.walk(repo_path):
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(abs_path, repo_path)
            if _handoff_excluded(rel_path):
                continue
            included.append((abs_path, rel_path))

    # Secrets scan — read text, skip binary files silently
    flagged: list[str] = []
    for abs_path, rel_path in included:
        try:
            text = open(abs_path, encoding="utf-8", errors="strict").read()
        except (UnicodeDecodeError, OSError):
            continue
        for pat in _HANDOFF_SECRET_PATTERNS:
            if pat.search(text):
                flagged.append(rel_path)
                break
    if flagged:
        raise RuntimeError(f"secrets scan failed: {flagged}")

    # Build HANDOFF.md content
    if endpoints:
        endpoints_block = "\n".join(f"- {url}" for url in endpoints)
    else:
        endpoints_block = "No live endpoints for this task."
    iso_date = datetime.date.today().isoformat()
    handoff_md = _HANDOFF_MD_TEMPLATE.format(
        task_summary=task_summary,
        endpoints_block=endpoints_block,
        iso_date=iso_date,
    )

    # Write zip (secrets clean — proceed)
    zip_path = os.path.join(output_dir, "handoff.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_path, rel_path in included:
            zf.write(abs_path, arcname=rel_path)
        zf.writestr("HANDOFF.md", handoff_md)

    # Also write standalone HANDOFF.md
    handoff_md_path = os.path.join(output_dir, "HANDOFF.md")
    with open(handoff_md_path, "w", encoding="utf-8") as fh:
        fh.write(handoff_md)

    return {
        "package_path":    zip_path,
        "handoff_md_path": handoff_md_path,
        "secrets_clean":   True,
    }
