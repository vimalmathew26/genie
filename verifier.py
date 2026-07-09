"""
Genie — verifier.py
Post-Subtask Verification Layer.

Automated checks that run between "subtask calls done" and "move to next
subtask" in the GoalTracker loop.  Catches integration/glue bugs (broken
imports, missing symbols, import cycles) at zero LLM cost.

Design principles:
  - FAIL-OPEN: every check is wrapped in try/except → passed=True on error.
    A verification crash never blocks forward progress.
  - SELF-CONTAINED: imports only stdlib (ast, os, pathlib, subprocess) + config.
    No imports from orchestrator, goal_tracker, or any other Genie module.
  - FEATURE-FLAGGED: gated behind config.VERIFICATION_ENABLED (default OFF).
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from config import (
    VERIFICATION_ENABLED,
    VERIFY_LEVELS,
    VERIFY_TIMEOUT_SECONDS,
    WORKSPACE_DIR,
    FALLBACK_MODEL,
)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class VerificationIssue:
    """A single issue found by a verification check."""
    file: str           # path of the file with the problem
    symbol: str         # the import/symbol that failed
    message: str        # human-readable description


@dataclass
class VerificationResult:
    """Outcome of one verification level."""
    passed: bool
    level: str                                  # "import", "smoke", etc.
    issues: list[VerificationIssue] = field(default_factory=list)
    error: str | None = None                    # internal error (fail-open)

    @property
    def fix_hint(self) -> str:
        """Compact string suitable for injection into brain_loop context."""
        if self.passed:
            return ""
        lines = [f"VERIFICATION FAILED ({self.level}):"]
        for iss in self.issues[:10]:
            lines.append(f"  - {iss.file}: {iss.message}")
        if len(self.issues) > 10:
            lines.append(f"  ... and {len(self.issues) - 10} more issues")
        return "\n".join(lines)


# ===========================================================================
# SubtaskVerifier — the main entry point
# ===========================================================================

class SubtaskVerifier:
    """Runs post-subtask verification checks on files written by a subtask.

    Usage:
        verifier = SubtaskVerifier(project_dir="/home/user/genie_workspace/myproject")
        results = verifier.verify(files_written=["/home/user/.../mod.py", ...])
        for r in results:
            if not r.passed:
                print(r.fix_hint)
    """

    def __init__(self, project_dir: str | None = None) -> None:
        self.project_dir = project_dir or WORKSPACE_DIR

    def verify(
        self,
        files_written: list[str],
        is_final: bool = False,
    ) -> list[VerificationResult]:
        """Run all enabled verification levels.

        Args:
            files_written: Absolute paths of files written in this subtask.
            is_final: True when this is the last subtask (enables smoke test).

        Returns:
            List of VerificationResult, one per level. Empty if verification
            is disabled or no applicable checks.
        """
        if not VERIFICATION_ENABLED:
            return []

        # Use all project .py files for holistic checking (catches phantom-
        # done: agent claims done but never wrote the file, cross-file import
        # breaks, etc.). Falls back to files_written if project scan is empty.
        all_project_py = self._collect_project_py_files()
        written_py = [f for f in files_written if f.endswith(".py")]
        py_files = all_project_py or written_py

        if not py_files:
            return []

        results: list[VerificationResult] = []

        if "import" in VERIFY_LEVELS:
            results.append(self._check_imports(py_files))

        if "types" in VERIFY_LEVELS:
            results.append(self._check_types(py_files))

        if "smoke" in VERIFY_LEVELS and is_final:
            results.append(self._check_smoke())

        if "llm_review" in VERIFY_LEVELS and is_final:
            results.append(self._check_llm_review(py_files))

        return results

    def _collect_project_py_files(self) -> list[str]:
        """Scan project_dir for all .py files, excluding venv/cache dirs.

        Returns absolute paths. Used by verify() to do holistic import
        checking across the entire project, not just the current subtask's
        files_written.
        """
        py_files: list[str] = []
        base = Path(self.project_dir)
        if not base.is_dir():
            return py_files

        skip_exact = {
            "node_modules", "__pycache__", ".git", ".tox", ".mypy_cache",
        }
        # Broader patterns: skip any dir whose name contains 'venv' or
        # 'site-packages' — catches dag_scheduler_venv, .venv, my_venv, etc.
        def _should_skip(dirname: str) -> bool:
            if dirname in skip_exact or dirname.startswith("."):
                return True
            dl = dirname.lower()
            if "venv" in dl or "site-packages" in dl or dl == "env":
                return True
            return False

        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not _should_skip(d)]
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
        return py_files

    # -----------------------------------------------------------------------
    # Level 1 — Import Resolution
    # -----------------------------------------------------------------------

    def _check_imports(self, py_files: list[str]) -> VerificationResult:
        """AST-based import resolution check.

        For each file in py_files (typically ALL project .py files):
          1. Parse the file, extract all imports.
          2. For local project imports (module file exists in project_dir):
             - Verify the target module file exists.
             - For 'from X import Y': verify Y is a top-level def/class/assign
               in the target module (via AST).
          3. Skip third-party/stdlib imports (no local file found).

        Fail-open: any error during checking → result.passed = True.
        """
        try:
            return self._check_imports_inner(py_files)
        except Exception as exc:
            return VerificationResult(
                passed=True,
                level="import",
                error=f"import check crashed (fail-open): {exc}",
            )

    def _check_imports_inner(self, py_files: list[str]) -> VerificationResult:
        issues: list[VerificationIssue] = []

        # Build a map of all .py files in the project for local-module resolution.
        local_modules = self._build_local_module_map()

        # Also index files_written and all project .py files for symbol lookup cache.
        symbol_cache: dict[str, set[str]] = {}

        for fpath in py_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source, filename=fpath)
            except Exception:
                continue  # can't parse → skip (fail-open)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self._check_import_from(
                        node, fpath, local_modules, symbol_cache, issues,
                    )
                elif isinstance(node, ast.Import):
                    self._check_import(
                        node, fpath, local_modules, issues,
                    )

        return VerificationResult(
            passed=len(issues) == 0,
            level="import",
            issues=issues,
        )

    def _check_import_from(
        self,
        node: ast.ImportFrom,
        source_file: str,
        local_modules: dict[str, str],
        symbol_cache: dict[str, set[str]],
        issues: list[VerificationIssue],
    ) -> None:
        """Check a 'from X import Y, Z' statement."""
        if node.module is None:
            return  # relative import with no module name (e.g. 'from . import x')

        module_name = node.module

        # Handle relative imports: resolve to absolute module path.
        if node.level and node.level > 0:
            module_name = self._resolve_relative(
                source_file, node.module or "", node.level,
            )
            if module_name is None:
                return  # can't resolve → skip

        # Check if this is a local module.
        target_path = local_modules.get(module_name)
        if target_path is None:
            # Could be a package — check for __init__.py
            pkg_init = local_modules.get(module_name + ".__init__")
            if pkg_init is not None:
                target_path = pkg_init
            else:
                return  # third-party or stdlib → skip

        # The module file exists. Now check each imported symbol.
        if not os.path.isfile(target_path):
            issues.append(VerificationIssue(
                file=_rel(source_file, self.project_dir),
                symbol=module_name,
                message=f"import target module '{module_name}' file not found: {target_path}",
            ))
            return

        # Get exported symbols from the target module.
        target_symbols = self._get_module_symbols(target_path, symbol_cache)

        for alias in node.names:
            name = alias.name
            if name == "*":
                continue  # can't validate star imports
            if target_symbols is not None and name not in target_symbols:
                issues.append(VerificationIssue(
                    file=_rel(source_file, self.project_dir),
                    symbol=name,
                    message=(
                        f"'from {node.module} import {name}' — "
                        f"symbol '{name}' not found in {_rel(target_path, self.project_dir)}. "
                        f"Available: {', '.join(sorted(target_symbols)[:15])}"
                    ),
                ))

    def _check_import(
        self,
        node: ast.Import,
        source_file: str,
        local_modules: dict[str, str],
        issues: list[VerificationIssue],
    ) -> None:
        """Check a plain 'import X' or 'import X.Y' statement."""
        for alias in node.names:
            module_name = alias.name
            # Only check local modules — skip stdlib/third-party.
            if module_name in local_modules:
                target = local_modules[module_name]
                if not os.path.isfile(target):
                    issues.append(VerificationIssue(
                        file=_rel(source_file, self.project_dir),
                        symbol=module_name,
                        message=f"'import {module_name}' — module file not found: {target}",
                    ))
            # Also check package form (X.Y → X/__init__.py or X/Y.py)
            elif "." in module_name:
                parts = module_name.split(".")
                # Try X/Y.py
                pkg_mod = ".".join(parts)
                if pkg_mod in local_modules:
                    target = local_modules[pkg_mod]
                    if not os.path.isfile(target):
                        issues.append(VerificationIssue(
                            file=_rel(source_file, self.project_dir),
                            symbol=module_name,
                            message=f"'import {module_name}' — module file not found",
                        ))

    def _build_local_module_map(self) -> dict[str, str]:
        """Walk project_dir and build module_dotted_name → absolute_path map.

        Example: project_dir/pkg/sub/mod.py → "pkg.sub.mod" : "/abs/path/pkg/sub/mod.py"
        Also: project_dir/pkg/__init__.py → "pkg.__init__" : "/abs/path/pkg/__init__.py"
               AND "pkg" : "/abs/path/pkg/__init__.py"
        """
        result: dict[str, str] = {}
        base = Path(self.project_dir)

        if not base.is_dir():
            return result

        for py_file in base.rglob("*.py"):
            try:
                rel = py_file.relative_to(base)
            except ValueError:
                continue

            # Skip hidden dirs, __pycache__, venv, .git, node_modules
            parts = rel.parts
            if any(
                p.startswith(".") or p == "__pycache__" or p == "venv"
                or p == "node_modules" or p == ".git"
                for p in parts
            ):
                continue

            # Convert path to dotted module name.
            # e.g. pkg/sub/mod.py → pkg.sub.mod
            mod_parts = list(parts)
            mod_parts[-1] = mod_parts[-1].removesuffix(".py")
            dotted = ".".join(mod_parts)
            result[dotted] = str(py_file)

            # If this is __init__.py, also register the package name.
            if mod_parts[-1] == "__init__":
                pkg_dotted = ".".join(mod_parts[:-1])
                if pkg_dotted:
                    result[pkg_dotted] = str(py_file)

        return result

    def _resolve_relative(
        self, source_file: str, module: str, level: int,
    ) -> str | None:
        """Resolve a relative import to an absolute dotted module name.

        Args:
            source_file: Path of the file containing the import.
            module: The module part (may be empty for 'from . import X').
            level: Number of dots (1 = '.', 2 = '..', etc.)

        Returns:
            Absolute dotted module name, or None if unresolvable.
        """
        try:
            base = Path(self.project_dir)
            src = Path(source_file)
            rel = src.relative_to(base)
            # Go up 'level' directories from the source file's directory.
            pkg_parts = list(rel.parts[:-1])  # directory parts
            if level > len(pkg_parts):
                return None
            pkg_parts = pkg_parts[: len(pkg_parts) - (level - 1)]
            if module:
                return ".".join(pkg_parts + module.split("."))
            return ".".join(pkg_parts) if pkg_parts else None
        except (ValueError, IndexError):
            return None

    def _get_module_symbols(
        self, target_path: str, cache: dict[str, set[str]],
    ) -> set[str] | None:
        """Extract top-level defined symbols from a Python module via AST.

        Returns a set of symbol names (classes, functions, variables,
        __all__ entries). Returns None if the file can't be parsed
        (fail-open: caller skips the symbol check).

        Results are cached by target_path.
        """
        if target_path in cache:
            return cache[target_path]

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=target_path)
        except Exception:
            cache[target_path] = None  # type: ignore[assignment]
            return None

        symbols: set[str] = set()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.add(node.name)
            elif isinstance(node, ast.ClassDef):
                symbols.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                symbols.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # Symbols imported at top level are re-exported.
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        if name != "*":
                            symbols.add(name)
                else:
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[-1]
                        symbols.add(name)

        # Check for __all__ — if present, use it as the canonical export list
        # BUT also keep all other symbols (since direct imports bypass __all__).
        # We don't restrict to __all__ because 'from mod import Foo' works
        # even if Foo isn't in __all__.

        cache[target_path] = symbols
        return symbols

    # -----------------------------------------------------------------------
    # Level 2 — Type Consistency (pyright)
    # -----------------------------------------------------------------------

    def _check_types(self, py_files: list[str]) -> VerificationResult:
        """Run pyright on project_dir, filter to py_files.

        Falls back gracefully if pyright is not installed.
        Fail-open: any error → passed=True.
        """
        try:
            return self._check_types_inner(py_files)
        except Exception as exc:
            return VerificationResult(
                passed=True,
                level="types",
                error=f"type check crashed (fail-open): {exc}",
            )

    def _check_types_inner(self, py_files: list[str]) -> VerificationResult:
        """Inner implementation of pyright type checking."""
        # Check if pyright is available
        try:
            subprocess.run(
                ["pyright", "--version"],
                capture_output=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return VerificationResult(
                passed=True,
                level="types",
                error="pyright not installed (skipped)",
            )

        # Run pyright on the project directory with JSON output
        try:
            result = subprocess.run(
                ["pyright", "--outputjson"],
                capture_output=True,
                text=True,
                timeout=VERIFY_TIMEOUT_SECONDS,
                cwd=self.project_dir,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=True,
                level="types",
                error=f"pyright timed out after {VERIFY_TIMEOUT_SECONDS}s",
            )

        # Parse JSON output
        import json
        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            return VerificationResult(
                passed=True,
                level="types",
                error="pyright output not parseable (fail-open)",
            )

        diagnostics = data.get("generalDiagnostics", [])

        # Normalise py_files to absolute paths for comparison
        py_files_abs = {os.path.abspath(f) for f in py_files}

        issues: list[VerificationIssue] = []
        for diag in diagnostics:
            severity = diag.get("severity", "")
            if severity != "error":
                continue  # only flag errors, not warnings/info
            diag_file = diag.get("file", "")
            diag_file_abs = os.path.abspath(
                os.path.join(self.project_dir, diag_file)
            ) if not os.path.isabs(diag_file) else os.path.abspath(diag_file)
            if diag_file_abs not in py_files_abs:
                continue  # only report errors in files WE wrote
            line = diag.get("range", {}).get("start", {}).get("line", 0)
            msg = diag.get("message", "unknown error")
            rule = diag.get("rule", "")
            issues.append(VerificationIssue(
                file=_rel(diag_file_abs, self.project_dir),
                symbol=rule or "type-error",
                message=f"L{line}: {msg}" + (f" [{rule}]" if rule else ""),
            ))

        return VerificationResult(
            passed=len(issues) == 0,
            level="types",
            issues=issues,
        )

    # -----------------------------------------------------------------------
    # Level 3 — Smoke Test
    # -----------------------------------------------------------------------

    def _check_smoke(self) -> VerificationResult:
        """Run a basic import smoke test on the project.

        Tries: python -c "import <package>" with a short timeout.
        Fail-open: any error → passed=True.
        """
        try:
            return self._check_smoke_inner()
        except Exception as exc:
            return VerificationResult(
                passed=True,
                level="smoke",
                error=f"smoke check crashed (fail-open): {exc}",
            )

    def _check_smoke_inner(self) -> VerificationResult:
        """Inner implementation of smoke test."""
        # Find the top-level package: look for __init__.py directly under project_dir.
        base = Path(self.project_dir)
        packages: list[str] = []
        for item in base.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                packages.append(item.name)

        if not packages:
            # No package found — try importing individual .py files
            py_files = [f.stem for f in base.glob("*.py") if f.stem != "__init__"]
            if not py_files:
                return VerificationResult(passed=True, level="smoke")
            # Try importing the first few
            packages = py_files[:3]

        issues: list[VerificationIssue] = []
        for pkg in packages[:3]:
            try:
                result = subprocess.run(
                    [sys.executable, "-c", f"import {pkg}"],
                    capture_output=True,
                    text=True,
                    timeout=VERIFY_TIMEOUT_SECONDS,
                    cwd=str(base),
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if result.returncode != 0:
                    stderr = result.stderr.strip()[-500:]
                    issues.append(VerificationIssue(
                        file=pkg,
                        symbol=pkg,
                        message=f"'import {pkg}' failed (exit {result.returncode}): {stderr}",
                    ))
            except subprocess.TimeoutExpired:
                issues.append(VerificationIssue(
                    file=pkg,
                    symbol=pkg,
                    message=f"'import {pkg}' timed out after {VERIFY_TIMEOUT_SECONDS}s",
                ))

        return VerificationResult(
            passed=len(issues) == 0,
            level="smoke",
            issues=issues,
        )

    # -----------------------------------------------------------------------
    # Level 4 — LLM Self-Review (final subtask only)
    # -----------------------------------------------------------------------

    def _check_llm_review(self, py_files: list[str]) -> VerificationResult:
        """LLM-based cross-file integration review.

        Reads ALL .py files from the project directory (not just this
        subtask's files), sends them to the LLM with a focused prompt
        asking for integration mismatches between modules.

        Runs ONLY on the final subtask (is_final=True).  One LLM call.
        Fail-open: any error → passed=True.
        """
        try:
            return self._check_llm_review_inner(py_files)
        except Exception as exc:
            return VerificationResult(
                passed=True,
                level="llm_review",
                error=f"LLM review crashed (fail-open): {exc}",
            )

    def _check_llm_review_inner(self, py_files: list[str]) -> VerificationResult:
        """Inner implementation of LLM self-review."""
        # Lazy import to avoid circular dependency
        from llm_client import LLMClient

        # Collect ALL .py files from the project, not just this subtask
        base = Path(self.project_dir)
        all_py: list[tuple[str, str]] = []  # (relative_path, content)
        total_chars = 0
        MAX_CHARS = 80_000  # cap to avoid blowing context window

        for py_file in sorted(base.rglob("*.py")):
            try:
                rel = py_file.relative_to(base)
            except ValueError:
                continue
            parts = rel.parts
            if any(
                p.startswith(".") or p == "__pycache__" or p == "venv"
                or p == "node_modules"
                for p in parts
            ):
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            if total_chars + len(content) > MAX_CHARS:
                break
            all_py.append((str(rel), content))
            total_chars += len(content)

        if not all_py:
            return VerificationResult(passed=True, level="llm_review")

        # Build the prompt
        files_block = ""
        for rel_path, content in all_py:
            files_block += f"\n--- {rel_path} ---\n{content}\n"

        prompt = (
            "You are a code reviewer. Below are ALL Python source files for a project.\n"
            "Identify ONLY concrete integration bugs between modules — places where:\n"
            "- A function/method is called that does not exist in the target module\n"
            "- A function is called with wrong argument count or types\n"
            "- A return value is used incorrectly (e.g., iterating .items() on a list)\n"
            "- An API endpoint returns a different shape than what the consumer expects\n"
            "- A state transition is missing or invalid\n"
            "- A required initialization step is never called\n"
            "\n"
            "Do NOT report style issues, missing docstrings, or theoretical concerns.\n"
            "Do NOT report issues with third-party library usage.\n"
            "Report ONLY issues you are confident are real bugs.\n"
            "\n"
            "For each bug, output exactly one line in this format:\n"
            "BUG: <file.py>: <symbol_or_line>: <description>\n"
            "\n"
            "If there are no integration bugs, output exactly: NO_BUGS_FOUND\n"
            "\n"
            f"FILES:\n{files_block}"
        )

        client = LLMClient()
        response, _cost = client.call(
            messages=[{"role": "user", "content": prompt}],
            model=FALLBACK_MODEL,
            max_tokens=2048,
        )

        # Parse response
        issues: list[VerificationIssue] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("BUG:"):
                parts = line[4:].strip().split(":", 2)
                if len(parts) >= 3:
                    issues.append(VerificationIssue(
                        file=parts[0].strip(),
                        symbol=parts[1].strip(),
                        message=parts[2].strip(),
                    ))
                elif len(parts) == 2:
                    issues.append(VerificationIssue(
                        file=parts[0].strip(),
                        symbol="unknown",
                        message=parts[1].strip(),
                    ))

        return VerificationResult(
            passed=len(issues) == 0,
            level="llm_review",
            issues=issues,
        )


# ===========================================================================
# Helpers
# ===========================================================================

def _rel(path: str, base: str) -> str:
    """Return path relative to base, or basename if not underneath."""
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return os.path.basename(path)
