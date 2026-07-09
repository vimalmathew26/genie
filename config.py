"""
Genie Configuration Module

Configuration settings for the Genie desktop automation assistant.
Loads environment variables from .env file and provides application constants.
"""

import os
from datetime import datetime

# Attempt to load dotenv - handle gracefully if .env file doesn't exist
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[GENIE] [{timestamp}] python-dotenv not installed. Using system environment variables only.")
except Exception as e:
    # .env file may not exist - that's okay, we'll use system env vars
    pass


# =============================================================================
# API Configuration
# =============================================================================

# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# GitHub API base URL — used by actions.open_pr() and any future GitHub actions.
# Kept here so tests can monkeypatch it without touching action code.
GITHUB_API_BASE_URL = "https://api.github.com"

# Vision fallback model — used by element_resolver.py for SoM disambiguation
VISION_MODEL = "anthropic/claude-haiku-4-5"

# Default AI model
MODEL = "qwen/qwen3-next-80b-a3b-instruct"

# Fallback model (used when primary model times out or is unavailable)
FALLBACK_MODEL = "deepseek/deepseek-chat"

# Model used by the normalization layer to reformat non-Qwen model responses
# into the expected action schema.  Should be cheap & fast — Qwen 80B is ideal.
NORMALIZER_MODEL = "qwen/qwen3-next-80b-a3b-instruct"

# Model used for the one-shot task-type classifier (~100 tokens, < $0.001).
CLASSIFIER_MODEL = "qwen/qwen3-next-80b-a3b-instruct"


# =============================================================================
# Model Roster — Specialist Routing (Phase 5.4)
# Qwen (Tier 0) acts as the router during GoalTracker decomposition.
# It assigns a tier to each subtask.  The orchestrator swaps self._model
# based on the tier before running _brain_loop() for that subtask.
#
# Keys: tier name (str).  Values: dict with model_id, fallback, description.
# "description" is injected verbatim into the decompose system prompt so the
# router LLM knows what each tier is good at.
# =============================================================================

MODEL_ROSTER = {
    "tier_0": {
        "model_id": "qwen/qwen3-next-80b-a3b-instruct",
        "fallback": "deepseek/deepseek-chat",
        "description": "General-purpose: UI automation, research, chat, planning, writing, simple tasks.",
    },
    "tier_1": {
        "model_id": "qwen/qwen3-coder-next",            # qwen3-coder-next nothink — fast, cheap, no hallucinated paths
        "fallback": "qwen/qwen3-next-80b-a3b-instruct",  # general-purpose 80B
        "description": (
            "Quick coding: small edits, single-file changes, config tweaks, simple scripts, "
            "scaffolding/boilerplate, writing simple tests, deployment scripts."
        ),
        "extra_body": {"reasoning": {"budget_tokens": 0}},  # nothink mode — skip chain-of-thought
    },
    "tier_2": {
        "model_id": "qwen/qwen3-coder",                # strong coder
        "fallback": "mistralai/mistral-small-2603",    # discover: 0% err, fast, $0.20/1M output
        "description": (
            "Standard coding: multi-file changes, feature implementation, moderate debugging, "
            "refactoring existing code, writing integration tests, fixing non-trivial bugs."
        ),
    },
    "tier_3": {
        "model_id": "qwen/qwen3-coder",              # same strong coder as tier_2, heavier fallback
        "fallback": "deepseek/deepseek-chat",
        "description": (
            "Complex coding: large refactors, architecture changes, cross-module rewrites, "
            "tricky debugging (race conditions, memory leaks), complex test harnesses."
        ),
    },
    "tier_4": {
        "model_id": "anthropic/claude-opus-4.6",
        "fallback": "anthropic/claude-sonnet-4.6",
        "description": "Nuclear: extremely difficult problems only. Use sparingly — very expensive.",
    },
}

# Tier names that use Qwen-family models (skip normalization layer)
QWEN_FAMILY_TIERS = {"tier_0", "tier_1", "tier_2", "tier_3"}  # all Qwen-family coding tiers

# Models that produce valid action JSON natively — skip the normalization layer.
# The normalization layer calls qwen3-next-80b to reformat responses into the
# expected action schema. Models already producing valid JSON don't need this
# and only incur extra latency + cost. Add only when confirmed reliable.
SKIP_NORMALIZATION_MODELS: frozenset[str] = frozenset({
    "qwen/qwen3-next-80b-a3b-instruct",  # IS the normalizer — never self-normalize
    "qwen/qwen3-coder-next",              # Qwen3-Coder-Next, produces valid JSON reliably
    "qwen/qwen3-coder",                  # 480B MoE, produces valid JSON reliably
    "deepseek/deepseek-chat",            # DeepSeek V3 — clean JSON output
    "deepseek/deepseek-v3.2",            # DeepSeek V3.2 — clean JSON output
    "mistralai/mistral-small-2603",      # Tier 2 fallback — clean JSON output
    "google/gemini-3-flash-preview",      # Gemini 3 Flash — 0% error, clean JSON (planning only)
})

# Per-model max_tokens cap for LLM calls.
# Tier_0/1/2 (Qwen, DeepSeek V3): 8192 is ample — they output compact JSON actions.
# Tier_3 (DeepSeek R1): generates verbose think blocks + long code; 16K base,
# retry doubles to 32K before falling back to partial content.
# Tier_4 (Claude): kept at 32K for the rare nuclear cases.
MODEL_MAX_TOKENS: dict[str, int] = {
    "qwen/qwen3-next-80b-a3b-instruct":        8192,
    "qwen/qwen3-coder-next":                    8192,  # Tier 1
    "qwen/qwen3-coder":                         8192,  # Tier 2
    "deepseek/deepseek-chat":                  8192,
    "deepseek/deepseek-v3.2":                  8192,
    "mistralai/mistral-small-2603":            8192,   # Tier 2 fallback
    "deepseek/deepseek-r1":                    32768,  # Full R1 — think blocks can be 15K+ tokens
    "deepseek/deepseek-r1-distill-llama-70b":  16384,
    "amazon/nova-micro-v1":                    8192,   # retired Tier 1
    "liquid/lfm-2-24b-a2b":                    8192,   # retired Tier 1 fallback
    "google/gemini-3-flash-preview":           8192,   # Tier 3
    "mistralai/ministral-8b-2512":             8192,
    "anthropic/claude-haiku-4-5":              8192,
    "anthropic/claude-sonnet-4.6":             32768,
    "anthropic/claude-opus-4.6":               32768,
}
DEFAULT_MAX_TOKENS = 8192  # fallback for any model not in MODEL_MAX_TOKENS


# =============================================================================
# Model Pricing (USD per 1M tokens — input/output)
# Used for model-aware per-task cost tracking.
# Update these if OpenRouter pricing changes.
# =============================================================================

MODEL_PRICING = {
    # model_id: (input_price_per_1M, output_price_per_1M)
    "minimax/minimax-m2.5":       (0.295,  1.20),   # Testing placeholder
    "qwen/qwen3.5-35b-a3b":                   (0.1625,  1.3),    # Qwen3.5-35B-A3B
    "qwen/qwen3-coder-next":                 (0.12,    0.75),   # Qwen3-Coder-Next — Tier 1
    "qwen/qwen3-coder":                      (0.22,    1.00),   # Qwen3-Coder — Tier 2 & 3
    "qwen/qwen3-next-80b-a3b-instruct":       (0.09,    1.10),   # Qwen3-Next 80B — Tier 0 router
    "deepseek/deepseek-chat":               (0.27,  1.10),   # DeepSeek V3 (legacy fallback)
    "deepseek/deepseek-v3.2":               (0.26,  0.38),   # retired Tier 2
    "mistralai/mistral-small-2603":          (0.15,  0.20),   # Tier 2 primary (discover)
    "deepseek/deepseek-r1":                        (0.55,  2.19),   # DeepSeek R1 (full)
    "deepseek/deepseek-r1-distill-llama-70b":      (0.70,  0.70),   # Tier 3 — distilled R1, no token explosion
    "amazon/nova-micro-v1":                   (0.035, 0.14),   # retired Tier 1 primary
    "liquid/lfm-2-24b-a2b":                   (0.030, 0.12),   # retired Tier 1 fallback
    "mistralai/ministral-8b-2512":            (0.15,  0.20),   # retired Tier 2 fallback
    "google/gemini-3-flash-preview":        (0.50,  3.00),   # Gemini 3 Flash — planning model
    "anthropic/claude-haiku-4-5":           (1.00,  5.00),   # Vision model (corrected pricing)
    "anthropic/claude-sonnet-4.6":          (3.00,  15.00),  # Tier 3 — complex coding
    "anthropic/claude-opus-4.6":            (5.00,  25.00),  # Tier 4 — nuclear
}


# =============================================================================
# Task-Type → Model Mapping
# GenieOrchestrator selects model based on task classification.
# =============================================================================

TASK_MODEL_MAP = {
    # task_type:         (primary_model,                            fallback_model)
    # These are task-level defaults for interactive/single-shot mode.
    # When GoalTracker is active, per-subtask model_tier overrides these.
    "ui_automation":    ("qwen/qwen3-next-80b-a3b-instruct",      "deepseek/deepseek-chat"),
    "code_generation":  ("qwen/qwen3-next-80b-a3b-instruct",      "deepseek/deepseek-chat"),
    "complex_reasoning":("deepseek/deepseek-r1",                   "qwen/qwen3-next-80b-a3b-instruct"),
    "research":         ("qwen/qwen3-next-80b-a3b-instruct",      "deepseek/deepseek-chat"),
    "content_production": ("qwen/qwen3-next-80b-a3b-instruct",    "deepseek/deepseek-chat"),
    "default":            ("qwen/qwen3-next-80b-a3b-instruct",    "deepseek/deepseek-chat"),
    # --- Phase 5.4 new task types ---
    "debugging":        ("deepseek/deepseek-v3.2",                      "qwen/qwen3-coder-next"),
    "testing":          ("qwen/qwen3-coder-next",                     "qwen/qwen3-next-80b-a3b-instruct"),
    "refactoring":      ("deepseek/deepseek-v3.2",                      "qwen/qwen3-coder-next"),
    "scaffolding":      ("qwen/qwen3-coder-next",                     "qwen/qwen3-next-80b-a3b-instruct"),
    # --- Phase 5.4 internal routing types ---
    # planning: used by plan_phase() and sequence_phase() in planner.py
    # Gemini Flash: fast (18s), cheap ($0.50/1M), 0% error. normalize=False in
    # planner.py prevents the normalization layer from mangling decompose output.
    # Fallback: mistral-small-2603 (fast, cheap, clean JSON output).
    "planning":           ("google/gemini-3-flash-preview",         "mistralai/mistral-small-2603"),
    # execution/summarisation: activated in Phase 5.3 (GoalTracker)
    "execution":          ("qwen/qwen3-next-80b-a3b-instruct",   "deepseek/deepseek-chat"),
    "summarisation":      ("qwen/qwen3-next-80b-a3b-instruct",   "deepseek/deepseek-chat"),
}


# =============================================================================
# Per-Task Budget Defaults (USD)
# GenieOrchestrator enforces these per task type.
# Configurable at task-approval time.
# =============================================================================

TASK_BUDGET_DEFAULTS = {
    "ui_automation":     0.50,
    "code_generation":   2.00,
    "complex_reasoning": 2.00,
    "research":          3.00,
    "content_production":5.00,
    "debugging":         2.00,
    "testing":           1.00,
    "refactoring":       3.00,
    "scaffolding":       0.50,
    "default":           2.00,
}

# Hard monthly API cap (USD). Enforced across all tasks.
MONTHLY_BUDGET_CAP = 200.00

# =============================================================================
# Task-Type Auto-Classifier (Phase 5.4)
# Single-shot prompt — the LLM picks one of the known task_types from the goal.
# =============================================================================

# Only user-facing task types (exclude internal routing types like planning,
# execution, summarisation — those are set programmatically).
CLASSIFIABLE_TASK_TYPES: list[str] = [
    "ui_automation",
    "code_generation",
    "complex_reasoning",
    "research",
    "content_production",
    "debugging",
    "testing",
    "refactoring",
    "scaffolding",
]

CLASSIFY_SYSTEM_PROMPT = (
    "You are a task classifier for a desktop automation agent.\n"
    "Given the user's goal, reply with EXACTLY ONE word — the task type "
    "that best describes it.\n\n"
    "Allowed task types:\n"
    + "\n".join(f"- {t}" for t in CLASSIFIABLE_TASK_TYPES)
    + "\n\n"
    "Definitions:\n"
    "- ui_automation: clicking, typing, navigating desktop/web UI elements\n"
    "- code_generation: writing new code, scripts, or config files from scratch\n"
    "- complex_reasoning: multi-step logic, math, analysis, architecture design\n"
    "- research: web searches, reading docs, gathering information\n"
    "- content_production: writing emails, documents, reports, presentations\n"
    "- debugging: finding and fixing bugs in existing code\n"
    "- testing: writing or running tests, test plans, QA\n"
    "- refactoring: restructuring existing code without changing behavior\n"
    "- scaffolding: ONLY project setup with no application logic — creating folder\n"
    "  structures, empty stubs, boilerplate config files. If the task involves\n"
    "  writing ANY application logic or multi-module systems, use code_generation.\n\n"
    "Reply with ONLY the task type, nothing else. No punctuation, no explanation."
)

# =============================================================================
# Complexity-Tier Classifier (single brain loop tasks)
# One-shot prompt — the LLM picks a MODEL_ROSTER tier based on goal complexity.
# =============================================================================

COMPLEXITY_CLASSIFIABLE_TIERS: list[str] = [
    "tier_0",
    "tier_1",
    "tier_2",
    "tier_3",
    "tier_4",
]

COMPLEXITY_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a complexity classifier for a desktop automation agent.\n"
    "Given the user's goal, reply with EXACTLY ONE tier name that best matches "
    "the complexity of the task.\n\n"
    "Allowed tiers:\n"
    + "\n".join(
        f"- {tier}: {MODEL_ROSTER[tier]['description']}"
        for tier in COMPLEXITY_CLASSIFIABLE_TIERS
    )
    + "\n\n"
    "Reply with ONLY the tier name, nothing else. No punctuation, no explanation."
)


# =============================================================================
# GTK Window Configuration
# =============================================================================

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 400

# Whether Genie asks clarifying questions before starting a task.
# Set to False in automated test runs to prevent blocking waits.
CLARIFY_ENABLED: bool = True

# Whether the ScriptPlanner generates bash script segments for deterministic
# subtasks before falling through to the ReAct brain loop.
SCRIPT_PLANNER_ENABLED: bool = True


# =============================================================================
# Window Registry (Layer 1)
# =============================================================================

# Seconds to poll xdotool search --pid <pid> before declaring launch failure.
# Per app tier — chosen to reflect realistic cold-launch times on this hardware.
LAUNCH_TIMEOUTS = {
    "atspi":     8,    # GTK apps (gedit, nautilus, etc.)
    "cdp":       20,   # Firefox, Chromium — cold launch on SSD
    "electron":  25,   # Slack, Discord, VS Code — worst case
    "terminal":  5,    # xterm, alacritty — sub-1s cold launch
    "default":   10,
}

# WID poll interval in seconds (250ms)
WID_POLL_INTERVAL = 0.25

# Timeout multipliers for kill-and-retry attempts (exponential backoff).
# Attempt 1 = 1.0x, attempt 2 = 1.5x, attempt 3 = 2.0x.
RETRY_TIMEOUT_MULTIPLIERS = [1.0, 1.5, 2.0]

# Maximum open_app attempts (original + retries)
MAX_OPEN_APP_RETRIES = 3

# Seconds to wait for a window to receive focus/input (UI convenience only)
FOCUS_WINDOW_TIMEOUT = 10


# =============================================================================
# App Profiles (Layer 2)
# Determines resolution tier and CDP port at launch time.
# Apps not listed here default to tier "atspi" with vision fallback.
# =============================================================================

APP_PROFILES = {
    # app_binary:  {"cdp_port": base_port, "tier": "cdp_primary", "launch_tier": timeout_key, "launch_flags": [...]}
    # launch_flags are injected verbatim into the Popen call after the binary name.
    # Flag syntax differs by app family — Chromium/Electron use --user-data-dir=.
    # --user-data-dir requires = syntax for Chrome/Electron.
    # Terminal apps carry an empty list.
    "chrome":   {"binary": "google-chrome",         "cdp_port": 9222, "tier": "cdp_primary",  "launch_tier": "cdp",      "wm_class": "Google-chrome", "launch_flags": ["--user-data-dir=/tmp/genie_chrome_profile", "--force-renderer-accessibility", "--disable-session-crashed-bubble", "--no-first-run", "--disable-blink-features=AutomationControlled"]},
    "code":     {"binary": "/usr/share/code/code",  "cdp_port": 9223, "tier": "cdp_primary",  "launch_tier": "electron", "wm_class": "Code",          "launch_flags": ["--user-data-dir=/tmp/genie_code_profile"]},
    "slack":    {"binary": "slack",                 "cdp_port": 9224, "tier": "cdp_primary",  "launch_tier": "electron", "wm_class": "Slack",         "launch_flags": ["--user-data-dir=/tmp/genie_slack_profile"]},
    "discord":  {"binary": "discord",               "cdp_port": 9225, "tier": "cdp_primary",  "launch_tier": "electron", "wm_class": "discord",       "launch_flags": ["--user-data-dir=/tmp/genie_discord_profile"]},
    "obsidian": {"binary": "obsidian",              "cdp_port": 9250, "tier": "cdp_primary",  "launch_tier": "electron", "wm_class": "obsidian",      "launch_flags": ["--user-data-dir=/tmp/genie_obsidian_profile"]},
    # Terminal apps
    "xterm":     {"binary": "xterm",     "cdp_port": None, "tier": "terminal", "launch_tier": "terminal", "launch_flags": []},
    "alacritty": {"binary": "alacritty", "cdp_port": None, "tier": "terminal", "launch_tier": "terminal", "launch_flags": []},
}

# CDP base port scan range — if base port is taken, increment up to this limit
CDP_PORT_SCAN_MAX = 9299

# Base ports for session port ledger — one monotonic counter per tier.
# Ledger keys are tier strings ("cdp", "electron"), not binary names.
# APP_PROFILES cdp_port values are reference hints only — not used at runtime.
CDP_BASE_PORT      = 9222   # firefox, chromium, chrome
ELECTRON_BASE_PORT = 9250   # code, slack, discord

# Maximum seconds to wait for CDP/Electron processes to die during lock cleanup.
# psutil.wait_procs(timeout=LOCK_CLEANUP_WAIT_TIMEOUT) — unpack (gone, alive).
# If alive is non-empty, log and proceed. D-state processes won't die faster than this.
LOCK_CLEANUP_WAIT_TIMEOUT = 5
PROCESS_WAIT_TIMEOUT = 5

CDP_RECV_TIMEOUT_SECONDS = 10  # Seconds to wait per ws.recv() call in _cdp_send before raising TransientError
CDP_EVENT_DISCARD_CAP = 50  # Max unsolicited CDP events discarded per _cdp_send recv-loop before raising UnrecoverableError


# =============================================================================
# AT-SPI Role Normalization (Layer 2)
# Canonical vocabulary emitted by LLM → toolkit-specific role string variants.
# The resolver matches against all aliases for a given canonical role.
# LLM system prompt defines only the canonical keys — never toolkit strings.
# =============================================================================

ROLE_ALIASES = {
    "button":    ["push button", "button"],
    "textfield": ["text", "entry", "edit", "editable text", "text area"],
    "checkbox":  ["check box", "checkbox"],
    "tab":       ["page tab"],
    "menuitem":  ["menu item"],
    "dropdown":  ["combo box", "combobox"],
    "link":      ["link"],
    "image":     ["image", "icon"],
    "list":      ["list", "list box"],
    "listitem":  ["list item"],
    "slider":    ["slider"],
    "spinbox":   ["spin button", "spin box"],
    "toolbar":   ["tool bar"],
    "statusbar": ["status bar"],
    "dialog":    ["dialog", "alert"],
    "frame":     ["frame", "panel", "filler"],
    "label":     ["label"],
}
# Note: Qt apps may expose role "unknown" with a numeric ID via pyatspi.
# The resolver must NOT skip "unknown" role nodes — fall through to name matching.


# =============================================================================
# CDP Role Normalization (Layer 2)
# Chrome DevTools Protocol equivalent of ROLE_ALIASES.
# Chrome's Accessibility.getFullAXTree returns AX role strings in camelCase —
# incompatible with AT-SPI strings above.
# CDP BFS tier uses this dict exclusively; AT-SPI BFS uses ROLE_ALIASES
# exclusively. The two are never mixed.
# =============================================================================

CDP_ROLE_ALIASES = {
    "button":    ["button"],
    "textfield": ["textField", "searchBox", "textBox", "textArea", "textbox",
                  "comboBox", "combobox"],  # Chrome address bar is combobox in CDP
    "checkbox":  ["checkBox"],
    "tab":       ["tab"],
    "menuitem":  ["menuItem"],
    "dropdown":  ["comboBox", "combobox", "listBox"],
    "link":      ["link"],
    "image":     ["image", "img"],
    "list":      ["list", "listBox"],
    "listitem":  ["listItem"],
    "slider":    ["slider"],
    "spinbox":   ["spinButton"],
    "toolbar":   ["toolBar"],
    "statusbar": ["status"],
    "dialog":    ["dialog", "alertDialog"],
    "frame":     ["generic", "none"],
    "label":     ["label"],
}


def _validate_role_alias_parity() -> None:
    """
    Structural invariant: ROLE_ALIASES and CDP_ROLE_ALIASES must share
    the exact same set of canonical keys. Raises ValueError on import
    if they ever diverge.
    """
    atspi_keys = set(ROLE_ALIASES)
    cdp_keys = set(CDP_ROLE_ALIASES)

    missing_from_cdp = atspi_keys - cdp_keys
    if missing_from_cdp:
        raise ValueError(f"CDP_ROLE_ALIASES missing canonical keys: {missing_from_cdp}")

    missing_from_atspi = cdp_keys - atspi_keys
    if missing_from_atspi:
        raise ValueError(f"ROLE_ALIASES missing canonical keys: {missing_from_atspi}")


_validate_role_alias_parity()


# =============================================================================
# File and Terminal Action Safety (Layer 2/3)
# =============================================================================

# Commands matching any pattern in this list are rejected before execution.
# Checked via substring match on the full cmd string after stripping whitespace.
CMD_BLOCKLIST = [
    ("substr", "rm -rf"),
    ("substr", "rm -r /"),
    ("substr", "dd if"),
    ("substr", "mkfs"),
    ("substr", "chmod -r /"),
    ("substr", "chown -r /"),
    ("substr", "> /dev/sd"),
    ("substr", ":(){ :|:& };:"),   # fork bomb
    ("substr", "curl | sh"),
    ("substr", "curl | bash"),
    ("substr", "wget | sh"),
    ("substr", "wget | bash"),
    ("regex",  r"\|\s*sh\b"),
    ("regex",  r"\|\s*bash\b"),
]

# Maximum bytes read by read_file action before truncation.
# Prevents context window overflow. ~150k bytes ≈ 37k tokens.
# Covers files up to ~2,500 lines. Safe for 8K-token model context with
# system prompt + action history occupying the remainder.
MAX_READ_FILE_BYTES = 150_000

# Truncation marker appended to read_file output when file exceeds limit.
READ_FILE_TRUNCATION_MARKER = "[TRUNCATED — file is {total_bytes} bytes, showing first {limit} bytes]"

# Timeout in seconds for run_command execution.
# Individual commands can override this via their "timeout" arg.
DEFAULT_CMD_TIMEOUT = 30

# Maximum characters read by read_element action before truncation.
# UI elements rarely need more than 2000 chars for LLM state assessment.
MAX_READ_ELEMENT_CHARS = 2000
READ_ELEMENT_TRUNCATION_MARKER = "[TRUNCATED — element text is {total_chars} chars, showing first {limit} chars]"

# Post-DoAction settle delay (ms) before reading AT-SPI element state.
# Same rationale as the CDP 100ms pre-observe delay — allows widget event
# handlers to complete before state is queried.
ATSPI_STATE_SETTLE_MS = 100

# Maximum nodes visited during pyatspi.findDescendant() STATE_FOCUSED search
# in observation.py. Bounds traversal cost on deep AT-SPI trees. Predicate uses a
# stateful counter and raises StopIteration when this limit is hit.
ATSPI_FOCUSED_SEARCH_MAX_NODES = 50


# =============================================================================
# Execution Logging (Layer 3)
# JSONL append-only logs. One JSON object per line, one entry per completed task.
# =============================================================================

# Maximum characters logged for large action args (write_file content,
# append_file content, type_text text, type_element text, run_command cmd).
# Appends truncation marker when cap is hit. Storage cap — not a context cap.
ARGS_TRUNCATION_CHARS = 500


# Maximum characters logged for run_command stdout and stderr in the
# observation block. Separate from MAX_READ_ELEMENT_CHARS — applies to
# shell output only, not UI element reads.
OBSERVATION_OUTPUT_TRUNCATION_CHARS = 2000

# Truncation marker appended to args fields when ARGS_TRUNCATION_CHARS is hit.
ARGS_TRUNCATION_MARKER = "[TRUNCATED — showing first {limit} chars]"

# Truncation marker appended to stdout/stderr when OBSERVATION_OUTPUT_TRUNCATION_CHARS is hit.
OBSERVATION_OUTPUT_TRUNCATION_MARKER = "[TRUNCATED — showing first {limit} chars]"

WORKSPACE_DIR = os.path.expanduser("~/genie_workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
SUCCESS_LOG_PATH    = os.path.join(LOG_DIR, "genie_success.jsonl")
INCOMPLETE_LOG_PATH = os.path.join(LOG_DIR, "genie_incomplete.jsonl")
CHECKPOINT_PATH     = os.path.join(LOG_DIR, "checkpoint.json")
COST_MONTHLY_PATH   = os.path.join(LOG_DIR, "cost_monthly.json")
TASK_LOG_PATH       = os.path.join(LOG_DIR, "task_log.jsonl")
TASK_QUEUE_DB_PATH  = os.path.join(LOG_DIR, "task_queue.db")

# --- Log Entry Schemas ---
#
# Per-action schema written to genie_incomplete.jsonl on every observe() call:
# {
#   "task_id": "uuid",        — UUID, shared across all actions in one task run
#   "sequence": 1,            — starts at 1 per task_id, increments per attempt
#                                (not per logical action)
#   "attempt": 1,             — starts at 1, increments on retry of the same
#                                logical action, resets to 1 on new action
#   "timestamp": "ISO8601",   — ISO 8601 with millisecond precision
#   "action": "click_element",— action name (e.g. "click_element",
#                                "run_command", "wait")
#   "args": {},               — action arguments as passed by Layer 4
#   "result": "success",      — one of: "success", "environmental_failure",
#                                "transient_failure", "unrecoverable"
#   "observation": {},        — action-specific passive capture fields;
#                                empty dict {} for wait and chat; never null
#   "error": null,            — null on success; on failure:
#                                {"type": "<ExceptionClassName>", "message": "..."}
#   "duration_ms": 340        — int or null; wall clock ms from Layer 4 t_start
#                                to observe() entry; null if t_start not passed
# }
#
# Success entries (result == "success") are additionally written to
# genie_success.jsonl — same schema, same entry.
# genie_incomplete.jsonl receives every entry.
# genie_success.jsonl receives success entries only.
#
# observation_partial: true is added to the observation dict when a passive
# sub-call fails but the action itself succeeded. Layer 4 checks this field
# and may elect to fire an explicit read_element as fallback.


# =============================================================================
# GoalTracker (Phase 5.3)
# =============================================================================

# Cap on decompose output to prevent runaway subtask lists.
# Prompt targets 2–30 depending on project size; hard cap at 60.
MAX_GOALTRACKER_SUBTASKS = 60

# How many of the most-recently-completed subtasks receive full interface
# blocks (AST-extracted signatures) in completed_context().  Subtasks outside
# this window receive file-list only.  Keeps injected context compact even at
# 30 subtasks while covering the symbols the next subtask is most likely to call.
SUBTASK_INTERFACE_RECENT_WINDOW = 5

# Per-subtask iteration cap — if a single subtask exceeds this many brain-loop
# iterations, it is marked failed and replan kicks in.  Prevents one stuck
# subtask from burning the entire task budget.  0 = disabled (rely on global cap).
MAX_ITERATIONS_PER_SUBTASK = 45

# Cap on clarifying questions generated by clarify() in planner.py.
# Complexity-adaptive: clarify() reads CLARIFY_QUESTION_TIERS to pick the
# right cap at runtime based on the LLM's own complexity estimate.
MAX_CLARIFY_QUESTIONS = 30
CLARIFY_QUESTION_TIERS = {
    "simple":  3,   # one-shot file creation, single command, quick lookup
    "medium":  8,   # multi-file generation, moderate feature work
    "large":  15,   # full system, broad spec
    "massive": 30,  # platform rewrite
}

# Per-task cap on mid-task ask_user calls issued by the brain loop.
# Keyed by task complexity (as returned by clarify()). Independent of
# the pre-task clarifying question cap.
ASK_USER_TIERS = {
    "simple":   2,   # tiny tasks — almost never needs to ask mid-task
    "medium":  15,   # moderate projects
    "large":   50,   # full systems / broad specs
    "massive": 50,   # platform rewrites
}

# Maximum fetch_url calls to the SAME domain within one task.
# Prevents runaway re-fetching of the same site (e.g. 126 fetches for 10 docs).
MAX_FETCH_PER_DOMAIN = 5

# Maximum lines in a single write_file for markdown docs.
# If the written file exceeds this, a warning is injected into the observation
# so the LLM can truncate/rewrite.  0 = disabled.
WRITE_FILE_MAX_LINES_MD = 200


# =============================================================================
# Post-Subtask Verifier (Phase 7)
# Automated checks between "subtask calls done" and "move to next subtask".
# Feature-flagged OFF by default — zero behaviour change unless opt-in.
# =============================================================================

# Master switch: set GENIE_VERIFY=0 env var to disable.
VERIFICATION_ENABLED: bool = os.environ.get("GENIE_VERIFY", "1") == "1"

# Which verification levels to run (when VERIFICATION_ENABLED is True).
# "import"     — AST-based import resolution (0 LLM calls, ~200ms)
# "types"      — pyright type checking (0 LLM calls, ~500ms)
# "smoke"      — python -c "from <pkg> import *" (0 LLM calls, ~2s, final subtask only)
# "llm_review" — LLM cross-file integration review (1 LLM call, ~3s, final subtask only)
VERIFY_LEVELS: list[str] = ["import", "types", "smoke", "llm_review"]

# Max re-entries into brain_loop to fix verification failures per subtask.
# After this many retries, mark_done proceeds anyway (fail-open).
MAX_VERIFY_FIX_ATTEMPTS: int = 2

# Hard timeout (seconds) per individual verification level.
VERIFY_TIMEOUT_SECONDS: int = 10


# =============================================================================
# TaskScratchpad — Shared Working Memory (Phase 6)
# Structured JSON object persisting across subtasks within a single task.
# Updated by ScratchpadWriter after every brain loop iteration.
# Injected into context by ScratchpadReader before every iteration.
# =============================================================================

# Compression target in tokens for the rendered scratchpad block.
# Not a hard cap — if protected entries exceed this, render in full.
# Silent data loss is worse than extra tokens in context.
SCRATCHPAD_TOKEN_BUDGET = 400

# Model used by ScratchpadWriter for fact extraction.
# ministral-8b-2512: 100% schema compliance on controlled tests, 1.30s avg.
SCRATCHPAD_WRITER_MODEL = "mistralai/ministral-8b-2512"

# Hard timeout (seconds) for each ScratchpadWriter LLM call.
# Abandoned if exceeded — scratchpad unchanged, brain loop continues,
# SCRATCHPAD_MISS_COUNTER incremented.
SCRATCHPAD_WRITER_TIMEOUT_S = 2

# Maximum characters of raw stdout passed to the writer.
# Pre-truncation — taken directly from _dispatch() return, not from history.
# Separate from OBSERVATION_OUTPUT_TRUNCATION_CHARS (history/JSONL limit).
SCRATCHPAD_WRITER_INPUT_CHARS = 4096

# Maximum characters of raw stderr passed to the writer.
# Smaller cap — stderr is usually shorter and denser than stdout.
# When action raised an exception, exception message is passed as stderr.
SCRATCHPAD_WRITER_STDERR_CHARS = 1024

# Recency window for eviction protection.
# Entries written or referenced within the last N subtasks are protected
# regardless of category. Outside this window, eviction priority applies:
# errors → subtask_outcomes (collapsed) → decisions (truncated) →
# files (oldest first) → facts (oldest first, last resort).
SCRATCHPAD_RECENCY_WINDOW = 5


# =============================================================================
# ReAct Brain Loop (Layer 4)
# =============================================================================

# Hard iteration cap per task (both interactive and autonomous modes)
MAX_ITERATIONS_PER_TASK = 500

# LLM call timeout constants (seconds). Read timeout covers DeepSeek R1 long reasoning.
LLM_CONNECT_TIMEOUT = 10
LLM_READ_TIMEOUT = 120

# Maximum LLM call retries on network failure before declaring UNRECOVERABLE.
# Independent of MAX_ACTION_RETRIES — LLM failures never enter classify_error().
MAX_LLM_RETRIES = 3

# HTTP status codes from OpenRouter that trigger immediate fallback model switch.
# Distinct from network/timeout errors which retry the primary model.
LLM_SERVICE_ERROR_CODES = [429, 503]

# Interactive mode: single-shot plan, no ReAct loop. User is watching.
# Autonomous mode: full ReAct loop. User approved goal, execution runs in background.
# GenieOrchestrator exposes both modes explicitly.

# Execution history: total entries kept in memory (deque maxlen).
# All entries are stored; only the recent window is sent in full detail.
REACT_HISTORY_WINDOW = 100

# Context batching: recent window — last N entries sent with full observation detail.
# Older entries are compressed to one-line summaries, cutting input tokens ~70-80%.
CONTEXT_RECENT_WINDOW = 5

# Maximum characters for the compressed history summary block.
# If the summary exceeds this, oldest lines are dropped.
CONTEXT_SUMMARY_MAX_CHARS = 3000

# Loop detection: sliding window over recent LLM-generated ACT decisions.
# If the same act hash appears LOOP_DETECT_THRESHOLD times within the last
# LOOP_DETECT_WINDOW decisions, the agent is declared stuck.
LOOP_DETECT_WINDOW = 8
LOOP_DETECT_THRESHOLD = 5

# Halt if this many consecutive schema-validation errors occur with no
# successfully-parsed action in between.  Prevents an infinite loop caused
# by the model repeatedly emitting <think> blocks with no <act> block.
SCHEMA_ERROR_HALT_THRESHOLD = 4

# If the agent performs this many consecutive read-only actions (read_file,
# list_dir) without any write/run/done action, inject a system nudge telling
# it to act.  Prevents the "read spiral" where models keep re-reading files
# without ever writing changes.  0 = disabled.
READ_STALL_THRESHOLD = 6

# Multi-action batching: when True, the LLM may return a JSON list of actions
# in a single <act> block. When False, only single-action dicts are accepted
# (pre-1.5 behavior). Toggle off to disable batching without code changes.
BATCH_ENABLED = True

# Maximum actions in a single batch. Safety valve — prevents runaway batches
# from consuming unbounded iterations in one LLM turn.
BATCH_MAX_ACTIONS = 20

# Orchestrator-level truncation for history entries in context assembly.
# Caps each observation field per history entry when serializing the deque.
# Separate from OBSERVATION_OUTPUT_TRUNCATION_CHARS (Layer 3 JSONL limit).
HISTORY_OBSERVATION_TRUNCATION_CHARS = 500


# Telegram bot token and chat ID for progress notifications
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# Telegram notification cadence — progress update every N iterations (autonomous mode only).
TELEGRAM_PROGRESS_INTERVAL = 10
# Telegram inbound long-poll timeout (seconds). Passed as timeout= to getUpdates.
TELEGRAM_POLL_TIMEOUT = 10




# =============================================================================
# Error Recovery Taxonomy (Layer 4)
# =============================================================================

# Error classes — returned by GenieAgent.classify_error()
ERROR_CLASS_TRANSIENT      = "TRANSIENT"       # retry same action, up to MAX_ACTION_RETRIES
ERROR_CLASS_ENVIRONMENTAL  = "ENVIRONMENTAL"   # inject as observation, let LLM replan
ERROR_CLASS_RESOURCE       = "RESOURCE"        # pause loop, surface to user
ERROR_CLASS_UNRECOVERABLE  = "UNRECOVERABLE"   # halt, log full trace, require human

# Maximum retries for TRANSIENT errors before escalating to ENVIRONMENTAL.
# TRANSIENT never escalates directly to UNRECOVERABLE — ENVIRONMENTAL is
# the next step. UNRECOVERABLE is only reached via loop detection.
MAX_ACTION_RETRIES = 3

# Backoff delay in seconds between TRANSIENT retries (index = attempt number)
RETRY_BACKOFF_SECONDS = [0.5, 1.5, 3.0]

# Files/directories to exclude when building a handoff package.
# Matched against the *relative* file path using simple substring/suffix logic.
HANDOFF_EXCLUDE_PATTERNS: list[str] = [
    ".git/", ".env", ".env.", "*.pem", "*.key", "*.p12", "*.pfx",
    "__pycache__/", ".pyc", "node_modules/", ".venv/", "venv/",
    "dist/", "build/", ".DS_Store",
]

# Idempotency flag per action name.
# TRANSIENT errors on idempotent actions → full retry cycle (MAX_ACTION_RETRIES).
# TRANSIENT errors on non-idempotent actions → immediately escalate to ENVIRONMENTAL.
# Non-idempotent actions must not be retried blindly — side effects may duplicate.
ACTION_IDEMPOTENT = {
    "click_element":  True,
    "type_element":   True,
    "read_element":   True,
    "look":           True,
    "focus_window":   True,
    "read_file":      True,
    "press_key":      True,
    "click":          True,
    "wait":           True,
    "type_text":      True,   # raw xdotool type — field state controlled by LLM
    "list_dir":       True,
    "chat":           True,
    "run_command":    True,
    "run_background": False,
    "open_app":       False,
    "write_file":     False,
    "append_file":    False,
    "delete_file":    False,
    "kill_process":   False,
    "checkpoint":     True,
    "ask_user":        True,
    "list_clipboard_history":  True,
    "get_clipboard_item":      True,
    "paste_clipboard_item":    False,
    "open_pr":                 False,   # non-idempotent: creates a real PR; TRANSIENT → ENVIRONMENTAL immediately
    "assemble_handoff":        False,
    # GitHub API actions
    "create_repo":             False,
    "delete_repo":             False,
    "list_repos":              True,
    "fork_repo":               False,
    "list_branches":           True,
    "create_branch":           False,
    "delete_branch":           False,
    "list_prs":                True,
    "merge_pr":                False,
    "close_pr":                False,
    "create_issue":            False,
    "close_issue":             False,
    "list_issues":             True,
    "create_release":          False,
    "add_collaborator":        False,
    "get_file_contents":       True,
    "put_file":                False,
    # GitHub API actions (extended)
    "get_authenticated_user":  True,
    "update_repo":             False,
    "set_repo_topics":         False,
    "search_repos":            True,
    "list_labels":             True,
    "create_label":            False,
    "protect_branch":          False,
    "list_webhooks":           True,
    "create_webhook":          False,
    "delete_webhook":          False,
    "list_workflows":          True,
    "trigger_workflow":        False,
    "list_workflow_runs":      True,
    "create_gist":             False,
    "list_gists":              True,
    "star_repo":               False,
    "unstar_repo":             False,
    "create_org_repo":         False,
    "list_org_members":        True,
    "list_teams":              True,
    "list_packages":           True,
    "delete_package_version":  False,
    "list_notifications":      True,
    "mark_notifications_read": False,
    "get_audit_log":           True,
    "search_codebase":          True,
    "ast_search":               True,
}

# Error classification rules — maps exception type names and message substrings
# to error classes. GenieAgent.classify_error() iterates this in order.
# First match wins. Fallback is UNRECOVERABLE.
ERROR_CLASSIFICATION_RULES = [
    # (match_type, match_value, error_class)
    # match_type: "exception_type" matches type(exc).__name__
    #             "message_substr" matches substring in str(exc).lower()

    # TRANSIENT — retry same action
    ("exception_type", "TimeoutExpired",        ERROR_CLASS_TRANSIENT),
    ("message_substr", "cdp connection refused", ERROR_CLASS_TRANSIENT),
    ("message_substr", "atspi tree not ready",   ERROR_CLASS_TRANSIENT),
    ("message_substr", "window not yet responsive", ERROR_CLASS_TRANSIENT),

    # ENVIRONMENTAL — replan
    ("exception_type", "PermissionError",        ERROR_CLASS_ENVIRONMENTAL),
    ("exception_type", "IsADirectoryError",       ERROR_CLASS_ENVIRONMENTAL),
    ("exception_type", "NotADirectoryError",      ERROR_CLASS_ENVIRONMENTAL),
    ("exception_type", "FileExistsError",         ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "permission denied",       ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "read-only file system",   ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "element not found",      ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "not a directory",        ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "file not found",         ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "no such file or directory", ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "window disappeared",     ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "wid not in client list", ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "github_token",             ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "cannot open pr",            ERROR_CLASS_ENVIRONMENTAL),
    ("exception_type", "RuntimeError",              ERROR_CLASS_ENVIRONMENTAL),

    # RESOURCE — pause, surface to user
    ("message_substr", "429",                    ERROR_CLASS_RESOURCE),
    ("message_substr", "402",                    ERROR_CLASS_RESOURCE),
    ("message_substr", "rate limit",             ERROR_CLASS_RESOURCE),
    ("message_substr", "quota",                  ERROR_CLASS_RESOURCE),
    ("message_substr", "disk full",              ERROR_CLASS_RESOURCE),
    ("message_substr", "no space left",          ERROR_CLASS_RESOURCE),

    # UNRECOVERABLE — halt
    # NOTE: blocklist hits are ENVIRONMENTAL, not UNRECOVERABLE — the model
    # gets clear feedback ("rm -rf blocked") and can choose a safer approach.
    # Killing the whole session for a blocklist hit is too harsh.
    ("message_substr", "command rejected by blocklist", ERROR_CLASS_ENVIRONMENTAL),
    ("message_substr", "need_clarification",     ERROR_CLASS_UNRECOVERABLE),
    ("exception_type", "LaunchFailureError",     ERROR_CLASS_UNRECOVERABLE),
]
# Note: non-zero exit_code from run_command is NOT an error — it is an
# observation (CLASS_ENVIRONMENTAL handled by ReAct loop naturally).
# classify_error() is never called for run_command non-zero exit codes.

# Required args and types per action — used by orchestrator schema validation.
# Validation: presence check + isinstance type check. First failure raises SchemaValidationError.
# wait: accepts (int, float) — LLM may emit either. click x/y: normalized to int at dispatch.
# done and abort are terminal actions validated identically to regular actions.
ACTION_SCHEMA = {
    "click_element":  {"app": str, "role": str, "name": str},
    "type_element":   {"app": str, "role": str, "name": str, "text": str},
    "read_element":   {"app": str, "role": str, "name": str},
    "look":           {"app": str},
    "open_app":       {"app": str},
    "focus_window":   {"app": str},
    "click":          {"x": (int, float), "y": (int, float)},
    "press_key":      {"key": str},
    "type_text":      {"text": str},
    "wait":           {"seconds": (int, float)},
    "read_file":      {"path": str},
    "write_file":     {"path": str, "content": str},
    "append_file":    {"path": str, "content": str},
    "delete_file":    {"path": str},
    "list_dir":       {"path": str},
    "index_codebase":  {"path": str},
    "run_command":    {"cmd": str},
    "run_background": {"cmd": str},
    "kill_process":   {"pid": int},
    "chat":           {"message": str},
    "ask_user":       {"question": str},
    "done":           {"summary": str, "message": str},
    "abort":          {"reason": str},
    "checkpoint":     {},
    "list_clipboard_history":  {},
    "get_clipboard_item":      {"index": int},
    "paste_clipboard_item":    {"index": int},
    "fetch_url":               {"url": str},
    "open_pr":                 {"repo": str, "title": str, "body": str, "head": str},
    "assemble_handoff":        {"repo_path": str, "task_summary": str},
    # GitHub API actions
    "create_repo":             {"name": str},
    "delete_repo":             {"repo": str},
    "list_repos":              {},
    "fork_repo":               {"repo": str},
    "list_branches":           {"repo": str},
    "create_branch":           {"repo": str, "branch": str},
    "delete_branch":           {"repo": str, "branch": str},
    "list_prs":                {"repo": str},
    "merge_pr":                {"repo": str, "pr_number": int},
    "close_pr":                {"repo": str, "pr_number": int},
    "create_issue":            {"repo": str, "title": str},
    "close_issue":             {"repo": str, "issue_number": int},
    "list_issues":             {"repo": str},
    "create_release":          {"repo": str, "tag": str, "name": str},
    "add_collaborator":        {"repo": str, "username": str},
    "get_file_contents":       {"repo": str, "path": str},
    "put_file":                {"repo": str, "path": str, "content": str, "message": str},
    # GitHub API actions (extended)
    "get_authenticated_user":  {},
    "update_repo":             {"repo": str},
    "set_repo_topics":         {"repo": str, "topics": list},
    "search_repos":            {"query": str},
    "list_labels":             {"repo": str},
    "create_label":            {"repo": str, "name": str, "color": str},
    "protect_branch":          {"repo": str, "branch": str},
    "list_webhooks":           {"repo": str},
    "create_webhook":          {"repo": str, "url": str},
    "delete_webhook":          {"repo": str, "hook_id": int},
    "list_workflows":          {"repo": str},
    "trigger_workflow":        {"repo": str, "workflow": str},
    "list_workflow_runs":      {"repo": str},
    "create_gist":             {"description": str, "files": dict},
    "list_gists":              {},
    "star_repo":               {"repo": str},
    "unstar_repo":             {"repo": str},
    "create_org_repo":         {"org": str, "name": str},
    "list_org_members":        {"org": str},
    "list_teams":              {"org": str},
    "list_packages":           {},
    "delete_package_version":  {"package_type": str, "package_name": str, "version_id": int},
    "list_notifications":      {},
    "mark_notifications_read": {},
    "get_audit_log":           {"org": str},
    "search_codebase":          {"path": str, "pattern": str},
    "ast_search":               {"path": str, "query_type": str, "name": str},
}


# =============================================================================
# Logging
# =============================================================================

def log(message: str) -> None:
    """
    Simple logging function with [GENIE] prefix and timestamp.

    Args:
        message: The message to log

    Valid status values: "success", "partial", "failure"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[GENIE] [{timestamp}] {message}")


# =============================================================================
# Validation
# =============================================================================

def validate_config() -> bool:
    """
    Validate that required configuration values are present.

    Returns:
        True if configuration is valid, False otherwise
    """
    if not OPENROUTER_API_KEY:
        log("WARNING: OPENROUTER_API_KEY not set. Please set it in your .env file or environment.")
        return False
    return True


# Run validation on module import
if not validate_config():
    log("Configuration incomplete. Some features may not work correctly.")