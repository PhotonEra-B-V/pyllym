#!/usr/bin/env bash
#
# PreToolUse hook: hard-block dangerous shell commands.
#
# Claude Code invokes this before every Bash tool call and pipes the tool
# input as JSON on stdin, e.g. {"tool_name":"Bash","tool_input":{"command":"..."}}.
#
# Exit 0            -> allow the command.
# Exit 2            -> block the command; stderr is shown to Claude as the reason.
# JSON on stdout    -> structured decision (we use exit codes for simplicity).
#
# This is defense-in-depth, NOT a sandbox. A determined pattern can evade a
# static denylist. The real boundary is human review + running untrusted code
# only where secrets/network are unreachable (see CLAUDE.md security model).

set -euo pipefail

input="$(cat)"

# Pull the command string out of the JSON without needing jq.
command="$(printf '%s' "$input" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print("")
    sys.exit(0)
print(data.get("tool_input", {}).get("command", ""))
' 2>/dev/null || printf '')"

# Nothing to inspect -> allow.
[ -z "$command" ] && exit 0

block() {
    echo "BLOCKED by block-dangerous-commands hook: $1" >&2
    echo "Command: $command" >&2
    exit 2
}

# --- Dangerous patterns (POSIX ERE, case-insensitive) -------------------------
# Each entry: "regex::human reason". Bash `[[ =~ ]]` uses POSIX ERE, which has
# NO \b word boundary and NO \d — use (^|[^alnum]) style guards and explicit
# whitespace via [[:space:]]. Keep patterns conservative to avoid false
# positives on legitimate project commands.
patterns=(
    'rm[[:space:]]+(-[[:alnum:]]*[rf][[:alnum:]]*[[:space:]]+)+(/|~|\$HOME|\*|\.($|[[:space:]]))::recursive/forced rm of root, home, or wildcard'
    ':[[:space:]]*\([[:space:]]*\)[[:space:]]*\{[[:space:]]*:[[:space:]]*\|[[:space:]]*:[[:space:]]*&::fork bomb'
    '(^|[^[:alnum:]])mkfs(\.[[:alnum:]]+)?([^[:alnum:]]|$)::filesystem format (mkfs)'
    '(^|[^[:alnum:]])dd[[:space:]][^|]*of=/dev/::dd writing to a raw device'
    '>[[:space:]]*/dev/(sd|nvme|disk|hd)::redirect into a raw disk device'
    '(^|[^[:alnum:]])shred([^[:alnum:]]|$)::secure-erase (shred)'
    '(^|[^[:alnum:]])(shutdown|reboot|halt|poweroff)([^[:alnum:]]|$)::power state change'
    'chmod[[:space:]]+-R[[:space:]]+0*777[[:space:]]+/::recursive world-writable on root'
    'chown[[:space:]]+-R[[:space:]].*[[:space:]]+/[[:space:]]*$::recursive chown of root'
    'git[[:space:]]+push[[:space:]].*--force::git push --force (use --force-with-lease)'
    'git[[:space:]]+push[[:space:]].*(^|[[:space:]])-f($|[[:space:]])::git push -f (use --force-with-lease)'
    'git[[:space:]]+reset[[:space:]]+--hard.*origin/::hard reset onto a remote ref (discards local work)'
    'git[[:space:]]+clean[[:space:]]+-[[:alpha:]]*f[[:alpha:]]*d::git clean -fd (deletes untracked files)'
    'curl[[:space:]].*\|[[:space:]]*(sudo[[:space:]]+)?(bash|sh|zsh)([^[:alnum:]]|$)::piping remote script straight into a shell'
    'wget[[:space:]].*\|[[:space:]]*(sudo[[:space:]]+)?(bash|sh|zsh)([^[:alnum:]]|$)::piping remote script straight into a shell'
    'eval[[:space:]].*\$\(curl::eval of remote content'
    'sudo[[:space:]]+rm([^[:alnum:]]|$)::sudo rm'
    '(pip|pip3|python[0-9.]*[[:space:]]+-m[[:space:]]+pip)[[:space:]]+.*--break-system-packages::pip --break-system-packages'
    '/dev/(tcp|udp)/::raw network socket via /dev/tcp'
    'history[[:space:]]+-c([^[:alnum:]]|$)::clearing shell history'
)

shopt -s nocasematch
for entry in "${patterns[@]}"; do
    regex="${entry%%::*}"
    reason="${entry##*::}"
    if [[ "$command" =~ $regex ]]; then
        block "$reason"
    fi
done

exit 0
