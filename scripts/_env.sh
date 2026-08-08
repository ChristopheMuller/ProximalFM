#!/usr/bin/env bash
# Shared prelude for scripts/. Source it, don't run it:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
#
# Three jobs, all of them things that differ between the laptop and the cluster:
#
#   1. cd to the repo root, so every relative path in these scripts (data/,
#      synt_eval/, transfer_rules.yaml) resolves no matter where you invoked
#      them from -- including from a scheduler's working directory.
#   2. Put the project venv on PATH. `uv run` is not available on the cluster,
#      so the scripts call plain `python` and this is what makes that the right
#      python. Override with VENV=/path/to/venv, or activate one yourself
#      beforehand (an already-active VIRTUAL_ENV is left alone).
#   3. Export WANDB_API_KEY from the repo-local key file. Optional -- with no key
#      file, wandb falls back to whatever is already exported, then to ~/.netrc
#      (what `wandb login` writes). See below.
#
# Where the wandb key comes from, in lookup order
# -----------------------------------------------
#   <repo>/.wandb_api_key          THE PROJECT DEFAULT, and it wins over
#                                  everything -- including an already-exported
#                                  $WANDB_API_KEY, which it overwrites. That is
#                                  deliberate: the checkout you are looking at
#                                  decides which account the run logs to, so a
#                                  cluster module or a stale ~/.bashrc export
#                                  cannot silently redirect it. Repo-scoped, so
#                                  no profile-wide state and no `wandb login`.
#                                  Gitignored, and refused outright if it ever
#                                  becomes tracked.
#   $WANDB_API_KEY_FILE            fallback, not an override -- the repo file
#                                  wins when both exist. For a scheduler or a
#                                  site-managed secrets path on a checkout that
#                                  has no key of its own.
#   ~/.config/proximalfm/wandb.key fallback, outside the repo, survives a re-clone.
#   (no file found)                an exported $WANDB_API_KEY is left as-is;
#                                  failing that, wandb falls back to ~/.netrc,
#                                  i.e. whatever `wandb login` wrote. Not an error.
#
# Create it so the key never lands in shell history and the file is never
# briefly world-readable:
#
#     (umask 077; cat > .wandb_api_key)     # paste the key, then Ctrl-D
#
# The value is never printed by this script. The cost of repo-scoping is that
# the key sits inside the tree you rsync/scp/tar around -- so if the checkout has
# ever been copied to a shared filesystem, rotate at https://wandb.ai/settings
# rather than hoping.
#
# No egress on the compute nodes? Set WANDB_MODE=offline (or --wandb_mode
# offline) and run `wandb sync wandb/offline-run-*` from a login node later.

# Fail loudly on a mistake, but only after the caller's own `set -euo pipefail`
# has been honoured -- callers set it themselves; this file does not.

_env_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${_env_repo_root}"

# --- python -------------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  VENV=${VENV:-"${_env_repo_root}/.venv"}
  if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "No venv at ${VENV}. Create it (uv sync) or set VENV=/path/to/venv." >&2
    exit 1
  fi
  # activate scripts reference unset vars under older Pythons; `set -u` would
  # abort on them, and the failure reads as a venv problem rather than a shell one.
  set +u
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  set -u
fi

# --- wandb credentials --------------------------------------------------------
for _env_key_file in \
  "${_env_repo_root}/.wandb_api_key" \
  "${WANDB_API_KEY_FILE:-}" \
  "${HOME}/.config/proximalfm/wandb.key"; do
  [[ -n "${_env_key_file}" && -f "${_env_key_file}" ]] || continue
  # .gitignore does not protect a file that is already tracked, and a key in
  # the index is one `git push` from being public. Refuse rather than warn.
  if git ls-files --error-unmatch "${_env_key_file}" >/dev/null 2>&1; then
    echo "${_env_key_file} is tracked by git. Remove it from the index" >&2
    echo "(git rm --cached) and rotate the key at wandb.ai/settings." >&2
    exit 1
  fi
  # Strip whitespace: a trailing newline is harmless, but a stray space or a
  # CRLF from editing the file on another machine produces a 401 that reads
  # like a revoked key.
  WANDB_API_KEY="$(tr -d '[:space:]' <"${_env_key_file}")"
  export WANDB_API_KEY
  if [[ "$(stat -c '%a' "${_env_key_file}" 2>/dev/null \
          || stat -f '%Lp' "${_env_key_file}" 2>/dev/null)" != "600" ]]; then
    echo "warning: ${_env_key_file} is not chmod 600" >&2
  fi
  break
done


unset _env_key_file
