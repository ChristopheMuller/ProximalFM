"""Resolve a training launch script into the config the trainer would see.

The paper's training table must be the run, not a transcription of it, so this
module reads ``scripts/pretrain_prod.sh`` (or any script shaped like it) and
returns the ``argparse.Namespace`` that ``python -m proximalfm.train`` would
have been given -- flags from the script layered on top of
``_train_config.build_parser()``'s defaults, which is what makes a knob the
script does not pass (``--weight_decay``, ``--scheduler``, the architecture
block) still reportable.

Why parse rather than execute: running the script launches a 75k-step run, and
sourcing it would still need a GPU-shaped environment. The subset of shell the
production scripts actually use is small and fixed -- ``VAR=${VAR:-default}``
assignments, line continuations, one ``python -m proximalfm.train`` invocation --
so it is parsed directly. Anything outside that subset (command substitution in
an assignment feeding the invocation, a conditionally built flag list) raises
rather than being guessed at: a silently mis-resolved flag would print a wrong
number into the paper.

Environment variables are **not** read from the ambient environment, so the
output does not depend on the shell the command is run from; pass ``--set
VAR=value`` to emulate an override.
"""

from __future__ import annotations

import re
import shlex
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping

from proximalfm.train._train_config import build_parser

# `VAR=value` / `export VAR=value`, the only assignment form the scripts use.
_ASSIGN_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
# `${VAR:-default}`, `${VAR}` or `$VAR`.
_VAR_RE = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)"
)
_ENTRY_RE = re.compile(r"^(?:uv\s+run\s+)?python[0-9.]*\s+-m\s+proximalfm\.train\b")


class ScriptParseError(RuntimeError):
    """The script is outside the shell subset this module resolves."""


@dataclass
class TrainingRun:
    """A training script resolved into flags, env and a parsed config."""

    script: Path
    config: Namespace
    """Full trainer config: script flags over ``build_parser()`` defaults."""

    argv: List[str]
    """The flag tokens as passed, after variable expansion."""

    env: Dict[str, str] = field(default_factory=dict)
    """Shell variables in force at the invocation, after overrides."""

    @property
    def explicit_flags(self) -> Dict[str, str]:
        """Flag -> value for the flags the script sets, in script order.

        Everything else in :attr:`config` is a parser default. Useful when a
        table wants to mark which values were chosen for this run.
        """
        flags: Dict[str, str] = {}
        key = None
        for tok in self.argv:
            if tok.startswith("--"):
                key = tok[2:]
                flags[key] = ""
            elif key is not None:
                flags[key] = f"{flags[key]} {tok}".strip()
        return flags


def _join_continuations(text: str) -> List[str]:
    """Collapse backslash-newline continuations into single logical lines."""
    return text.replace("\\\n", " ").splitlines()


def _expand(value: str, env: Mapping[str, str]) -> str:
    """Expand ``${VAR:-default}`` / ``$VAR`` against ``env``.

    An unset variable with no default expands to the empty string, as in bash.
    """

    def sub(match: re.Match) -> str:
        name = match.group(1) or match.group(3)
        default = match.group(2)
        if name in env:
            return env[name]
        return default if default is not None else ""

    return _VAR_RE.sub(sub, value)


def _strip_quotes(value: str) -> str:
    parts = shlex.split(value, comments=True)
    if len(parts) > 1:
        raise ScriptParseError(f"assignment value is not a single word: {value!r}")
    return parts[0] if parts else ""


def parse_script(
    path: str | Path, overrides: Mapping[str, str] | None = None
) -> TrainingRun:
    """Resolve a training launch script into a :class:`TrainingRun`.

    ``overrides`` stands in for the environment a ``VAR=...`` prefix would set:
    each entry is seeded into the variable map before the script's own
    ``${VAR:-default}`` assignments, so the default is skipped exactly as bash
    would skip it.
    """
    path = Path(path)
    lines = _join_continuations(path.read_text())

    env: Dict[str, str] = dict(overrides or {})
    command: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if _ENTRY_RE.match(stripped):
            if command is not None:
                raise ScriptParseError(
                    f"{path} invokes proximalfm.train more than once; this module "
                    "resolves single-invocation scripts only"
                )
            command = stripped
            continue

        match = _ASSIGN_RE.match(stripped)
        if match and command is None:
            name, raw = match.group(1), match.group(2)
            if "$(" in raw or "`" in raw:
                # A command substitution would need a shell to evaluate; only
                # fail if its value could reach the invocation.
                env.pop(name, None)
                continue
            env[name] = _strip_quotes(_expand(raw, env))

    if command is None:
        raise ScriptParseError(f"{path} contains no `python -m proximalfm.train` call")

    tokens = shlex.split(command, comments=True)
    argv = [_expand(tok, env) for tok in tokens[tokens.index("proximalfm.train") + 1 :]]

    unresolved = [tok for tok in argv if "$" in tok]
    if unresolved:
        raise ScriptParseError(f"unresolved shell variables in {path}: {unresolved}")

    # Unknown flags raise here, which is the point: the parser is the schema, so
    # a script that has drifted from `_train_config.py` fails loudly rather than
    # printing a stale table.
    config = build_parser().parse_args(argv)

    return TrainingRun(script=path, config=config, argv=argv, env=env)
