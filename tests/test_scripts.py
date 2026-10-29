"""Checks on the command-line scripts that no other test would catch.

A flag can be declared and never read. Nothing fails, nothing warns, and the
script accepts the flag and ignores it -- which is worse than rejecting it,
because the run looks like it did what was asked. That happened here: a
`--compare-judges` flag was declared in one edit and the code reading it was
lost in another, and the only symptom was an evaluation that quietly graded
with one judge instead of two.

These read the scripts as syntax trees rather than importing them, so a script
whose import has side effects, or whose dependencies are missing, is still
checked.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ragsys.config import PROJECT_ROOT

SCRIPTS = sorted((PROJECT_ROOT / "scripts").glob("*.py"))


def tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def declared_flags(module: ast.Module) -> set[str]:
    """The attribute name argparse will give each declared option."""
    names = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        dest = next(
            (kw.value.value for kw in node.keywords if kw.arg == "dest"),
            None,
        )
        if isinstance(dest, str):
            names.add(dest)
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                names.add(argument.value.lstrip("-").replace("-", "_"))
                break
    return names


def read_attributes(module: ast.Module) -> set[str]:
    """Every `<something>.name` read anywhere in the script.

    Deliberately not restricted to a variable called `args`: the parsed
    namespace is sometimes passed on or unpacked under another name, and a
    false pass here would defeat the purpose of the check.
    """
    return {
        node.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }


def test_there_are_scripts_to_check() -> None:
    """Guards against the glob silently matching nothing."""
    assert SCRIPTS, "no scripts found; the checks below would pass vacuously"


def test_the_check_catches_an_unread_flag() -> None:
    """A negative control: a check that cannot fail is not a check.

    Without this, the parametrised test below would also pass if
    `declared_flags` quietly returned nothing.
    """
    module = ast.parse(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--used', action='store_true')\n"
        "parser.add_argument('--forgotten', action='store_true')\n"
        "args = parser.parse_args()\n"
        "print(args.used)\n"
    )
    assert declared_flags(module) == {"used", "forgotten"}
    assert declared_flags(module) - read_attributes(module) == {"forgotten"}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_every_declared_flag_is_read(path: Path) -> None:
    module = tree(path)
    unread = declared_flags(module) - read_attributes(module)
    assert not unread, f"{path.name} declares {sorted(unread)} but never reads them"
