from __future__ import annotations

from ..fol_ast import (
    Atom,
    BiImplication,
    Conjunction,
    Disjunction,
    Exists,
    Forall,
    Formula,
    Implication,
    Negation,
)


def to_string(formula: Formula) -> str:
    if isinstance(formula, Atom):
        return f"{formula.pred}({','.join(a.name for a in formula.args)})"
    if isinstance(formula, Negation):
        return f"¬{paren(to_string(formula.body))}"
    if isinstance(formula, Conjunction):
        return "(" + " ∧ ".join(to_string(p) for p in formula.parts) + ")"
    if isinstance(formula, Disjunction):
        return "(" + " ∨ ".join(to_string(p) for p in formula.parts) + ")"
    if isinstance(formula, Implication):
        return f"({to_string(formula.left)} → {to_string(formula.right)})"
    if isinstance(formula, BiImplication):
        return f"({to_string(formula.left)} ↔ {to_string(formula.right)})"
    if isinstance(formula, Exists):
        return f"∃{' '.join(v.name for v in formula.vars)}.{paren(to_string(formula.body))}"
    if isinstance(formula, Forall):
        return f"∀{' '.join(v.name for v in formula.vars)}.{paren(to_string(formula.body))}"
    return repr(formula)


def paren(s: str) -> str:
    if s.startswith("(") and s.endswith(")"):
        return s
    return f"({s})"
