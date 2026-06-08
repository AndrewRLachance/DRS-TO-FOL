from __future__ import annotations

from .scoring import is_better_lift

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
    Var,
)


def make_and(parts: tuple[Formula, ...] | list[Formula]) -> Formula:
    parts = tuple(parts)
    if not parts:
        raise ValueError("Cannot build empty conjunction")
    if len(parts) == 1:
        return parts[0]
    return Conjunction(parts)


def free_vars(formula: Formula) -> frozenset[str]:
    if isinstance(formula, Atom):
        return frozenset(arg.name for arg in formula.args if isinstance(arg, Var))
    if isinstance(formula, Negation):
        return free_vars(formula.body)
    if isinstance(formula, (Conjunction, Disjunction)):
        out: frozenset[str] = frozenset()
        for part in formula.parts:
            out |= free_vars(part)
        return out
    if isinstance(formula, (Implication, BiImplication)):
        return free_vars(formula.left) | free_vars(formula.right)
    if isinstance(formula, (Exists, Forall)):
        bound = frozenset(v.name for v in formula.vars)
        return free_vars(formula.body) - bound
    return frozenset()


def rewrite_children(formula: Formula) -> Formula:
    if isinstance(formula, Negation):
        return Negation(lift_quantifiers(formula.body))
    if isinstance(formula, Conjunction):
        return Conjunction(tuple(lift_quantifiers(part) for part in formula.parts))
    if isinstance(formula, Disjunction):
        return Disjunction(tuple(lift_quantifiers(part) for part in formula.parts))
    if isinstance(formula, Implication):
        return Implication(lift_quantifiers(formula.left), lift_quantifiers(formula.right))
    if isinstance(formula, BiImplication):
        return BiImplication(lift_quantifiers(formula.left), lift_quantifiers(formula.right))
    if isinstance(formula, Exists):
        return Exists(formula.vars, lift_quantifiers(formula.body))
    if isinstance(formula, Forall):
        return Forall(formula.vars, lift_quantifiers(formula.body))
    return formula


def lift_quantifiers_once(formula: Formula) -> Formula:
    """One-step classical quantifier abstraction.

    Rules:
      ¬¬P                         -> P
      ¬∃x.(A1 ∧ ... ∧ ¬B)          -> ∀x.((A1 ∧ ...) → B)
      ¬∃x.P                       -> ∀x.¬P
      ¬∀x.P                       -> ∃x.¬P
    """
    if isinstance(formula, Negation) and isinstance(formula.body, Negation):
        candidate = formula.body.body
        return candidate if is_better_lift(formula, candidate) else formula

    if isinstance(formula, Negation) and isinstance(formula.body, Exists):
        exists = formula.body

        if isinstance(exists.body, Conjunction):
            parts = list(exists.body.parts)
            negated_parts = [part for part in parts if isinstance(part, Negation)]

            # Conservative: only lift bounded universal when there is exactly one
            # negated consequent candidate.
            if len(negated_parts) == 1:
                neg = negated_parts[0]
                premises = [part for part in parts if part is not neg]
                if premises:
                    candidate = Forall(exists.vars, Implication(make_and(premises), neg.body))
                    return candidate if is_better_lift(formula, candidate) else formula

        candidate = Forall(exists.vars, Negation(exists.body))
        return candidate if is_better_lift(formula, candidate) else formula

    if isinstance(formula, Negation) and isinstance(formula.body, Forall):
        forall = formula.body
        candidate = Exists(forall.vars, Negation(forall.body))
        return candidate if is_better_lift(formula, candidate) else formula

    return formula


def lift_quantifiers(formula: Formula, max_steps: int = 100) -> Formula:
    current = formula

    for _ in range(max_steps):
        # Try the current node before rewriting children. This preserves
        # higher-value outer abstractions such as:
        #   ¬∃x.(A(x) ∧ ¬∃e.B(e,x)) -> ∀x.(A(x) → ∃e.B(e,x))
        # and prevents child rewrites from obscuring the parent pattern.
        top = lift_quantifiers_once(current)
        if top != current:
            current = top
            continue

        child_rewritten = rewrite_children(current)
        if child_rewritten == current:
            return current

        current = child_rewritten

    raise RuntimeError("Quantifier lifting did not converge")
