from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering
from typing import Iterable

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


@dataclass(frozen=True)
class FormulaScore:
    """Structural and abstraction-oriented score for a formula.

    Higher `total` means the formula is considered more abstract/readable.
    This is a heuristic, not a semantic proof. Use it to choose among logically
    equivalent lift candidates and to prevent rewrite cycles.
    """

    total: int
    abstraction: int
    penalty: int
    node_count: int
    atom_count: int
    negation_count: int
    conjunction_count: int
    disjunction_count: int
    implication_count: int
    biimplication_count: int
    exists_count: int
    forall_count: int
    max_depth: int
    quantifier_depth: int

    @property
    def quantifier_count(self) -> int:
        return self.exists_count + self.forall_count

    @property
    def connective_count(self) -> int:
        return (
            self.negation_count
            + self.conjunction_count
            + self.disjunction_count
            + self.implication_count
            + self.biimplication_count
        )

    def decision_key(self) -> tuple[int, int, int, int, int, int, int]:
        """Ordering key used to decide whether one formula is preferable.

        Preferences, in order:
          1. higher total score
          2. fewer nodes
          3. fewer negations
          4. fewer disjunctions
          5. more implications/biconditionals
          6. more universal quantifiers
          7. shallower tree
        """
        return (
            self.total,
            -self.node_count,
            -self.negation_count,
            -self.disjunction_count,
            self.implication_count + (2 * self.biimplication_count),
            self.forall_count,
            -self.max_depth,
        )


def score_formula(formula: Formula) -> FormulaScore:
    """Return the heuristic abstraction score for a formula."""
    return _score(formula, depth=1, quantifier_depth=0)


def is_better_lift(before: Formula, after: Formula, *, allow_equal: bool = False) -> bool:
    """Return True if `after` should replace `before`.

    `allow_equal=True` permits structurally different rewrites with the same
    decision key. The default is strict improvement.
    """
    before_key = score_formula(before).decision_key()
    after_key = score_formula(after).decision_key()

    if allow_equal:
        return after_key >= before_key and after != before
    return after_key > before_key


def best_formula(candidates: Iterable[Formula]) -> Formula:
    """Return the highest-scoring formula from a non-empty iterable."""
    iterator = iter(candidates)
    try:
        best = next(iterator)
    except StopIteration as exc:
        raise ValueError("best_formula() requires at least one candidate") from exc

    best_key = score_formula(best).decision_key()
    for candidate in iterator:
        key = score_formula(candidate).decision_key()
        if key > best_key:
            best = candidate
            best_key = key
    return best


def score_delta(before: Formula, after: Formula) -> int:
    """Return `score(after) - score(before)`."""
    return score_formula(after).total - score_formula(before).total


def _combine(
    *,
    abstraction_bonus: int,
    local_penalty: int,
    node_count: int,
    atom_count: int,
    negation_count: int,
    conjunction_count: int,
    disjunction_count: int,
    implication_count: int,
    biimplication_count: int,
    exists_count: int,
    forall_count: int,
    max_depth: int,
    quantifier_depth: int,
    children: tuple[FormulaScore, ...] = (),
) -> FormulaScore:
    abstraction = abstraction_bonus + sum(c.abstraction for c in children)
    penalty = local_penalty + sum(c.penalty for c in children)

    return FormulaScore(
        total=abstraction - penalty,
        abstraction=abstraction,
        penalty=penalty,
        node_count=node_count + sum(c.node_count for c in children),
        atom_count=atom_count + sum(c.atom_count for c in children),
        negation_count=negation_count + sum(c.negation_count for c in children),
        conjunction_count=conjunction_count + sum(c.conjunction_count for c in children),
        disjunction_count=disjunction_count + sum(c.disjunction_count for c in children),
        implication_count=implication_count + sum(c.implication_count for c in children),
        biimplication_count=biimplication_count + sum(c.biimplication_count for c in children),
        exists_count=exists_count + sum(c.exists_count for c in children),
        forall_count=forall_count + sum(c.forall_count for c in children),
        max_depth=max((max_depth, *(c.max_depth for c in children))),
        quantifier_depth=max((quantifier_depth, *(c.quantifier_depth for c in children))),
    )


def _score(formula: Formula, *, depth: int, quantifier_depth: int) -> FormulaScore:
    if isinstance(formula, Atom):
        return _combine(
            abstraction_bonus=0,
            local_penalty=0,
            node_count=1,
            atom_count=1,
            negation_count=0,
            conjunction_count=0,
            disjunction_count=0,
            implication_count=0,
            biimplication_count=0,
            exists_count=0,
            forall_count=0,
            max_depth=depth,
            quantifier_depth=quantifier_depth,
        )

    if isinstance(formula, Negation):
        child = _score(formula.body, depth=depth + 1, quantifier_depth=quantifier_depth)
        return _combine(
            abstraction_bonus=0,
            local_penalty=7,
            node_count=1,
            atom_count=0,
            negation_count=1,
            conjunction_count=0,
            disjunction_count=0,
            implication_count=0,
            biimplication_count=0,
            exists_count=0,
            forall_count=0,
            max_depth=depth,
            quantifier_depth=quantifier_depth,
            children=(child,),
        )

    if isinstance(formula, Conjunction):
        children = tuple(_score(p, depth=depth + 1, quantifier_depth=quantifier_depth) for p in formula.parts)
        return _combine(
            abstraction_bonus=0,
            local_penalty=max(0, len(formula.parts) - 1),
            node_count=1,
            atom_count=0,
            negation_count=0,
            conjunction_count=1,
            disjunction_count=0,
            implication_count=0,
            biimplication_count=0,
            exists_count=0,
            forall_count=0,
            max_depth=depth,
            quantifier_depth=quantifier_depth,
            children=children,
        )

    if isinstance(formula, Disjunction):
        children = tuple(_score(p, depth=depth + 1, quantifier_depth=quantifier_depth) for p in formula.parts)
        return _combine(
            abstraction_bonus=0,
            local_penalty=3 + max(0, len(formula.parts) - 1),
            node_count=1,
            atom_count=0,
            negation_count=0,
            conjunction_count=0,
            disjunction_count=1,
            implication_count=0,
            biimplication_count=0,
            exists_count=0,
            forall_count=0,
            max_depth=depth,
            quantifier_depth=quantifier_depth,
            children=children,
        )

    if isinstance(formula, Implication):
        children = (
            _score(formula.left, depth=depth + 1, quantifier_depth=quantifier_depth),
            _score(formula.right, depth=depth + 1, quantifier_depth=quantifier_depth),
        )
        return _combine(
            abstraction_bonus=14,
            local_penalty=0,
            node_count=1,
            atom_count=0,
            negation_count=0,
            conjunction_count=0,
            disjunction_count=0,
            implication_count=1,
            biimplication_count=0,
            exists_count=0,
            forall_count=0,
            max_depth=depth,
            quantifier_depth=quantifier_depth,
            children=children,
        )

    if isinstance(formula, BiImplication):
        children = (
            _score(formula.left, depth=depth + 1, quantifier_depth=quantifier_depth),
            _score(formula.right, depth=depth + 1, quantifier_depth=quantifier_depth),
        )
        return _combine(
            abstraction_bonus=22,
            local_penalty=0,
            node_count=1,
            atom_count=0,
            negation_count=0,
            conjunction_count=0,
            disjunction_count=0,
            implication_count=0,
            biimplication_count=1,
            exists_count=0,
            forall_count=0,
            max_depth=depth,
            quantifier_depth=quantifier_depth,
            children=children,
        )

    if isinstance(formula, Exists):
        child = _score(formula.body, depth=depth + 1, quantifier_depth=quantifier_depth + 1)
        return _combine(
            abstraction_bonus=4,
            local_penalty=0,
            node_count=1,
            atom_count=0,
            negation_count=0,
            conjunction_count=0,
            disjunction_count=0,
            implication_count=0,
            biimplication_count=0,
            exists_count=1,
            forall_count=0,
            max_depth=depth,
            quantifier_depth=quantifier_depth + 1,
            children=(child,),
        )

    if isinstance(formula, Forall):
        child = _score(formula.body, depth=depth + 1, quantifier_depth=quantifier_depth + 1)
        return _combine(
            abstraction_bonus=10,
            local_penalty=0,
            node_count=1,
            atom_count=0,
            negation_count=0,
            conjunction_count=0,
            disjunction_count=0,
            implication_count=0,
            biimplication_count=0,
            exists_count=0,
            forall_count=1,
            max_depth=depth,
            quantifier_depth=quantifier_depth + 1,
            children=(child,),
        )

    raise TypeError(f"Unsupported formula type: {type(formula).__name__}")
