from __future__ import annotations

from dataclasses import dataclass

from ..fol_ast import Formula
from .quantifier_lift import lift_quantifiers
from .scoring import FormulaScore, is_better_lift, score_formula


@dataclass(frozen=True)
class LiftResult:
    """Formula-lifting result with before/after scores."""

    original: Formula
    lifted: Formula
    original_score: FormulaScore
    lifted_score: FormulaScore

    @property
    def improved(self) -> bool:
        return is_better_lift(self.original, self.lifted)

    @property
    def delta(self) -> int:
        return self.lifted_score.total - self.original_score.total


def lift_formula(formula: Formula) -> Formula:
    """Lift a formula to higher-level quantifier/connective abstractions.

    Candidate rewrites are accepted only when the scoring heuristic ranks the
    rewritten formula higher than the original formula for that pass.

    This imports MatchPy lazily so parsing/scoring can be used without MatchPy
    installed. Calling this function still requires `pip install matchpy`.
    """
    from .matchpy_lift import lift_connectives

    original = formula

    formula = lift_quantifiers(formula)
    formula = lift_connectives(formula)
    # A second quantifier pass catches shapes exposed by connective lifting.
    formula = lift_quantifiers(formula)

    return formula if is_better_lift(original, formula) else original


def lift_formula_with_scores(formula: Formula) -> LiftResult:
    """Lift a formula and return score metadata for inspection/debugging."""
    lifted = lift_formula(formula)
    return LiftResult(
        original=formula,
        lifted=lifted,
        original_score=score_formula(formula),
        lifted_score=score_formula(lifted),
    )
