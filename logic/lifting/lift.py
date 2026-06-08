from __future__ import annotations

from ..fol_ast import Formula
from .matchpy_lift import lift_connectives
from .quantifier_lift import lift_quantifiers


def lift_formula(formula: Formula) -> Formula:
    """Lift a formula to higher-level quantifier/connective abstractions."""
    formula = lift_quantifiers(formula)
    formula = lift_connectives(formula)
    # A second quantifier pass catches shapes exposed by connective lifting.
    formula = lift_quantifiers(formula)
    return formula
