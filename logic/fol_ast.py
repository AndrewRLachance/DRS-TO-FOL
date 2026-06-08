from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Term:
    name: str
    alternative_names: frozenset[str] = field(default_factory=frozenset, kw_only=True)
    symbol: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if not self.symbol:
            object.__setattr__(self, "symbol", self.name)


@dataclass(frozen=True)
class Var(Term):
    sort: str | None = None


@dataclass(frozen=True)
class Const(Term):
    pass


@dataclass(frozen=True)
class Formula:
    alternative_names: frozenset[str] = field(default_factory=frozenset, kw_only=True)
    symbol: str = field(default="", kw_only=True)


@dataclass(frozen=True)
class Quantifier(Formula):
    vars: tuple[Var, ...]
    body: Formula


@dataclass(frozen=True)
class Connective(Formula):
    pass


@dataclass(frozen=True)
class UnaryConnective(Connective):
    body: Formula


@dataclass(frozen=True)
class BinarySymmetricConnective(Connective):
    parts: tuple[Formula, ...]


@dataclass(frozen=True)
class BinaryAsymmetricConnective(Connective):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Atom(Formula):
    pred: str
    args: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not self.symbol:
            object.__setattr__(self, "symbol", self.pred)


@dataclass(frozen=True)
class Negation(UnaryConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"not", "negation", "logical not"}),
        kw_only=True,
    )
    symbol: str = field(default="¬", kw_only=True)


@dataclass(frozen=True)
class Conjunction(BinarySymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"and", "conjunction", "logical and"}),
        kw_only=True,
    )
    symbol: str = field(default="∧", kw_only=True)


@dataclass(frozen=True)
class NonConjunction(BinarySymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"nand", "non-conjunction", "not both"}),
        kw_only=True,
    )
    symbol: str = field(default="↑", kw_only=True)


@dataclass(frozen=True)
class NonDisjunction(BinarySymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"nor", "non-disjunction", "neither nor"}),
        kw_only=True,
    )
    symbol: str = field(default="↓", kw_only=True)


@dataclass(frozen=True)
class Disjunction(BinarySymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"or", "disjunction", "logical or"}),
        kw_only=True,
    )
    symbol: str = field(default="∨", kw_only=True)


@dataclass(frozen=True)
class ExclusiveDisjunction(BinarySymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"xor", "exclusive or", "exclusive disjunction"}),
        kw_only=True,
    )
    symbol: str = field(default="⊕", kw_only=True)


@dataclass(frozen=True)
class Implication(BinaryAsymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"implies", "implication", "conditional"}),
        kw_only=True,
    )
    symbol: str = field(default="→", kw_only=True)


@dataclass(frozen=True)
class ConverseImplication(BinaryAsymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"converse implication", "reverse implication", "implied by"}),
        kw_only=True,
    )
    symbol: str = field(default="←", kw_only=True)


@dataclass(frozen=True)
class BiImplication(BinaryAsymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"iff", "if and only if", "biconditional", "bi-implication"}),
        kw_only=True,
    )
    symbol: str = field(default="↔", kw_only=True)


@dataclass(frozen=True)
class NonImplication(BinaryAsymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"non-implication", "does not imply"}),
        kw_only=True,
    )
    symbol: str = field(default="↛", kw_only=True)


@dataclass(frozen=True)
class ConverseNonImplication(BinaryAsymmetricConnective):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"converse non-implication", "reverse non-implication", "not implied by"}),
        kw_only=True,
    )
    symbol: str = field(default="↚", kw_only=True)


@dataclass(frozen=True)
class Exists(Quantifier):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"exists", "there exists", "existential"}),
        kw_only=True,
    )
    symbol: str = field(default="∃", kw_only=True)


@dataclass(frozen=True)
class Forall(Quantifier):
    alternative_names: frozenset[str] = field(
        default_factory=lambda: frozenset({"forall", "for all", "universal"}),
        kw_only=True,
    )
    symbol: str = field(default="∀", kw_only=True)
