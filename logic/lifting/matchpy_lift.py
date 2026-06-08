from __future__ import annotations

try:
    from matchpy import Arity, Operation, Pattern, Symbol, Wildcard, match
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matchpy is required for connective lifting. Install it with: pip install matchpy"
    ) from exc

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

NOT = Operation.new("NOT", Arity.unary) # type: ignore
AND = Operation.new("AND", Arity.polyadic, associative=True, commutative=True) # type: ignore
OR = Operation.new("OR", Arity.polyadic, associative=True, commutative=True) # type: ignore
IMPLIES = Operation.new("IMPLIES", Arity.binary) # type: ignore
IFF = Operation.new("IFF", Arity.binary) # type: ignore
EXISTS = Operation.new("EXISTS", Arity.binary) # type: ignore
FORALL = Operation.new("FORALL", Arity.binary) # type: ignore

A_ = Wildcard.dot("A")
B_ = Wildcard.dot("B")
P_ = Wildcard.dot("P")


def _make_and(parts):
    parts = tuple(parts)
    if len(parts) == 1:
        return parts[0]
    return AND(*parts)


def _rewrite_rules():
    return [
        # Most specific first.
        (Pattern(AND(IMPLIES(A_, B_), IMPLIES(B_, A_))), lambda s: IFF(s["A"], s["B"])), # type: ignore
        (Pattern(OR(NOT(A_), B_)), lambda s: IMPLIES(s["A"], s["B"])), # type: ignore
        (Pattern(NOT(NOT(P_))), lambda s: s["P"]), # type: ignore
    ]


class AtomEncoder:
    def __init__(self) -> None:
        self.atom_to_symbol: dict[Atom, Symbol] = {}
        self.symbol_to_atom: dict[str, Atom] = {}

    @staticmethod
    def key(atom: Atom) -> str:
        return f"{atom.pred}({','.join(arg.name for arg in atom.args)})"

    def encode(self, atom: Atom) -> Symbol:
        sym = self.atom_to_symbol.get(atom)
        if sym is not None:
            return sym
        key = self.key(atom)
        sym = Symbol(key)
        self.atom_to_symbol[atom] = sym
        self.symbol_to_atom[key] = atom
        return sym

    def decode(self, symbol: Symbol) -> Atom:
        key = str(symbol)
        try:
            return self.symbol_to_atom[key]
        except KeyError as exc:
            raise KeyError(f"No atom registered for MatchPy symbol {key!r}") from exc


def to_matchpy(formula: Formula, enc: AtomEncoder):
    if isinstance(formula, Atom):
        return enc.encode(formula)
    if isinstance(formula, Negation):
        return NOT(to_matchpy(formula.body, enc)) # type: ignore
    if isinstance(formula, Conjunction):
        return AND(*(to_matchpy(part, enc) for part in formula.parts)) # type: ignore
    if isinstance(formula, Disjunction):
        return OR(*(to_matchpy(part, enc) for part in formula.parts))
    if isinstance(formula, Implication):
        return IMPLIES(to_matchpy(formula.left, enc), to_matchpy(formula.right, enc))
    if isinstance(formula, BiImplication):
        return IFF(to_matchpy(formula.left, enc), to_matchpy(formula.right, enc))
    if isinstance(formula, Exists):
        return EXISTS(Symbol(" ".join(v.name for v in formula.vars)), to_matchpy(formula.body, enc)) # type: ignore
    if isinstance(formula, Forall):
        return FORALL(Symbol(" ".join(v.name for v in formula.vars)), to_matchpy(formula.body, enc)) # type: ignore
    raise TypeError(f"Unsupported formula type: {type(formula).__name__}")


def _vars_from_symbol(symbol: Symbol):
    from ..fol_ast import Var

    names = str(symbol).split()
    return tuple(Var(name) for name in names)


def from_matchpy(expr, enc: AtomEncoder) -> Formula:
    if isinstance(expr, Symbol):
        return enc.decode(expr)

    name = type(expr).__name__
    ops = tuple(expr.operands)

    if name == "NOT":
        return Negation(from_matchpy(ops[0], enc))
    if name == "AND":
        return Conjunction(tuple(from_matchpy(x, enc) for x in ops))
    if name == "OR":
        return Disjunction(tuple(from_matchpy(x, enc) for x in ops))
    if name == "IMPLIES":
        return Implication(from_matchpy(ops[0], enc), from_matchpy(ops[1], enc))
    if name == "IFF":
        return BiImplication(from_matchpy(ops[0], enc), from_matchpy(ops[1], enc))
    if name == "EXISTS":
        return Exists(_vars_from_symbol(ops[0]), from_matchpy(ops[1], enc))
    if name == "FORALL":
        return Forall(_vars_from_symbol(ops[0]), from_matchpy(ops[1], enc))

    raise TypeError(f"Unsupported MatchPy expression: {expr!r}")


def rewrite_once(expr):
    for pattern, replacement in _rewrite_rules():
        for subst in match(expr, pattern):
            return replacement(subst)
    return expr


def is_operation_expr(expr) -> bool:
    return hasattr(expr, "operands")


def rewrite_bottom_up(expr):
    if is_operation_expr(expr):
        expr = type(expr)(*(rewrite_bottom_up(child) for child in expr.operands))
    return rewrite_once(expr)


def rewrite_fixed_point(expr, max_steps: int = 100):
    current = expr
    for _ in range(max_steps):
        nxt = rewrite_bottom_up(current)
        if nxt == current:
            return current
        current = nxt
    raise RuntimeError("MatchPy connective lifting did not converge")


def lift_connectives(formula: Formula) -> Formula:
    enc = AtomEncoder()
    expr = to_matchpy(formula, enc)
    lifted = rewrite_fixed_point(expr)
    return from_matchpy(lifted, enc)
