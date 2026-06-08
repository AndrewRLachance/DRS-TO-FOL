from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from ..fol_ast import (
    Atom,
    Conjunction,
    Const,
    Disjunction,
    Exists,
    Forall,
    Formula,
    Negation,
    Term,
    Var,
)

_TOKEN_RE = re.compile(
    r"""
    (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
  | (?P<COMMA>,)
  | (?P<DOT>\.)
  | (?P<AMP>&)
  | (?P<PIPE>\|)
  | (?P<DASH>-)
  | (?P<WS>\s+)
  | (?P<MISMATCH>.)
    """,
    re.VERBOSE,
)

_VAR_RE = re.compile(r"^[etx]\d+$")


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    pos: int


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(src):
        kind = match.lastgroup
        value = match.group()
        pos = match.start()

        if kind == "WS":
            continue
        if kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character {value!r} at position {pos}")

        tokens.append(Token(kind, value, pos))

    tokens.append(Token("EOF", "", len(src)))
    return tokens


def parse_term_name(name: str) -> Term:
    if _VAR_RE.match(name):
        return Var(name)
    return Const(name)


class FOLParser:
    """Recursive-descent parser for the compact FOL-like strings in the dataset."""

    def __init__(self, src: str):
        self.src = src
        self.tokens = tokenize(src)
        self.i = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.i]

    def accept(self, kind: str, value: str | None = None) -> Token | None:
        tok = self.current
        if tok.kind != kind:
            return None
        if value is not None and tok.value != value:
            return None
        self.i += 1
        return tok

    def expect(self, kind: str, value: str | None = None) -> Token:
        tok = self.accept(kind, value)
        if tok is None:
            expected = kind if value is None else f"{kind}({value!r})"
            got = f"{self.current.kind}({self.current.value!r})"
            raise SyntaxError(f"Expected {expected}, got {got} at position {self.current.pos}")
        return tok

    def at_ident(self, value: str) -> bool:
        return self.current.kind == "IDENT" and self.current.value == value

    def parse(self) -> Formula:
        formula = self.parse_formula()
        self.expect("EOF")
        return formula

    def parse_formula(self) -> Formula:
        # Minimal precedence: conjunction/disjunction are parsed left at this layer.
        # The uploaded corpus primarily uses '&'.
        return self.parse_disjunction()

    def parse_disjunction(self) -> Formula:
        parts = [self.parse_conjunction()]
        while self.accept("PIPE"):
            parts.append(self.parse_conjunction())
        if len(parts) == 1:
            return parts[0]
        flat: list[Formula] = []
        for p in parts:
            if isinstance(p, Disjunction):
                flat.extend(p.parts)
            else:
                flat.append(p)
        return Disjunction(tuple(flat))

    def parse_conjunction(self) -> Formula:
        parts = [self.parse_prefix()]
        while self.accept("AMP"):
            parts.append(self.parse_prefix())
        if len(parts) == 1:
            return parts[0]
        flat: list[Formula] = []
        for p in parts:
            if isinstance(p, Conjunction):
                flat.extend(p.parts)
            else:
                flat.append(p)
        return Conjunction(tuple(flat))

    def parse_prefix(self) -> Formula:
        if self.accept("DASH"):
            return Negation(self.parse_prefix())
        if self.at_ident("exists"):
            return self.parse_quantifier("exists")
        if self.at_ident("forall"):
            return self.parse_quantifier("forall")
        if self.accept("LPAREN"):
            inner = self.parse_formula()
            self.expect("RPAREN")
            return inner
        return self.parse_atom()

    def parse_quantifier(self, kind: str) -> Formula:
        self.expect("IDENT", kind)
        vars_: list[Var] = []
        while self.current.kind == "IDENT":
            name = self.current.value
            if name in {"exists", "forall"}:
                raise SyntaxError(f"Expected variable before '.', got {name!r} at {self.current.pos}")
            self.i += 1
            vars_.append(Var(name))
        if not vars_:
            raise SyntaxError(f"{kind!r} quantifier requires at least one variable")
        self.expect("DOT")
        body = self.parse_prefix()
        if kind == "exists":
            return Exists(tuple(vars_), body)
        if kind == "forall":
            return Forall(tuple(vars_), body)
        raise AssertionError(kind)

    def parse_atom(self) -> Atom:
        pred = self.expect("IDENT").value
        self.expect("LPAREN")
        args: list[Term] = []
        if not self.accept("RPAREN"):
            while True:
                name = self.expect("IDENT").value
                args.append(parse_term_name(name))
                if self.accept("COMMA"):
                    continue
                self.expect("RPAREN")
                break
        return Atom(pred, tuple(args))


def parse_fol(src: str) -> Formula:
    return FOLParser(src).parse()


def parse_fol_list_text(text: str) -> list[Formula]:
    raw_items = ast.literal_eval(text)
    if not isinstance(raw_items, list):
        raise TypeError(f"Expected list[str], got {type(raw_items).__name__}")

    formulas: list[Formula] = []
    for i, item in enumerate(raw_items):
        if not isinstance(item, str):
            raise TypeError(f"Item {i} is not a string: {type(item).__name__}")
        try:
            formulas.append(parse_fol(item))
        except SyntaxError as exc:
            raise SyntaxError(f"Failed to parse formula at index {i}: {item}") from exc
    return formulas


def parse_fol_list_file(path: str | Path) -> list[Formula]:
    return parse_fol_list_text(Path(path).read_text(encoding="utf-8"))
