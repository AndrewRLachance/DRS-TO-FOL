from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from ..fol_ast import Var, Term, Formula

@dataclass(frozen=True)
class Entity:
    var: Var
    synset: str | None = None
    name: str | None = None
    properties: tuple[str, ...] = ()


@dataclass(frozen=True)
class Event:
    var: Var
    synset: str
    roles: dict[str, Term] = field(default_factory=dict)


@dataclass(frozen=True)
class Box:
    refs: tuple[Var, ...]
    conditions: tuple[Formula | Entity | Event, ...]