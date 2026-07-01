#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from nltk.sem.drt import DrtExpression

from logic.semantic_ir import (
    AccessibilityFrame,
    DiscourseFrame,
    EntityFrame,
    EqualityFrame,
    EventFrame,
    GeneralizedQuantifierFrame,
    ModalFrame,
    NameFrame,
    PropositionFrame,
    PropositionTarget,
    QuantityFrame,
    SemanticBox,
    SemanticCondition,
    SemanticDocument,
    SemanticReferent,
    TimeFrame,
)
from pmb_scripts.sbn2penman import SBNGraph
from pmb_scripts.sbn_spec import SBNSpec, SBN_EDGE_TYPE, SBN_NODE_TYPE, split_synset_id


# =============================================================================
# Identifier handling
# =============================================================================

NLTK_RESERVED = {
    "all", "exists", "exist",
    "and", "or", "not",
    "true", "false",
    "iff", "implies",
}


def sanitize_ident(s: str, *, prefix: str = "C") -> str:
    s = s.strip().strip('"').strip("'")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

    if not s:
        s = "EMPTY"

    if not re.match(r"^[A-Za-z]", s):
        s = f"{prefix}_{s}"

    if s.lower() in NLTK_RESERVED:
        s = f"{prefix}_{s}"

    if re.fullmatch(r"[A-Za-z]", s):
        s = f"{prefix}_{s}"

    return s


def pred_from_synset(token: str) -> str:
    return sanitize_ident(token.replace(".", "_").replace("-", "_"), prefix="p")


def pred_from_role(label: str) -> str:
    return sanitize_ident(label.replace("-", "_"), prefix="p")


def const_from_token(token: str) -> str:
    return sanitize_ident(token, prefix="C")


def var_for_synset(node_id, node_data) -> str:
    token = node_data["token"]
    parsed = split_synset_id(token)
    idx = node_id[1] + 1

    if not parsed:
        return f"x{idx}"

    lemma, pos, _sense = parsed

    if pos == "n" and lemma == "time":
        return f"t{idx}"

    if pos in {"n", "x"}:
        return f"x{idx}"

    return f"e{idx}"


# =============================================================================
# Diagnostics
# =============================================================================

@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    label: Optional[str] = None
    scope_marker: Optional[str] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    edge_id: Optional[str] = None
    details: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }

        for key in ("label", "scope_marker", "source_id", "target_id", "edge_id"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value

        if self.details:
            out["details"] = dict(self.details)

        return out


class Diagnostics:
    def __init__(self) -> None:
        self._items: List[Diagnostic] = []
        self._seen: Set[Tuple[Tuple[str, Any], ...]] = set()

    def add(
        self,
        code: str,
        severity: str,
        message: str,
        *,
        label: Optional[str] = None,
        scope_marker: Optional[str] = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_id: Optional[str] = None,
        details: Optional[Dict[str, str]] = None,
    ) -> None:
        diagnostic = Diagnostic(
            code=code,
            severity=severity,
            message=message,
            label=label,
            scope_marker=scope_marker,
            source_id=source_id,
            target_id=target_id,
            edge_id=edge_id,
            details=details or {},
        )
        key = tuple(
            sorted(
                (k, json.dumps(v, sort_keys=True) if isinstance(v, dict) else v)
                for k, v in diagnostic.to_dict().items()
            )
        )
        if key in self._seen:
            return
        self._seen.add(key)
        self._items.append(diagnostic)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self._items]


def graph_id_to_str(node_id) -> str:
    kind, idx = node_id
    value = kind.value if hasattr(kind, "value") else str(kind)
    return f"{value}:{idx}"


# =============================================================================
# AST
# =============================================================================

@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[str, ...]

    def render(self) -> str:
        return f"{self.pred}({','.join(self.args)})"


@dataclass(frozen=True)
class Eq:
    left: str
    right: str

    def render(self) -> str:
        # NLTK DRT accepts equality as "x = y" inside DRS conditions.
        return f"{self.left} = {self.right}"


@dataclass(frozen=True)
class Not:
    body: "DRS"

    def render(self) -> str:
        return f"-({self.body.render()})"


@dataclass(frozen=True)
class Imp:
    ant: "DRS"
    cons: "DRS"

    def render(self) -> str:
        return f"({self.ant.render()} -> {self.cons.render()})"


@dataclass(frozen=True)
class Or:
    left: "DRS"
    right: "DRS"

    def render(self) -> str:
        return f"({self.left.render()} | {self.right.render()})"


Cond = Union[Atom, Eq, Not, Imp, Or, "DRS"]


@dataclass
class DRS:
    refs: List[str] = field(default_factory=list)
    conds: List[Cond] = field(default_factory=list)

    def dedupe(self) -> "DRS":
        self.refs = unique_preserving_order(self.refs)

        seen: Set[str] = set()
        new_conds: List[Cond] = []

        for cond in self.conds:
            key = cond.render() if hasattr(cond, "render") else str(cond)
            if key not in seen:
                seen.add(key)
                new_conds.append(cond)

        self.conds = new_conds
        return self

    def extend(self, other: "DRS") -> None:
        self.refs.extend(other.refs)
        self.conds.extend(other.conds)
        self.dedupe()

    def without_conds(self, excluded: Iterable[Cond]) -> "DRS":
        excluded_keys = {
            c.render() if hasattr(c, "render") else str(c)
            for c in excluded
        }

        return DRS(
            refs=list(self.refs),
            conds=[
                c for c in self.conds
                if (c.render() if hasattr(c, "render") else str(c)) not in excluded_keys
            ],
        ).dedupe()

    def render(self) -> str:
        self.dedupe()
        refs = ",".join(self.refs)
        conds = ", ".join(cond.render() for cond in self.conds)
        return f"DRS([{refs}],[{conds}])"


# =============================================================================
# Graph helpers
# =============================================================================

def is_synset(G, node_id) -> bool:
    return G.nodes[node_id]["type"] == SBN_NODE_TYPE.SYNSET


def is_constant(G, node_id) -> bool:
    return G.nodes[node_id]["type"] == SBN_NODE_TYPE.CONSTANT


def is_box(G, node_id) -> bool:
    return G.nodes[node_id]["type"] == SBN_NODE_TYPE.BOX


def node_token(G, node_id) -> str:
    return str(G.nodes[node_id].get("token", ""))


def get_or_make_term(G, node_id, node_terms: Dict) -> Optional[str]:
    if node_id in node_terms:
        return node_terms[node_id]

    node_data = G.nodes[node_id]

    if node_data["type"] == SBN_NODE_TYPE.SYNSET:
        term = var_for_synset(node_id, node_data)
    elif node_data["type"] == SBN_NODE_TYPE.CONSTANT:
        term = const_from_token(node_data["token"])
    else:
        return None

    node_terms[node_id] = term
    return term


def root_box(G):
    roots = [n for n, deg in G.in_degree() if deg == 0 and is_box(G, n)]

    if not roots:
        raise ValueError("No root BOX node found.")

    return sorted(roots, key=lambda n: n[1])[0]


def box_members(G, box_id) -> List:
    return [
        child
        for _, child, data in G.out_edges(box_id, data=True)
        if data["type"] == SBN_EDGE_TYPE.BOX_CONNECT
    ]


def grouped_child_boxes(G, box_id) -> Dict[str, List]:
    groups: Dict[str, List] = defaultdict(list)

    for _, child, data in G.out_edges(box_id, data=True):
        if data["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
            groups[str(data["token"])].append(child)

    return groups


def unique_preserving_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()

    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)

    return out


def is_scope_marker_constant(G, node_id) -> bool:
    if not is_constant(G, node_id):
        return False

    tok = node_token(G, node_id)
    return tok.startswith("<") or tok.startswith(">")


SUPPORTED_SCOPAL_LABELS = {
    "NEGATION",
    "CONJUNCTION",
    "CONDITION",
    "PRECONDITION",
    "CONSEQUENCE",
    "ALTERNATION",
    "ATTRIBUTION",
    "COMMENTARY",
    "CONTINUATION",
    "ELABORATION",
    "EXPLANATION",
    "CONTRAST",
    "RESULT",
    "SOURCE",
}

MODAL_LABELS = {
    "POSSIBILITY",
    "NECESSITY",
}

DISCOURSE_RELATION_LABELS = {
    "ATTRIBUTION",
    "COMMENTARY",
    "CONTINUATION",
    "CONTRAST",
    "ELABORATION",
    "EXPLANATION",
    "RESULT",
    "SOURCE",
}

STRUCTURAL_DISCOURSE_LABELS = {
    "CONTINUATION",
    "ELABORATION",
    "EXPLANATION",
    "CONTRAST",
    "RESULT",
    "SOURCE",
}

SUPPORTED_SCOPAL_LABELS.update(MODAL_LABELS)

TEMPORAL_ROLES = {
    "Time",
    "ClockTime",
    "DayOfMonth",
    "DayOfWeek",
    "Decade",
    "MonthOfYear",
    "YearOfCentury",
    "Duration",
    "Start",
    "Finish",
}

TEMPORAL_OPERATORS = {
    "EQU",
    "NEQ",
    "APX",
    "LES",
    "LEQ",
    "TPR",
    "TAB",
    "TIN",
}

EQUALITY_LABELS = {"EQU", "EQ", "=", "Equal"}
QUANTITY_ROLES = {"Quantity", "Measure", "Unit", "Value", "Degree", "Extent"}
AMBIGUOUS_QUANTITY_VALUES = {"?", "+", "-"}


def edge_context(G, source, target, data) -> Dict[str, Optional[str]]:
    scope_marker = data.get("scope_marker")
    if scope_marker is None and target in G.nodes and is_scope_marker_constant(G, target):
        scope_marker = node_token(G, target)

    return {
        "label": str(data["token"]),
        "scope_marker": str(scope_marker) if scope_marker is not None else None,
        "source_id": graph_id_to_str(source),
        "target_id": graph_id_to_str(target),
        "edge_id": str(data.get("_id", "")) or None,
    }


def collect_graph_diagnostics(G: SBNGraph, diagnostics: Diagnostics) -> None:
    if getattr(G, "is_possibly_ill_formed", False):
        diagnostics.add(
            "possibly_ill_formed_graph",
            "warning",
            "SBNGraph marked this graph as possibly ill formed.",
        )

    for source, target, data in G.edges(data=True):
        edge_type = data["type"]
        label = str(data["token"])
        context = edge_context(G, source, target, data)

        if edge_type == SBN_EDGE_TYPE.ROLE and label not in SBNSpec.ROLES:
            diagnostics.add(
                "unknown_role",
                "warning",
                f"Role {label!r} is not listed in SBNSpec.ROLES.",
                **context,
            )

        if edge_type == SBN_EDGE_TYPE.DRS_OPERATOR and label not in SBNSpec.DRS_OPERATORS:
            diagnostics.add(
                "unknown_operator",
                "warning",
                f"DRS operator {label!r} is not listed in SBNSpec.DRS_OPERATORS.",
                **context,
            )

        if edge_type == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
            if label not in SBNSpec.NEW_BOX_INDICATORS:
                diagnostics.add(
                    "unknown_scopal_operator",
                    "warning",
                    f"Scopal operator {label!r} is not listed in SBNSpec.NEW_BOX_INDICATORS.",
                    **context,
                )
            elif label not in SUPPORTED_SCOPAL_LABELS:
                diagnostics.add(
                    "unsupported_scopal_structure",
                    "warning",
                    f"Scopal operator {label!r} is preserved structurally.",
                    **context,
                )

        if (
            edge_type in {SBN_EDGE_TYPE.ROLE, SBN_EDGE_TYPE.DRS_OPERATOR}
            and target in G.nodes
            and is_constant(G, target)
            and SBNSpec.ROLE_INDEX_PATTERN.match(node_token(G, target))
        ):
            diagnostics.add(
                "malformed_graph_fallback_edge",
                "warning",
                "Relative role offset resolved outside the available synset range and was kept as a constant.",
                **context,
            )


def scoped_box_edges_for_marker(G, scope_marker: str) -> List[Tuple[Any, Any, Dict[str, Any]]]:
    return [
        (source, target, data)
        for source, target, data in G.edges(data=True)
        if (
            data["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT
            and str(data.get("scope_marker", "")) == scope_marker
        )
    ]


def empty_semantic_model() -> Dict[str, Any]:
    return SemanticDocument.empty().to_dict()


def referent_kind(node_data: Dict[str, Any]) -> str:
    parsed = split_synset_id(str(node_data.get("token", "")))
    if not parsed:
        return "unknown"

    lemma, pos, _sense = parsed
    if pos == "n" and lemma == "time":
        return "time"
    if pos in {"n", "x"}:
        return "entity"
    if pos == "v":
        return "event"
    return "unknown"


def raw_value(G, node_id) -> str:
    return node_token(G, node_id).strip().strip('"').strip("'")


def node_box_map(G) -> Dict[Any, Any]:
    out: Dict[Any, Any] = {}
    for box_id in [node for node in G.nodes if is_box(G, node)]:
        for member in box_members(G, box_id):
            out[member] = box_id
    return out


def first_incoming_box_edge(G, box_id):
    for source, target, data in G.in_edges(box_id, data=True):
        if data["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
            return source, target, data
    return None


def condition_target(G, target, node_terms: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if is_synset(G, target):
        return graph_id_to_str(target), get_or_make_term(G, target, node_terms), None
    if is_constant(G, target):
        return graph_id_to_str(target), const_from_token(node_token(G, target)), raw_value(G, target)
    return graph_id_to_str(target), None, None


def graph_to_semantic_ir(
    G: SBNGraph,
    diagnostics: Optional[Diagnostics] = None,
) -> SemanticDocument:
    node_terms: Dict = {}
    node_boxes = node_box_map(G)
    frames: List[Any] = []
    referents: List[SemanticReferent] = []
    conditions: List[SemanticCondition] = []
    boxes: List[SemanticBox] = []

    for box_id in [node for node in G.nodes if is_box(G, node)]:
        incoming = first_incoming_box_edge(G, box_id)
        if incoming is None:
            parent_id = None
            operator = "ROOT"
            scope_marker = None
        else:
            parent, _target, data = incoming
            parent_id = graph_id_to_str(parent)
            operator = str(data["token"])
            scope_marker = str(data["scope_marker"]) if data.get("scope_marker") is not None else None

        local_referents = tuple(
            get_or_make_term(G, member, node_terms)
            for member in box_members(G, box_id)
            if is_synset(G, member)
        )
        child_boxes = tuple(
            graph_id_to_str(child)
            for _source, child, data in G.out_edges(box_id, data=True)
            if data["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT
        )
        boxes.append(
            SemanticBox(
                id=graph_id_to_str(box_id),
                parent_id=parent_id,
                operator=operator,
                scope_marker=scope_marker,
                local_referents=tuple(ref for ref in local_referents if ref is not None),
                child_boxes=child_boxes,
            )
        )

    for node_id, node_data in G.nodes(data=True):
        if not is_synset(G, node_id):
            continue

        term = get_or_make_term(G, node_id, node_terms)
        if term is None:
            continue

        kind = referent_kind(node_data)
        box_id = node_boxes.get(node_id)
        referents.append(
            SemanticReferent(
                id=graph_id_to_str(node_id),
                term=term,
                source_id=graph_id_to_str(node_id),
                source_token=node_token(G, node_id),
                kind=kind,
                box_id=graph_id_to_str(box_id) if box_id is not None else None,
            )
        )

        if kind == "entity":
            frames.append(
                EntityFrame(
                    id=f"entity:{graph_id_to_str(node_id)}",
                    referent_id=graph_id_to_str(node_id),
                    term=term,
                    synset=node_token(G, node_id),
                    box_id=graph_id_to_str(box_id) if box_id is not None else None,
                )
            )
        elif kind == "event":
            roles: List[Dict[str, str]] = []
            for _source, target, data in G.out_edges(node_id, data=True):
                if data["type"] != SBN_EDGE_TYPE.ROLE or is_scope_marker_constant(G, target):
                    continue
                target_id, target_term, value = condition_target(G, target, node_terms)
                role = {
                    "role": str(data["token"]),
                    "target_id": target_id or "",
                }
                if target_term is not None:
                    role["target_term"] = target_term
                if value is not None:
                    role["value"] = value
                roles.append(role)
            frames.append(
                EventFrame(
                    id=f"event:{graph_id_to_str(node_id)}",
                    event_id=graph_id_to_str(node_id),
                    term=term,
                    synset=node_token(G, node_id),
                    roles=tuple(roles),
                    box_id=graph_id_to_str(box_id) if box_id is not None else None,
                )
            )
        elif kind == "time":
            frames.append(
                TimeFrame(
                    id=f"time:{graph_id_to_str(node_id)}",
                    time_id=graph_id_to_str(node_id),
                    term=term,
                    relation="referent",
                    value=node_token(G, node_id),
                    source_id=graph_id_to_str(node_id),
                )
            )

    for i, (source, target, data) in enumerate(G.edges(data=True)):
        edge_type = data["type"]
        if edge_type not in {SBN_EDGE_TYPE.ROLE, SBN_EDGE_TYPE.DRS_OPERATOR}:
            continue
        if source not in G.nodes or target not in G.nodes:
            continue

        label = str(data["token"])
        source_term = get_or_make_term(G, source, node_terms)
        target_id, target_term, value = condition_target(G, target, node_terms)
        source_box = node_boxes.get(source)

        conditions.append(
            SemanticCondition(
                id=f"condition:{i}",
                label=label,
                kind="operator" if edge_type == SBN_EDGE_TYPE.DRS_OPERATOR else "role",
                source_id=graph_id_to_str(source),
                source_term=source_term,
                target_id=target_id,
                target_term=target_term,
                value=value,
                box_id=graph_id_to_str(source_box) if source_box is not None else None,
            )
        )

        if label == "Name" and source_term is not None:
            frames.append(
                NameFrame(
                    id=f"name:{i}",
                    referent_id=graph_id_to_str(source),
                    term=source_term,
                    name=value or target_term or "",
                )
            )

        if label in EQUALITY_LABELS and source_term is not None:
            frames.append(
                EqualityFrame(
                    id=f"equality:{i}",
                    operator=label,
                    left_id=graph_id_to_str(source),
                    left_term=source_term,
                    right_id=target_id,
                    right_term=target_term,
                    value=value,
                )
            )

        source_is_time = is_synset(G, source) and referent_kind(G.nodes[source]) == "time"
        target_is_time = is_synset(G, target) and referent_kind(G.nodes[target]) == "time"
        if label in TEMPORAL_ROLES or (label in TEMPORAL_OPERATORS and (source_is_time or target_is_time)):
            time_id = graph_id_to_str(target) if target_is_time else None
            if time_id is None and source_is_time:
                time_id = graph_id_to_str(source)
            frames.append(
                TimeFrame(
                    id=f"time-relation:{i}",
                    time_id=time_id,
                    term=target_term if time_id == graph_id_to_str(target) else source_term,
                    relation=label,
                    value=value,
                    source_id=graph_id_to_str(source),
                    target_id=target_id,
                )
            )

        if label in QUANTITY_ROLES and source_term is not None:
            frames.append(
                QuantityFrame(
                    id=f"quantity:{i}",
                    source_id=graph_id_to_str(source),
                    source_term=source_term,
                    role=label,
                    value=value or target_term,
                    unit=value if label == "Unit" else None,
                    measure=value if label == "Measure" else None,
                )
            )
            if value in AMBIGUOUS_QUANTITY_VALUES and diagnostics is not None:
                diagnostics.add(
                    "generalized_quantifier_uncertain",
                    "warning",
                    "Quantity-like marker may require generalized quantifier semantics; preserving existential DRS behavior.",
                    **edge_context(G, source, target, data),
                )
                frames.append(
                    GeneralizedQuantifierFrame(
                        id=f"generalized-quantifier:{i}",
                        source_id=graph_id_to_str(source),
                        quantifier=value,
                        status="uncertain",
                        details={"role": label},
                    )
                )

    for i, (source, target, data) in enumerate(G.edges(data=True)):
        if data["type"] != SBN_EDGE_TYPE.BOX_BOX_CONNECT:
            continue

        parent_refs = tuple(
            get_or_make_term(G, member, node_terms)
            for member in box_members(G, source)
            if is_synset(G, member)
        )
        child_refs = tuple(
            get_or_make_term(G, member, node_terms)
            for member in box_members(G, target)
            if is_synset(G, member)
        )
        frames.append(
            AccessibilityFrame(
                id=f"accessibility:{i}",
                parent_box_id=graph_id_to_str(source),
                child_box_id=graph_id_to_str(target),
                operator=str(data["token"]),
                scope_marker=str(data["scope_marker"]) if data.get("scope_marker") is not None else None,
                local_referents=tuple(ref for ref in child_refs if ref is not None),
                inherited_referents=tuple(ref for ref in parent_refs if ref is not None),
            )
        )

        label = str(data["token"])
        scope_marker = data.get("scope_marker")
        target_drs = drs_for_box(G, target, node_terms).render()

        if label in MODAL_LABELS:
            frames.append(
                ModalFrame(
                    operator=label,
                    source_box_id=graph_id_to_str(source),
                    target_box_id=graph_id_to_str(target),
                    scope_marker=str(scope_marker) if scope_marker is not None else None,
                    drs=target_drs,
                    lowered_to_fol=False,
                    fol_handling="embedded_content_preserved_modality_dropped",
                )
            )
            if diagnostics is not None:
                diagnostics.add(
                    "modal_scope_modeled",
                    "info",
                    "Modal scope was modeled in semantic output but modality was not lowered to FOL.",
                    **edge_context(G, source, target, data),
                )

        if label in DISCOURSE_RELATION_LABELS:
            fol_handling = (
                "merged_structurally"
                if label in STRUCTURAL_DISCOURSE_LABELS
                else "embedded_drs_content"
            )
            frames.append(
                DiscourseFrame(
                    relation=label,
                    source_box_id=graph_id_to_str(source),
                    target_box_id=graph_id_to_str(target),
                    scope_marker=str(scope_marker) if scope_marker is not None else None,
                    drs=target_drs,
                    lowered_to_fol=False,
                    fol_handling=fol_handling,
                )
            )
            if diagnostics is not None:
                diagnostics.add(
                    "discourse_relation_modeled",
                    "info",
                    "Discourse relation was modeled in semantic output but not lowered to FOL.",
                    **edge_context(G, source, target, data),
                )

    for source, target, data in G.edges(data=True):
        if data["type"] != SBN_EDGE_TYPE.ROLE:
            continue
        if str(data["token"]) != "Proposition":
            continue
        if not is_scope_marker_constant(G, target):
            continue

        scope_marker = node_token(G, target)
        source_term = get_or_make_term(G, source, node_terms)
        if source_term is None:
            continue

        targets: List[PropositionTarget] = []
        for _box_source, box_target, box_data in scoped_box_edges_for_marker(G, scope_marker):
            target_drs = drs_for_box(G, box_target, node_terms).render()
            targets.append(
                PropositionTarget(
                    box_id=graph_id_to_str(box_target),
                    operator=str(box_data["token"]),
                    drs=target_drs,
                )
            )

        frames.append(
            PropositionFrame(
                source_id=graph_id_to_str(source),
                source_term=source_term,
                source_token=node_token(G, source),
                role=str(data["token"]),
                scope_marker=scope_marker,
                lowered_to_fol=False,
                target_boxes=tuple(targets),
            )
        )

        if diagnostics is not None:
            diagnostics.add(
                "proposition_scope_modeled",
                "info",
                "Proposition scope was modeled in semantic output but not lowered to FOL.",
                **edge_context(G, source, target, data),
            )
            if not targets:
                diagnostics.add(
                    "unresolved_proposition_scope",
                    "warning",
                    "Proposition scope marker has no matching scoped box in the parsed graph.",
                    **edge_context(G, source, target, data),
                )

    return SemanticDocument(
        boxes=tuple(boxes),
        referents=tuple(referents),
        conditions=tuple(conditions),
        frames=tuple(frames),
        diagnostics=tuple(diagnostics.to_dicts() if diagnostics is not None else ()),
    )


def graph_to_semantic_model(
    G: SBNGraph,
    diagnostics: Optional[Diagnostics] = None,
) -> Dict[str, Any]:
    return graph_to_semantic_ir(G, diagnostics).to_dict()


# =============================================================================
# Local AST construction
# =============================================================================

def local_drs_for_box(
    G,
    box_id,
    node_terms: Dict,
    diagnostics: Optional[Diagnostics] = None,
) -> DRS:
    """
    Build only the local contents of a box:
    - local synset referents
    - local synset predicates
    - role/operator edges whose source synset is a local member

    This intentionally does not recurse into scoped child boxes.
    """
    refs: List[str] = []
    conds: List[Cond] = []
    emitted_synsets: Set = set()

    members = box_members(G, box_id)
    member_set = set(members)

    # 1. Local synset predicates.
    for node_id in members:
        if not is_synset(G, node_id):
            continue

        term = get_or_make_term(G, node_id, node_terms)
        if term is None:
            continue

        refs.append(term)

        if node_id not in emitted_synsets:
            pred = pred_from_synset(G.nodes[node_id]["token"])
            conds.append(Atom(pred, (term,)))
            emitted_synsets.add(node_id)

    # 2. Local role/operator edges from local source synsets.
    for source, target, data in G.edges(data=True):
        if source not in member_set:
            continue

        edge_type = data["type"]
        if edge_type not in {SBN_EDGE_TYPE.ROLE, SBN_EDGE_TYPE.DRS_OPERATOR}:
            continue

        if is_scope_marker_constant(G, target):
            if diagnostics is not None:
                diagnostics.add(
                    "skipped_scope_marker_edge",
                    "info",
                    "Scope-marker edge was skipped during DRS/FOL emission.",
                    **edge_context(G, source, target, data),
                )
            continue

        source_term = get_or_make_term(G, source, node_terms)
        target_term = get_or_make_term(G, target, node_terms)

        if source_term is None or target_term is None:
            continue

        label = str(data["token"])

        if edge_type == SBN_EDGE_TYPE.DRS_OPERATOR and label.upper() in {"EQU", "EQ", "="}:
            conds.append(Eq(source_term, target_term))
        else:
            pred = pred_from_role(label)
            conds.append(Atom(pred, (source_term, target_term)))

    return DRS(refs, conds).dedupe()


# =============================================================================
# Scoped child handling
# =============================================================================

def drs_has_content(drs: DRS) -> bool:
    return bool(drs.refs or drs.conds)


def emit_conditionals(
    G,
    groups: Dict[str, List],
    node_terms: Dict,
    diagnostics: Optional[Diagnostics] = None,
) -> Tuple[List[Cond], Set]:
    out: List[Cond] = []
    used: Set = set()

    antecedents: List = []
    antecedents.extend(groups.get("CONDITION", []))
    antecedents.extend(groups.get("PRECONDITION", []))

    consequents = groups.get("CONSEQUENCE", [])

    n = min(len(antecedents), len(consequents))

    for i in range(n):
        ant_drs = drs_for_box(G, antecedents[i], node_terms, diagnostics)
        cons_drs = drs_for_box(G, consequents[i], node_terms, diagnostics)

        out.append(Imp(ant_drs, cons_drs))
        used.add(antecedents[i])
        used.add(consequents[i])

    return out, used


def emit_alternation(
    G,
    groups: Dict[str, List],
    node_terms: Dict,
    diagnostics: Optional[Diagnostics] = None,
) -> Tuple[List[Cond], Set]:
    out: List[Cond] = []
    used: Set = set()

    alts = groups.get("ALTERNATION", [])

    if len(alts) >= 2:
        drs_items = [drs_for_box(G, alt, node_terms, diagnostics) for alt in alts]

        expr: Cond = Or(drs_items[0], drs_items[1])
        for item in drs_items[2:]:
            # Or expects DRS operands, so wrap previous Or in a DRS condition.
            expr = Or(DRS([], [expr]), item)

        out.append(expr)
        used.update(alts)

    return out, used


def try_negated_implication_from_box(
    G,
    negated_box,
    node_terms: Dict,
    diagnostics: Optional[Diagnostics] = None,
) -> Optional[Imp]:
    """
    Detect SBN/DRS conditional pattern:

        ¬( A ∧ ¬B )

    represented as a NEGATION child box containing local antecedent material
    plus exactly one nested NEGATION child box.

    Return:

        A -> B

    The caller is responsible for using this only when processing a NEGATION
    edge into `negated_box`.
    """
    local = local_drs_for_box(G, negated_box, node_terms, diagnostics)
    groups = grouped_child_boxes(G, negated_box)

    nested_negs = groups.get("NEGATION", [])

    # Be conservative. Multiple inner negations are not safely rewritable
    # without more structural information.
    if len(nested_negs) != 1:
        return None

    inner_neg_box = nested_negs[0]

    # Antecedent is the local content plus non-negation child boxes that are
    # part of the same conjunction/background material.
    ant = DRS(list(local.refs), list(local.conds)).dedupe()

    for label, boxes in groups.items():
        if label == "NEGATION":
            continue

        if label == "CONJUNCTION":
            for child in boxes:
                ant.extend(drs_for_box(G, child, node_terms, diagnostics))
        elif label in STRUCTURAL_DISCOURSE_LABELS:
            for child in boxes:
                ant.extend(drs_for_box(G, child, node_terms, diagnostics))
        else:
            # Unknown scoped content in the negated box makes the rewrite risky.
            if diagnostics is not None:
                diagnostics.add(
                    "unsupported_scopal_structure",
                    "warning",
                    f"Nested scopal operator {label!r} prevents safe negated-implication rewriting.",
                    label=label,
                    target_id=graph_id_to_str(boxes[0]) if boxes else None,
                )
            return None

    cons = drs_for_box(G, inner_neg_box, node_terms, diagnostics)

    if not drs_has_content(ant) or not drs_has_content(cons):
        return None

    return Imp(ant, cons)


def add_remaining_child_boxes(
    G,
    target: DRS,
    groups: Dict[str, List],
    used_child_boxes: Set,
    node_terms: Dict,
    diagnostics: Optional[Diagnostics] = None,
) -> None:
    for label, boxes in groups.items():
        for child in boxes:
            if child in used_child_boxes:
                continue

            if label == "NEGATION":
                implication = try_negated_implication_from_box(G, child, node_terms, diagnostics)
                if implication is not None:
                    target.conds.append(implication)
                else:
                    child_drs = drs_for_box(G, child, node_terms, diagnostics)
                    target.conds.append(Not(child_drs))

            elif label == "CONJUNCTION":
                # SBN CONJUNCTION should behave as merged conjunctive material,
                # not as an embedded DRS condition.
                child_drs = drs_for_box(G, child, node_terms, diagnostics)
                target.extend(child_drs)

            elif label in MODAL_LABELS:
                # NLTK DRT has no native modal FOL operator.
                # Preserve the propositional content but drop modality.
                if diagnostics is not None:
                    diagnostics.add(
                        "unsupported_modal_operator",
                        "warning",
                        f"Modal operator {label!r} is not faithfully represented; embedded content was preserved.",
                        label=label,
                        target_id=graph_id_to_str(child),
                    )
                child_drs = drs_for_box(G, child, node_terms, diagnostics)
                target.extend(child_drs)

            elif label in {"CONDITION", "PRECONDITION", "CONSEQUENCE"}:
                # Paired cases are handled earlier. Unpaired cases are preserved
                # as embedded DRS content rather than silently discarded.
                if diagnostics is not None:
                    diagnostics.add(
                        "unpaired_scopal_operator",
                        "warning",
                        f"Unpaired scopal operator {label!r} was preserved as embedded DRS content.",
                        label=label,
                        target_id=graph_id_to_str(child),
                    )
                child_drs = drs_for_box(G, child, node_terms, diagnostics)
                target.conds.append(child_drs)

            elif label == "ALTERNATION":
                # Paired alternations are handled earlier. A single unpaired
                # alternation has no disjunction to form, so preserve content.
                if diagnostics is not None:
                    diagnostics.add(
                        "unpaired_alternation",
                        "warning",
                        "Unpaired ALTERNATION was preserved as embedded DRS content.",
                        label=label,
                        target_id=graph_id_to_str(child),
                    )
                child_drs = drs_for_box(G, child, node_terms, diagnostics)
                target.conds.append(child_drs)

            else:
                # CONTINUATION, ATTRIBUTION, EXPLANATION, CONTRAST,
                # ELABORATION, RESULT, SOURCE, etc.
                child_drs = drs_for_box(G, child, node_terms, diagnostics)

                if label in STRUCTURAL_DISCOURSE_LABELS:
                    if diagnostics is not None:
                        diagnostics.add(
                            "structural_discourse_relation",
                            "info",
                            f"Discourse relation {label!r} was merged structurally without full semantics.",
                            label=label,
                            target_id=graph_id_to_str(child),
                        )
                    target.extend(child_drs)
                elif label in DISCOURSE_RELATION_LABELS:
                    if diagnostics is not None:
                        diagnostics.add(
                            "embedded_discourse_relation",
                            "info",
                            f"Discourse relation {label!r} was preserved as embedded DRS content without full semantics.",
                            label=label,
                            target_id=graph_id_to_str(child),
                        )
                    target.conds.append(child_drs)
                else:
                    if diagnostics is not None:
                        diagnostics.add(
                            "unsupported_scopal_structure",
                            "warning",
                            f"Scopal operator {label!r} was preserved as embedded DRS content.",
                            label=label,
                            target_id=graph_id_to_str(child),
                        )
                    target.conds.append(child_drs)

    target.dedupe()


def drs_for_box(
    G,
    box_id,
    node_terms: Dict,
    diagnostics: Optional[Diagnostics] = None,
) -> DRS:
    drs = local_drs_for_box(G, box_id, node_terms, diagnostics)

    groups = grouped_child_boxes(G, box_id)
    used_child_boxes: Set = set()

    conditional_conds, conditional_used = emit_conditionals(G, groups, node_terms, diagnostics)
    drs.conds.extend(conditional_conds)
    used_child_boxes.update(conditional_used)

    alt_conds, alt_used = emit_alternation(G, groups, node_terms, diagnostics)
    drs.conds.extend(alt_conds)
    used_child_boxes.update(alt_used)

    add_remaining_child_boxes(
        G,
        drs,
        groups,
        used_child_boxes,
        node_terms,
        diagnostics,
    )

    return drs.dedupe()


# =============================================================================
# Normalization
# =============================================================================

def normalize_cond(cond: Cond) -> Cond:
    if isinstance(cond, Not):
        return normalize_not(cond)

    if isinstance(cond, Imp):
        return Imp(normalize_drs(cond.ant), normalize_drs(cond.cons))

    if isinstance(cond, Or):
        return Or(normalize_drs(cond.left), normalize_drs(cond.right))

    if isinstance(cond, DRS):
        return normalize_drs(cond)

    return cond


def normalize_not(cond: Not) -> Cond:
    """
    General AST-level rewrite:

        ¬ DRS(A_refs, A_conds + [¬B])
        =>
        DRS(A_refs, A_conds) -> B

    This catches nested-negation conditionals even if they were not caught
    during SBN box construction.
    """
    body = normalize_drs(cond.body)

    inner_nots = [c for c in body.conds if isinstance(c, Not)]

    if len(inner_nots) == 1:
        inner = inner_nots[0]
        ant = body.without_conds([inner])
        cons = normalize_drs(inner.body)

        if drs_has_content(ant) and drs_has_content(cons):
            return Imp(ant, cons)

    return Not(body)


def normalize_drs(drs: DRS) -> DRS:
    return DRS(
        refs=list(drs.refs),
        conds=[normalize_cond(c) for c in drs.conds],
    ).dedupe()


# =============================================================================
# Public conversion functions
# =============================================================================

def graph_to_ast(G: SBNGraph) -> DRS:
    node_terms: Dict = {}
    ast = drs_for_box(G, root_box(G), node_terms)
    return normalize_drs(ast)


def graph_to_ast_with_diagnostics(G: SBNGraph) -> Tuple[DRS, List[Dict[str, Any]]]:
    diagnostics = Diagnostics()
    collect_graph_diagnostics(G, diagnostics)
    graph_to_semantic_model(G, diagnostics)

    node_terms: Dict = {}
    ast = drs_for_box(G, root_box(G), node_terms, diagnostics)
    return normalize_drs(ast), diagnostics.to_dicts()


def graph_to_drs(G: SBNGraph) -> str:
    return graph_to_ast(G).render()


def graph_to_drs_with_diagnostics(G: SBNGraph) -> Tuple[str, List[Dict[str, Any]]]:
    ast, diagnostics = graph_to_ast_with_diagnostics(G)
    return ast.render(), diagnostics


def graph_to_conversion_artifacts(
    G: SBNGraph,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    diagnostics = Diagnostics()
    collect_graph_diagnostics(G, diagnostics)
    semantic = graph_to_semantic_model(G, diagnostics)

    node_terms: Dict = {}
    ast = drs_for_box(G, root_box(G), node_terms, diagnostics)
    diagnostics_dicts = diagnostics.to_dicts()
    semantic["ir"]["diagnostics"] = diagnostics_dicts
    return normalize_drs(ast).render(), semantic, diagnostics_dicts


def sbn_to_ast(sbn: str, *, is_single_line: bool = False) -> DRS:
    G = SBNGraph().from_string(sbn, is_single_line=is_single_line)
    return graph_to_ast(G)


def sbn_to_ast_with_diagnostics(
    sbn: str,
    *,
    is_single_line: bool = False,
) -> Tuple[DRS, List[Dict[str, Any]]]:
    G = SBNGraph().from_string(sbn, is_single_line=is_single_line)
    return graph_to_ast_with_diagnostics(G)


def sbn_to_drs(sbn: str, *, is_single_line: bool = False) -> str:
    return sbn_to_ast(sbn, is_single_line=is_single_line).render()


def sbn_to_drs_with_diagnostics(
    sbn: str,
    *,
    is_single_line: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    ast, diagnostics = sbn_to_ast_with_diagnostics(sbn, is_single_line=is_single_line)
    return ast.render(), diagnostics


def sbn_to_semantic_model(
    sbn: str,
    *,
    is_single_line: bool = False,
) -> Dict[str, Any]:
    G = SBNGraph().from_string(sbn, is_single_line=is_single_line)
    return graph_to_semantic_model(G)


def sbn_to_semantic_ir(
    sbn: str,
    *,
    is_single_line: bool = False,
) -> SemanticDocument:
    G = SBNGraph().from_string(sbn, is_single_line=is_single_line)
    return graph_to_semantic_ir(G)


def sbn_to_fol(sbn: str, *, is_single_line: bool = False):
    drs = sbn_to_drs(sbn, is_single_line=is_single_line)
    return DrtExpression.fromstring(drs).fol()


def sbn_to_fol_with_diagnostics(
    sbn: str,
    *,
    is_single_line: bool = False,
) -> Tuple[Optional[str], str, List[Dict[str, Any]], Optional[str]]:
    drs, diagnostics = sbn_to_drs_with_diagnostics(sbn, is_single_line=is_single_line)

    try:
        fol = str(DrtExpression.fromstring(drs).fol())
        error = None
    except Exception as exc:
        fol = None
        error = f"{type(exc).__name__}: {exc}"

    return fol, drs, diagnostics, error


# =============================================================================
# CLI
# =============================================================================

def load_items(path: str, *, single_line: bool):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    loaded = json.loads(text)

    if single_line:
        if isinstance(loaded, str):
            return [{"sbn": line.strip(), "raw": line.strip()} for line in loaded.splitlines() if line.strip()]

        if isinstance(loaded, list):
            return [{"sbn": str(x).strip(), "raw": str(x).strip()} for x in loaded if str(x).strip()]

        raise TypeError("--single-line expects JSON string or JSON list of strings")

    if not isinstance(loaded, list):
        raise TypeError("Expected JSON list of objects when not using --single-line")

    out = []
    for item in loaded:
        if isinstance(item, dict):
            if "sbn" not in item:
                raise KeyError("Input object is missing required key: 'sbn'")
            out.append({
                "sbn": item["sbn"],
                "raw": item.get("raw"),
            })
        else:
            out.append({
                "sbn": str(item),
                "raw": None,
            })

    return out


def convert_item(sbn: str, *, single_line: bool) -> Dict[str, Any]:
    try:
        graph = SBNGraph().from_string(sbn, is_single_line=single_line)
        drs, semantic, diagnostics = graph_to_conversion_artifacts(graph)

        try:
            fol = str(DrtExpression.fromstring(drs).fol())
            error = None
        except Exception as exc:
            fol = None
            error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return {
            "drs": None,
            "fol": None,
            "error": f"{type(exc).__name__}: {exc}",
            "semantic": empty_semantic_model(),
            "diagnostics": [
                Diagnostic(
                    code="conversion_error",
                    severity="error",
                    message="SBN conversion failed before DRS/FOL output could be produced.",
                    details={"exception_type": type(exc).__name__},
                ).to_dict()
            ],
            "sbn": sbn,
        }

    if error is not None:
        diagnostics.append(
            Diagnostic(
                code="fol_conversion_error",
                severity="error",
                message="Generated DRS could not be converted to FOL by NLTK.",
                details={"error": error},
            ).to_dict()
        )

    return {
        "drs": drs,
        "fol": fol,
        "error": error,
        "semantic": semantic,
        "diagnostics": diagnostics,
        "sbn": sbn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/ai-developer/development/drs-to-fol/gold.json",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--single-line", default=False, action="store_true")
    parser.add_argument(
        "--pretty",
        default=False,
        action="store_true",
        help="Pretty-print JSON to stdout when --output is not provided.",
    )
    args = parser.parse_args()

    items = load_items(args.input, single_line=args.single_line)

    data = []

    for item in items:
        result = convert_item(item["sbn"], single_line=args.single_line)
        result["raw"] = item.get("raw")
        data.append(result)

    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    else:
        if args.pretty:
            print(json.dumps(data, indent=4, ensure_ascii=False))
        else:
            print(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    main()
