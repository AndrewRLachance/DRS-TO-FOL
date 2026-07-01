from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


def _drop_none(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class SemanticReferent:
    id: str
    term: str
    source_id: str
    source_token: str
    kind: str
    box_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "id": self.id,
                "term": self.term,
                "source_id": self.source_id,
                "source_token": self.source_token,
                "kind": self.kind,
                "box_id": self.box_id,
            }
        )


@dataclass(frozen=True)
class SemanticCondition:
    id: str
    label: str
    kind: str
    source_id: Optional[str] = None
    source_term: Optional[str] = None
    target_id: Optional[str] = None
    target_term: Optional[str] = None
    value: Optional[str] = None
    box_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "id": self.id,
                "label": self.label,
                "kind": self.kind,
                "source_id": self.source_id,
                "source_term": self.source_term,
                "target_id": self.target_id,
                "target_term": self.target_term,
                "value": self.value,
                "box_id": self.box_id,
            }
        )


@dataclass(frozen=True)
class SemanticBox:
    id: str
    parent_id: Optional[str]
    operator: str
    scope_marker: Optional[str] = None
    local_referents: Tuple[str, ...] = ()
    child_boxes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "id": self.id,
                "parent_id": self.parent_id,
                "operator": self.operator,
                "scope_marker": self.scope_marker,
                "local_referents": list(self.local_referents),
                "child_boxes": list(self.child_boxes),
            }
        )


@dataclass(frozen=True)
class EntityFrame:
    id: str
    referent_id: str
    term: str
    synset: str
    box_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "type": "entity",
                "id": self.id,
                "referent_id": self.referent_id,
                "term": self.term,
                "synset": self.synset,
                "box_id": self.box_id,
            }
        )


@dataclass(frozen=True)
class EventFrame:
    id: str
    event_id: str
    term: str
    synset: str
    roles: Tuple[Dict[str, str], ...] = ()
    box_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "type": "event",
                "id": self.id,
                "event_id": self.event_id,
                "term": self.term,
                "synset": self.synset,
                "roles": [dict(role) for role in self.roles],
                "box_id": self.box_id,
            }
        )


@dataclass(frozen=True)
class TimeFrame:
    id: str
    time_id: Optional[str]
    term: Optional[str]
    relation: str
    value: Optional[str] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "type": "time",
                "id": self.id,
                "time_id": self.time_id,
                "term": self.term,
                "relation": self.relation,
                "value": self.value,
                "source_id": self.source_id,
                "target_id": self.target_id,
            }
        )


@dataclass(frozen=True)
class NameFrame:
    id: str
    referent_id: str
    term: str
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "name",
            "id": self.id,
            "referent_id": self.referent_id,
            "term": self.term,
            "name": self.name,
        }


@dataclass(frozen=True)
class QuantityFrame:
    id: str
    source_id: str
    source_term: str
    role: str
    value: Optional[str] = None
    unit: Optional[str] = None
    measure: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "type": "quantity",
                "id": self.id,
                "source_id": self.source_id,
                "source_term": self.source_term,
                "role": self.role,
                "value": self.value,
                "unit": self.unit,
                "measure": self.measure,
            }
        )


@dataclass(frozen=True)
class EqualityFrame:
    id: str
    operator: str
    left_id: str
    left_term: str
    right_id: Optional[str] = None
    right_term: Optional[str] = None
    value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "type": "equality",
                "id": self.id,
                "operator": self.operator,
                "left_id": self.left_id,
                "left_term": self.left_term,
                "right_id": self.right_id,
                "right_term": self.right_term,
                "value": self.value,
            }
        )


@dataclass(frozen=True)
class GeneralizedQuantifierFrame:
    id: str
    source_id: str
    quantifier: str
    status: str
    details: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "type": "generalized_quantifier",
            "id": self.id,
            "source_id": self.source_id,
            "quantifier": self.quantifier,
            "status": self.status,
        }
        if self.details:
            out["details"] = dict(self.details)
        return out


@dataclass(frozen=True)
class AccessibilityFrame:
    id: str
    parent_box_id: str
    child_box_id: str
    operator: str
    scope_marker: Optional[str] = None
    local_referents: Tuple[str, ...] = ()
    inherited_referents: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return _drop_none(
            {
                "type": "accessibility",
                "id": self.id,
                "parent_box_id": self.parent_box_id,
                "child_box_id": self.child_box_id,
                "operator": self.operator,
                "scope_marker": self.scope_marker,
                "local_referents": list(self.local_referents),
                "inherited_referents": list(self.inherited_referents),
            }
        )


@dataclass(frozen=True)
class PropositionTarget:
    box_id: str
    operator: str
    drs: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "box_id": self.box_id,
            "operator": self.operator,
            "drs": self.drs,
        }


@dataclass(frozen=True)
class PropositionFrame:
    source_id: str
    source_term: str
    source_token: str
    role: str
    scope_marker: str
    lowered_to_fol: bool
    target_boxes: Tuple[PropositionTarget, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "proposition_scope",
            "source_id": self.source_id,
            "source_term": self.source_term,
            "source_token": self.source_token,
            "role": self.role,
            "scope_marker": self.scope_marker,
            "lowered_to_fol": self.lowered_to_fol,
            "target_boxes": [target.to_dict() for target in self.target_boxes],
        }


@dataclass(frozen=True)
class ModalFrame:
    operator: str
    source_box_id: str
    target_box_id: str
    scope_marker: Optional[str]
    drs: str
    lowered_to_fol: bool
    fol_handling: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "modal_scope",
            "operator": self.operator,
            "source_box_id": self.source_box_id,
            "target_box_id": self.target_box_id,
            "scope_marker": self.scope_marker,
            "drs": self.drs,
            "lowered_to_fol": self.lowered_to_fol,
            "fol_handling": self.fol_handling,
        }


@dataclass(frozen=True)
class DiscourseFrame:
    relation: str
    source_box_id: str
    target_box_id: str
    scope_marker: Optional[str]
    drs: str
    lowered_to_fol: bool
    fol_handling: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "discourse_relation",
            "relation": self.relation,
            "source_box_id": self.source_box_id,
            "target_box_id": self.target_box_id,
            "scope_marker": self.scope_marker,
            "drs": self.drs,
            "lowered_to_fol": self.lowered_to_fol,
            "fol_handling": self.fol_handling,
        }


@dataclass(frozen=True)
class SemanticDocument:
    boxes: Tuple[SemanticBox, ...] = ()
    referents: Tuple[SemanticReferent, ...] = ()
    conditions: Tuple[SemanticCondition, ...] = ()
    frames: Tuple[Any, ...] = ()
    diagnostics: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        frames = [frame.to_dict() for frame in self.frames]
        return {
            "ir": {
                "boxes": [box.to_dict() for box in self.boxes],
                "referents": [referent.to_dict() for referent in self.referents],
                "conditions": [condition.to_dict() for condition in self.conditions],
                "frames": frames,
                "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
            },
            "proposition_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, PropositionFrame)
            ],
            "modal_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, ModalFrame)
            ],
            "discourse_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, DiscourseFrame)
            ],
            "entity_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, EntityFrame)
            ],
            "event_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, EventFrame)
            ],
            "time_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, TimeFrame)
            ],
            "name_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, NameFrame)
            ],
            "quantity_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, QuantityFrame)
            ],
            "equality_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, EqualityFrame)
            ],
            "generalized_quantifier_frames": [
                frame.to_dict()
                for frame in self.frames
                if isinstance(frame, GeneralizedQuantifierFrame)
            ],
            "accessibility_frames": [
                frame.to_dict() for frame in self.frames if isinstance(frame, AccessibilityFrame)
            ],
        }

    @classmethod
    def empty(cls) -> "SemanticDocument":
        return cls()
