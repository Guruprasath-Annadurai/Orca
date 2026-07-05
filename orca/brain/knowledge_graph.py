"""
World Model / Knowledge Graph — per-session entity-relationship graph,
extracted from conversation via the LLM itself.

HONEST SCOPE — read before assuming this is a production knowledge graph:
  - This is a v1 foundation, not a distributed graph database. No entity
    resolution across sessions (the same "Alice" mentioned in two different
    sessions is two different graph nodes — there's no cross-session
    identity linking). No temporal fact versioning (if a fact changes, the
    old and new versions both sit in the graph as separate edges, not
    reconciled). No conflict detection between contradictory relationships.
  - Extraction is LLM-prompted triple extraction (subject, predicate,
    object) — heuristic quality bounded by the underlying model's
    reasoning, not a specialized NER/relation-extraction model. Same
    honesty posture as every other LLM-as-judge mechanism in this project
    (orca/train/redteam.py, orca/train/persona_eval.py): a floor, useful
    signal, not ground truth.
  - Scoped per-session, not global across all users — this is a deliberate
    privacy choice, not a limitation to fix later. A shared cross-user
    graph would need much more careful privacy/consent design (whose facts
    are these, who can see them, how does right-to-delete cascade through
    shared nodes) that this project's "local-first, session-scoped, user
    controls their own data" model doesn't currently support. Keeping it
    session-scoped means it plugs directly into the existing right-to-delete
    orchestration (orca/serve/account_delete.py) with the same
    clear()-on-delete pattern as DocStore and EpisodicMemory.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING

from orca.config import ORCA_HOME

if TYPE_CHECKING:
    from orca.brain.providers import OrcaBrain

KNOWLEDGE_DIR = ORCA_HOME / "knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

_EXTRACTION_PROMPT = """\
Extract factual relationships from this text as (subject, predicate, object) triples.
Only extract clear, specific facts — not opinions, hypotheticals, or vague statements.
Include an entity type for the subject and object: person, organization, technology,
event, place, or concept.

Text:
{text}

Reply with ONLY a JSON array of objects in this exact shape, or [] if no clear facts:
[{{"subject": "...", "subject_type": "...", "predicate": "...", "object": "...", "object_type": "..."}}]"""


@dataclass
class Entity:
    name: str
    entity_type: str = "concept"
    first_seen: float = field(default_factory=time.time)
    mention_count: int = 1


@dataclass
class Relationship:
    subject: str
    predicate: str
    object: str
    source: str = ""          # e.g. "chat" or a doc filename
    created_at: float = field(default_factory=time.time)


def _normalize(name: str) -> str:
    """Lowercase, whitespace-collapsed key for dedup — 'Alice Smith' and 'alice smith' are the same node."""
    return re.sub(r"\s+", " ", name.strip().lower())


class KnowledgeGraph:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._path = KNOWLEDGE_DIR / f"{session_id}.json"
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._entities = {k: Entity(**v) for k, v in data.get("entities", {}).items()}
            self._relationships = [Relationship(**r) for r in data.get("relationships", [])]
        except Exception:
            pass  # corrupt/missing file — start fresh rather than crash the session

    def _save(self) -> None:
        try:
            data = {
                "entities": {k: asdict(v) for k, v in self._entities.items()},
                "relationships": [asdict(r) for r in self._relationships],
            }
            self._path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass  # persistence failure shouldn't crash the conversation

    def _touch_entity(self, name: str, entity_type: str) -> None:
        key = _normalize(name)
        if key in self._entities:
            self._entities[key].mention_count += 1
        else:
            self._entities[key] = Entity(name=name, entity_type=entity_type)

    def _add_relationship(self, subject: str, predicate: str, obj: str,
                          subject_type: str, object_type: str, source: str) -> Relationship:
        self._touch_entity(subject, subject_type)
        self._touch_entity(obj, object_type)

        # Dedup: same (subject, predicate, object) triple already recorded — don't
        # pile up duplicates every time the same fact gets mentioned again.
        s_key, o_key = _normalize(subject), _normalize(obj)
        for r in self._relationships:
            if _normalize(r.subject) == s_key and _normalize(r.object) == o_key and r.predicate.lower() == predicate.lower():
                return r

        rel = Relationship(subject=subject, predicate=predicate, object=obj, source=source)
        self._relationships.append(rel)
        return rel

    def extract_and_add(self, text: str, source: str, brain: "OrcaBrain") -> list[Relationship]:
        """
        Prompts the LLM to extract (subject, predicate, object) triples from
        text and adds them to the graph. Returns the relationships added
        (including ones that already existed — dedup happens internally).
        Never raises — extraction failure just means nothing gets added
        this turn, not a broken conversation.
        """
        try:
            raw = brain.complete(
                [{"role": "user", "content": _EXTRACTION_PROMPT.format(text=text[:3000])}],
                temperature=0.1, max_tokens=500,
            )
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            triples = json.loads(match.group())
        except Exception:
            return []

        added = []
        for t in triples:
            if not isinstance(t, dict):
                continue
            subj, pred, obj = t.get("subject"), t.get("predicate"), t.get("object")
            if not (subj and pred and obj):
                continue
            rel = self._add_relationship(
                subj, pred, obj,
                t.get("subject_type", "concept"), t.get("object_type", "concept"),
                source,
            )
            added.append(rel)

        if added:
            self._save()
        return added

    def query_entity(self, name: str) -> dict | None:
        """Returns the entity plus every relationship it appears in, as subject or object."""
        key = _normalize(name)
        entity = self._entities.get(key)
        if not entity:
            return None

        as_subject = [asdict(r) for r in self._relationships if _normalize(r.subject) == key]
        as_object = [asdict(r) for r in self._relationships if _normalize(r.object) == key]

        return {
            "entity": asdict(entity),
            "relationships_as_subject": as_subject,
            "relationships_as_object": as_object,
        }

    def neighbors(self, name: str) -> list[dict]:
        """Directly connected entities — one hop out, either direction."""
        info = self.query_entity(name)
        if not info:
            return []
        seen = set()
        result = []
        for r in info["relationships_as_subject"]:
            key = _normalize(r["object"])
            if key not in seen:
                seen.add(key)
                result.append({"name": r["object"], "via": r["predicate"], "direction": "outgoing"})
        for r in info["relationships_as_object"]:
            key = _normalize(r["subject"])
            if key not in seen:
                seen.add(key)
                result.append({"name": r["subject"], "via": r["predicate"], "direction": "incoming"})
        return result

    def all_entities(self) -> list[dict]:
        return [asdict(e) for e in self._entities.values()]

    def count(self) -> dict:
        return {"entities": len(self._entities), "relationships": len(self._relationships)}

    def clear(self) -> None:
        """For right-to-delete — matches the same clear() pattern DocStore already uses."""
        self._entities = {}
        self._relationships = []
        if self._path.exists():
            self._path.unlink()
