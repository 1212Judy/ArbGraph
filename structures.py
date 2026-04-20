from dataclasses import dataclass, field, asdict
from typing import List, Dict
from enum import Enum


# ============================
# Edge Types (Paper-aligned)
# ============================

class EdgeType(Enum):
    SUPPORT = "support"
    CONTRADICTION = "contradiction"


# ============================
# Atomic Claim
# ============================

@dataclass
class AtomicClaim:
    """
    Atomic factual claim (node in the evidence graph).
    """
    id: str
    text: str
    evidence: str = ""
    is_meta: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================
# Evidence Graph Edge
# ============================

@dataclass
class EvidenceEdge:
    """
    Logical relation between two atomic claims.
    """
    source: str        # source claim id
    target: str        # target claim id
    edge_type: EdgeType
    explanation: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["edge_type"] = self.edge_type.value
        return d


# ============================
# Evidence Graph
# ============================

@dataclass
class EvidenceGraph:
    """
    Evidence graph G = (V, E), where:
    - V are atomic claims
    - E are support / contradiction relations
    """
    nodes: Dict[str, AtomicClaim] = field(default_factory=dict)
    edges: List[EvidenceEdge] = field(default_factory=list)

    def add_node(self, claim: AtomicClaim):
        self.nodes[claim.id] = claim

    def add_edge(self, edge: EvidenceEdge):
        # Optional safety check (can be removed if you want it minimal)
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Edge refers to non-existent claim id.")
        self.edges.append(edge)

    def to_dict(self) -> Dict:
        return {
            "nodes": [c.to_dict() for c in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }
