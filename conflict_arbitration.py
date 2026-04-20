import json
import math
import re
import networkx as nx
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class ArbitrationRound:
    round_id: int
    resolved_pairs: List[Tuple[str, str]]
    decisions: Dict[str, Any]
    credibility_snapshot: Dict[str, float]


class ArbGraphArbitrator:
    # iterative credibility arbitration (Algorithm 1)

    def __init__(
        self,
        llm,
        max_rounds: int = 3,
        accept_threshold: float = 0.3,
        arbitration_budget: int = 3,
        step_size: float = 0.8,
    ):
        self.llm = llm
        self.max_rounds = max_rounds
        self.accept_threshold = accept_threshold
        self.arbitration_budget = arbitration_budget
        self.eta = step_size

        self.logits: Dict[str, float] = {}
        self.defeat_counts: Dict[str, int] = {}
        self.history: List[ArbitrationRound] = []

    # ============================

    def arbitrate(self, graph: nx.DiGraph, query: str) -> Dict[str, Any]:
        self._initialize(graph)

        for t in range(1, self.max_rounds + 1):
            pairs = self._select_conflicts(graph, t)
            if not pairs:
                break

            round_record = ArbitrationRound(
                round_id=t,
                resolved_pairs=[],
                decisions={},
                credibility_snapshot=self._snapshot(),
            )

            for ci, cj in pairs:
                result = self._resolve_pair(graph, ci, cj, query)
                self._update(result)

                round_record.resolved_pairs.append((ci, cj))
                round_record.decisions[f"{ci}||{cj}"] = result

            self.history.append(round_record)

        return self._finalize(graph, query)

    # ============================

    def _initialize(self, graph: nx.DiGraph):
        self.logits = {cid: 0.0 for cid in graph.nodes}
        self.defeat_counts = {cid: 0 for cid in graph.nodes}
        self.history = []

    # ============================

    def _select_conflicts(self, graph: nx.DiGraph, round_id: int):
        candidates = []
        seen = set()

        for u, v, d in graph.edges(data=True):
            if d.get("type") != "contradiction":
                continue

            # avoid duplicate (A,B) / (B,A)
            pair = tuple(sorted([u, v]))
            if pair in seen:
                continue
            seen.add(pair)

            pu, pv = self._prob(u), self._prob(v)
            if pu < self.accept_threshold or pv < self.accept_threshold:
                continue

            intensity = (pu + pv) / (1.0 + abs(pu - pv))
            candidates.append((u, v, intensity))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(u, v) for u, v, _ in candidates[: self.arbitration_budget]]

    # ============================

    def _resolve_pair(self, graph, ci, cj, query):
        node_i, node_j = graph.nodes[ci], graph.nodes[cj]

        # support context
        ctx_i = self._get_support_context(graph, ci)
        ctx_j = self._get_support_context(graph, cj)

        prompt = f"""
Query: {query}

Claim A: {node_i.get("text")}
Supporting evidence A:
{ctx_i}

Claim B: {node_j.get("text")}
Supporting evidence B:
{ctx_j}

Choose one:
A_entails_B, B_entails_A, A_contradicts_B, B_contradicts_A, unknown

Return JSON:
{{"label": "..."}}
"""

        raw = self.llm.generate(prompt)
        parsed = self._parse_json(raw) or {}

        label = parsed.get("label")

        gate = 1 if label in {
            "A_entails_B", "A_contradicts_B",
            "B_entails_A", "B_contradicts_A"
        } else 0

        if label in {"A_entails_B", "A_contradicts_B"}:
            winner, loser = ci, cj
        elif label in {"B_entails_A", "B_contradicts_A"}:
            winner, loser = cj, ci
        else:
            winner, loser = (ci, cj) if self._prob(ci) >= self._prob(cj) else (cj, ci)

        return {
            "winner": winner,
            "loser": loser,
            "gate": gate,
            "label": label,
        }

    # ============================

    def _get_support_context(self, graph, cid):
        texts = []
        for u, v, d in graph.edges(cid, data=True):
            if d.get("type") == "support":
                texts.append(graph.nodes[v].get("text", ""))
        return "\n".join(texts[:5])

    # ============================

    def _update(self, result):
        if result["gate"] != 1:
            return

        win, lose = result["winner"], result["loser"]
        self.logits[win] += self.eta
        self.logits[lose] -= self.eta
        self.defeat_counts[lose] += 1

    # ============================

    def _finalize(self, graph, query):
        accepted, suppressed = [], []

        for cid in graph.nodes:
            p = self._prob(cid)
            node = graph.nodes[cid]

            item = {
                "id": cid,
                "text": node.get("text"),
                "confidence": round(p, 3),
                "evidence": node.get("evidence", ""),
                "source_id": node.get("source_id"),
                "doc_id": node.get("doc_id"),
                "doc_title": node.get("doc_title"),
            }

            if p >= self.accept_threshold:
                accepted.append(item)
            else:
                suppressed.append(item)

        return {
            "query": query,
            "validated_claims": accepted,
            "suppressed_claims": suppressed,
            "arbitration_history": [r.__dict__ for r in self.history],
        }

    # ============================

    def _prob(self, cid):
        return 1 / (1 + math.exp(-self.logits[cid]))

    def _snapshot(self):
        return {k: round(self._prob(k), 3) for k in self.logits}

    def _parse_json(self, text):
        m = re.search(r"\{.*\}", str(text), re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None