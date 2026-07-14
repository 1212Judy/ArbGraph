import json
import torch
import networkx as nx
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util


class EvidenceGraphBuilder:
    def __init__(
        self,
        llm,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        device: str = None,
    ):
        self.llm = llm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(embedding_model, device=self.device)

    def build(
        self,
        claims: List[Dict[str, Any]],
        query: str,
        query_relevance_threshold: float = 0.3,
        pair_similarity_threshold: float = 0.75,
        relation_confidence_threshold: float = 0.5,
        max_support_edges: int = 60,
    ) -> nx.DiGraph:
        claims = self._filter_by_query(claims, query, query_relevance_threshold)

        graph = nx.DiGraph()

        for c in claims:
            cid = c["id"]
            source_ids = c.get("source_ids", [])
            graph.add_node(
                cid,
                id=cid,
                text=c["text"],
                evidence=c.get("evidence", ""),
                source_id=c.get("source_id") or (source_ids[0] if source_ids else ""),
                source_ids=source_ids,
                source_attributions=c.get("source_attributions", []),
                members=c.get("members", []),
                doc_id=c.get("doc_id"),
                doc_title=c.get("doc_title"),
            )

        if len(claims) < 2:
            return graph

        candidate_pairs = self._select_candidate_pairs(
            claims,
            pair_similarity_threshold=pair_similarity_threshold,
        )

        support_edges = []
        contradiction_edges = []

        for ci, cj, sim_score in candidate_pairs:
            relation, relation_confidence = self._verify_relation(
                graph.nodes[ci]["text"],
                graph.nodes[cj]["text"],
            )

            if relation_confidence < relation_confidence_threshold:
                continue

            edge = (ci, cj, sim_score, relation_confidence)
            if relation == "support":
                support_edges.append(edge)
            elif relation == "contradiction":
                contradiction_edges.append(edge)

        # Paper implementation details use a global top-M support-edge cap.
        support_edges.sort(key=lambda x: x[2], reverse=True)
        support_edges = support_edges[:max_support_edges]

        for ci, cj, similarity, relation_confidence in support_edges:
            attributes = {
                "type": "support",
                "score": similarity,
                "similarity": similarity,
                "relation_confidence": relation_confidence,
            }
            graph.add_edge(ci, cj, **attributes)
            graph.add_edge(cj, ci, **attributes)

        # Preserve all verifier-approved contradiction edges.
        for ci, cj, similarity, relation_confidence in contradiction_edges:
            attributes = {
                "type": "contradiction",
                "score": similarity,
                "similarity": similarity,
                "relation_confidence": relation_confidence,
            }
            graph.add_edge(ci, cj, **attributes)
            graph.add_edge(cj, ci, **attributes)

        return graph

    def _filter_by_query(
        self,
        claims: List[Dict[str, Any]],
        query: str,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        if not claims:
            return []

        query_emb = self.encoder.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        claim_texts = [c["text"] for c in claims]
        claim_embs = self.encoder.encode(
            claim_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        sims = util.cos_sim(query_emb, claim_embs)[0]
        kept = []

        for c, sim in zip(claims, sims):
            if sim.item() >= threshold:
                kept.append(c)

        return kept

    def _select_candidate_pairs(
        self,
        claims: List[Dict[str, Any]],
        pair_similarity_threshold: float,
    ) -> List[Tuple[str, str, float]]:
        texts = [c["text"] for c in claims]
        ids = [c["id"] for c in claims]

        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        sim_matrix = util.cos_sim(embeddings, embeddings)

        pairs = []
        n = len(ids)

        for i in range(n):
            for j in range(i + 1, n):
                score = sim_matrix[i, j].item()
                if score >= pair_similarity_threshold:
                    pairs.append((ids[i], ids[j], score))

        return pairs

    def _verify_relation(self, text_a: str, text_b: str) -> Tuple[str, float]:
        prompt = f"""
Determine the logical relationship between the following two claims.

Claim A: {text_a}
Claim B: {text_b}

Choose exactly one label:
- support
- contradiction
- neutral

Confidence must be a number between 0 and 1.

Return ONLY valid JSON in the following format:
{{"label": "support", "confidence": 0.0}}
""".strip()

        raw = self.llm.generate(prompt) if hasattr(self.llm, "generate") else self.llm(prompt)
        parsed = self._parse_json(raw)

        if parsed is None:
            return "neutral", 0.0

        label = str(parsed.get("label", "neutral")).strip().lower()
        if label not in {"support", "contradiction", "neutral"}:
            return "neutral", 0.0

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return label, min(max(confidence, 0.0), 1.0)

    def _parse_json(self, text):
        try:
            text = text.content if hasattr(text, "content") else str(text)
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end < start:
                return None
            return json.loads(text[start:end + 1])
        except Exception:
            return None
