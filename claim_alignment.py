from typing import List, Dict, Any, Set
import re
import numpy as np
from sentence_transformers import SentenceTransformer


class NodeAligner:
    # semantic alignment + canonical clustering
    def __init__(
        self,
        llm_engine=None,
        encoder_name: str = "BAAI/bge-large-en-v1.5",
        similarity_threshold: float = 0.88,
    ):
        self.llm = llm_engine
        self.encoder = SentenceTransformer(encoder_name)
        self.similarity_threshold = similarity_threshold

    def align_and_merge(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not claims:
            return []

        valid_claims = [c for c in claims if isinstance(c, dict) and c.get("text", "").strip()]
        if not valid_claims:
            return []

        if len(valid_claims) == 1:
            return [self._make_canonical_node(valid_claims)]

        texts = [c["text"].strip() for c in valid_claims]

        embeddings = self.encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        similarity_matrix = np.matmul(embeddings, embeddings.T)

        visited = [False] * len(valid_claims)
        canonical_nodes: List[Dict[str, Any]] = []

        for i in range(len(valid_claims)):
            if visited[i]:
                continue

            cluster_indices = [i]
            visited[i] = True

            for j in range(i + 1, len(valid_claims)):
                if visited[j]:
                    continue

                sim = float(similarity_matrix[i, j])
                if sim < self.similarity_threshold:
                    continue

                if not self._is_mergeable(valid_claims[i], valid_claims[j]):
                    continue

                cluster_indices.append(j)
                visited[j] = True

            cluster_claims = [valid_claims[idx] for idx in cluster_indices]
            canonical_nodes.append(self._make_canonical_node(cluster_claims))

        return canonical_nodes

    def _is_mergeable(self, claim_a: Dict[str, Any], claim_b: Dict[str, Any]) -> bool:
        text_a = claim_a.get("text", "")
        text_b = claim_b.get("text", "")

        ents_a = self._extract_key_terms(text_a)
        ents_b = self._extract_key_terms(text_b)

        nums_a = self._extract_numbers(text_a)
        nums_b = self._extract_numbers(text_b)

        years_a = self._extract_years(text_a)
        years_b = self._extract_years(text_b)

        if ents_a and ents_b and not (ents_a & ents_b):
            return False

        if years_a and years_b and years_a != years_b:
            return False

        if nums_a and nums_b and nums_a != nums_b:
            return False

        return True

    def _make_canonical_node(self, cluster_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        representative = self._select_representative_claim(cluster_claims)

        source_ids: List[str] = []
        seen_sources: Set[str] = set()

        for c in cluster_claims:
            sid = c.get("source_id")
            if sid is not None:
                sid = str(sid)
                if sid not in seen_sources:
                    seen_sources.add(sid)
                    source_ids.append(sid)

        merged_evidence = self._merge_evidence(cluster_claims)

        canonical_id = representative.get("id")
        if canonical_id is None:
            canonical_id = f"canon_{abs(hash(representative.get('text', '')))}"

        node = {
            "id": canonical_id,
            "text": representative.get("text", "").strip(),
            "source_ids": source_ids,
            "source_attributions": source_ids.copy(),
            "members": cluster_claims,
            "evidence": merged_evidence,
            "is_canonical": True,
        }

        for key in ["doc_id", "doc_title"]:
            if key in representative:
                node[key] = representative[key]

        return node

    def _select_representative_claim(self, cluster_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        def score(claim: Dict[str, Any]) -> tuple:
            text = claim.get("text", "").strip()
            evidence = claim.get("evidence", "").strip()
            return (len(text), len(evidence))

        return max(cluster_claims, key=score)

    def _merge_evidence(self, cluster_claims: List[Dict[str, Any]]) -> str:
        evidences = []
        seen = set()

        for c in cluster_claims:
            ev = c.get("evidence", "")
            if not isinstance(ev, str):
                continue
            ev = ev.strip()
            if ev and ev not in seen:
                seen.add(ev)
                evidences.append(ev)

        return "\n".join(evidences)

    def _extract_key_terms(self, text: str) -> Set[str]:
        text = text.strip()
        if not text:
            return set()

        terms = set()

        for tok in re.findall(r"\b[A-Z][a-zA-Z0-9\-]+\b", text):
            if len(tok) >= 3:
                terms.add(tok.lower())

        for tok in re.findall(r"\b[a-zA-Z]{3,}[a-zA-Z0-9\-]*\b", text):
            if len(tok) >= 5:
                terms.add(tok.lower())

        stop = {
            "which", "where", "there", "their", "about", "would", "could",
            "should", "after", "before", "because", "while", "under",
            "using", "based", "through", "these", "those", "being", "other"
        }
        return {t for t in terms if t not in stop}

    def _extract_numbers(self, text: str) -> Set[str]:
        return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))

    def _extract_years(self, text: str) -> Set[str]:
        return set(re.findall(r"\b(?:18|19|20)\d{2}\b", text))
