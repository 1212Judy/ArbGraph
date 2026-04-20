import json
import re
import uuid
from typing import List, Dict, Any, Optional


class ClaimExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(
        self,
        text: str,
        source_id: str,
        doc_id: Optional[str] = None,
        doc_title: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []

        prompt = f"""
Decompose the following text into a list of atomic factual claims.

Requirements:
- Each claim must express exactly one factual proposition.
- Each claim must be minimal yet self-contained.
- Each claim must be independently verifiable from the text.
- Do not merge multiple facts into one claim.
- Keep wording faithful to the source text.
- Do not include speculation, opinions, or background commentary.

Text:
\"\"\"{text}\"\"\"

Return ONLY a JSON list in the following format:
[
  {{
    "claim": "atomic factual statement",
    "evidence": "supporting span from the text"
  }}
]
""".strip()

        raw = self.llm.generate(prompt) if hasattr(self.llm, "generate") else self.llm(prompt)
        parsed = self._parse_json(str(raw))

        if not parsed:
            return []

        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue

            claim_text = str(item.get("claim", "")).strip()
            if not claim_text:
                continue

            evidence = str(item.get("evidence", "")).strip()

            claim = {
                "id": self._make_claim_id(source_id, idx),
                "text": claim_text,
                "evidence": evidence,
                "source_id": source_id,
            }

            if doc_id is not None:
                claim["doc_id"] = doc_id
            if doc_title is not None:
                claim["doc_title"] = doc_title

            claims.append(claim)

        return claims

    def _make_claim_id(self, source_id: str, idx: int) -> str:
        safe_source = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(source_id))
        return f"claim_{safe_source}_{idx}_{uuid.uuid4().hex[:8]}"

    def _parse_json(self, text: str) -> List[Dict[str, Any]]:
        try:
            cleaned = re.sub(r"```json|```", "", text).strip()
            match = re.search(r"\[.*\]", cleaned, re.S)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(cleaned)

            return data if isinstance(data, list) else []
        except Exception:
            return []