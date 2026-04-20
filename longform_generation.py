import re
from typing import Dict, Any, List


class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, query: str, context: Dict[str, Any]) -> str:
        validated_claims: List[Dict[str, Any]] = context.get("validated_claims", [])

        evidence_blocks = []

        for i, item in enumerate(validated_claims, 1):
            text = (item.get("text") or "").strip()
            evidence = (item.get("evidence") or "").strip()

            if not text:
                continue

            block = f"[Claim {i}]\n{text}"
            if evidence:
                block += f"\nEvidence: {evidence}"

            evidence_blocks.append(block)

        if evidence_blocks:
            evidence_text = "\n\n".join(evidence_blocks)
        else:
            evidence_text = "No validated evidence is available."

        prompt = f"""
Answer the following question using only the validated evidence provided below.

Question:
{query}

Validated Evidence:
{evidence_text}

Requirements:
- Base the answer strictly on the evidence.
- Maintain a neutral, academic tone.
- If the evidence is insufficient or conflicting, state this explicitly.
- Do not introduce unsupported facts.
""".strip()

        if not hasattr(self.llm, "generate"):
            return "Error: LLM interface does not support generation."

        raw = self.llm.generate(
            prompt,
            max_new_tokens=512,
            temperature=0.2,
        )

        response = raw[0] if isinstance(raw, (list, tuple)) else raw
        if hasattr(response, "content"):
            response = response.content

        response = str(response)
        response = response.replace("</s>", "").strip()
        response = re.sub(r"\n{3,}", "\n\n", response)

        return response