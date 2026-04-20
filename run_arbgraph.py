import os
import json
from typing import Dict

from local_llm.hf_adapter import HFAdapter
from atomization import ClaimExtractor
from retrieval.retriever import WikipediaRetriever

from claim_alignment import NodeAligner
from evidence_graph import EvidenceGraphBuilder
from conflict_arbitration import ArbGraphArbitrator
from longform_generation import AnswerGenerator


MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")


def extract_claims_from_documents(extractor, documents):
    claims = []

    for doc in documents:
        text = doc.text
        chunk_size = 2500
        step = chunk_size - 200
        chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), step)
        ]

        for chunk_idx, chunk in enumerate(chunks):
            chunk_claims = extractor.extract(
                text=chunk,
                source_id=doc.id,
                doc_id=doc.id,
                doc_title=getattr(doc, "title", None),
            )
            claims.extend(chunk_claims)

    return claims


def process_single_question(
    question_data: Dict,
    components: Dict,
    output_file: str
):
    retriever = components["retriever"]
    extractor = components["extractor"]
    aligner = components["aligner"]
    graph_builder = components["graph_builder"]
    arbitrator = components["arbitrator"]
    generator = components["generator"]

    query = question_data["Question"]
    qid = question_data["id"]

    result = {
        "id": qid,
        "question": query,
        "documents": [],
        "arbgraph_meta": {
            "rounds": 0,
            "validated_claims": 0,
            "suppressed_claims": 0,
            "retrieval_status": "failed",
            "status": "init",
            "error_stage": None,
        },
        "answer": ""
    }

    try:
        docs = retriever.retrieve(query, max_docs=5)
        if not docs:
            result["arbgraph_meta"]["status"] = "no_documents"
        else:
            result["arbgraph_meta"]["retrieval_status"] = "success"

            for idx, doc in enumerate(docs, 1):
                result["documents"].append({
                    "id": idx,
                    "title": getattr(doc, "title", ""),
                    "url": getattr(doc, "url", ""),
                    "source": "Wikipedia",
                    "context": getattr(doc, "text", "")[:3000],
                })

            raw_claims = extract_claims_from_documents(extractor, docs)
            if not raw_claims:
                result["arbgraph_meta"]["status"] = "no_raw_claims"
                result["arbgraph_meta"]["error_stage"] = "claim_extraction"
            else:
                canonical_claims = aligner.align_and_merge(raw_claims)
                if not canonical_claims:
                    result["arbgraph_meta"]["status"] = "no_canonical_claims"
                    result["arbgraph_meta"]["error_stage"] = "claim_alignment"
                else:
                    graph = graph_builder.build(canonical_claims, query=query)

                    arbitration_result = arbitrator.arbitrate(graph, query)

                    validated_claims = arbitration_result.get("validated_claims", [])
                    suppressed_claims = arbitration_result.get("suppressed_claims", [])

                    result["arbgraph_meta"]["validated_claims"] = len(validated_claims)
                    result["arbgraph_meta"]["suppressed_claims"] = len(suppressed_claims)

                    history = arbitration_result.get("arbitration_history", [])
                    result["arbgraph_meta"]["rounds"] = len(history)

                    result["answer"] = generator.generate(
                        query=query,
                        context={"validated_claims": validated_claims},
                    )
                    result["arbgraph_meta"]["status"] = "success"

    except Exception as e:
        result["arbgraph_meta"]["status"] = "exception"
        result["arbgraph_meta"]["error_stage"] = "runtime"
        result["answer"] = f"Error: {str(e)}"

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def load_checkpoint(output_file: str) -> set:
    processed = set()

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["id"])
                except Exception:
                    continue

    return processed


def main():
    input_file = "data/longfact/longfact_test.json"
    output_file = "outputs/longfact/arbgraph_predictions.jsonl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    processed_ids = load_checkpoint(output_file)

    llm = HFAdapter(MODEL_NAME)

    retriever = WikipediaRetriever()
    extractor = ClaimExtractor(llm)
    aligner = NodeAligner(llm)

    graph_builder = EvidenceGraphBuilder(llm)
    arbitrator = ArbGraphArbitrator(llm=llm, max_rounds=2)
    generator = AnswerGenerator(llm)

    components = {
        "retriever": retriever,
        "extractor": extractor,
        "aligner": aligner,
        "graph_builder": graph_builder,
        "arbitrator": arbitrator,
        "generator": generator,
    }

    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        if q["id"] in processed_ids:
            continue

        process_single_question(q, components, output_file)

    print("[Done] All questions processed.")


if __name__ == "__main__":
    main()