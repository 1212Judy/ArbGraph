# ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form RAG

This repository provides an implementation of **ArbGraph**, a framework for improving the reliability of long-form retrieval-augmented generation (RAG) via pre-generation evidence arbitration.

The code accompanies our paper and is released for research use and partial reproducibility.

---

## Overview

ArbGraph addresses a key limitation of long-form RAG systems: handling noisy and contradictory evidence.

Instead of resolving conflicts during generation, ArbGraph performs **pre-generation arbitration** by:

- decomposing documents into atomic claims,
- modeling support and contradiction relations,
- estimating claim credibility via conflict-aware arbitration,
- generating outputs from a validated evidence set.

---

## Pipeline

1. Retrieval  
2. Atomic Claim Extraction (`atomization.py`)  
3. Claim Alignment (`claim_alignment.py`)  
4. Evidence Graph Construction (`evidence_graph.py`)  
5. Conflict Arbitration (`conflict_arbitration.py`)  
6. Generation (`longform_generation.py`)

---

## Usage

```bash
python run_arbgraph.py
