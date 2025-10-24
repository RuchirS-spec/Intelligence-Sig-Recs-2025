# 🧠 Graph2Seq: Leveraging Dependency Graphs for Deep Semantic Understanding

## 🚀 Overview
This project implements a **Graph-to-Sequence (Graph2Seq)** model that uses **dependency graphs** to encode linguistic and relational structure before generating output sequences.  
By integrating graph representations with pretrained transformers, the model captures **deep contextual dependencies** that traditional sequence-based models often miss.

---

## 🌐 Motivation
Natural language is **not purely sequential** — words depend on each other in complex hierarchical ways.  
While recurrent and transformer-based architectures excel at sequential context modeling, they often overlook **explicit syntactic and dependency relations** between tokens.

Dependency graphs bridge this gap by:
- Exposing **true linguistic structure** through head–modifier relationships.
- Enabling **graph message passing** to propagate meaning across related tokens.
- Creating a **richer semantic space** for decoding coherent and context-aware sequences.

---

## 🧩 Methodology

### 1. Dependency Graph Construction
Each input sentence is parsed into a **dependency graph**, where:
- **Nodes** represent tokens.
- **Edges** represent syntactic dependencies (e.g., *subject*, *object*, *modifier*).

This structure encodes both **lexical semantics** and **syntactic roles**, forming the foundation for graph-based reasoning.

### 2. Graph Encoding
A **Graph Neural Network (GNN)** performs message passing across nodes:
```math
h_i^{(t+1)} = \sigma\left( W_1 h_i^{(t)} + \sum_{j \in \mathcal{N}(i)} W_2 h_j^{(t)} + b \right)
