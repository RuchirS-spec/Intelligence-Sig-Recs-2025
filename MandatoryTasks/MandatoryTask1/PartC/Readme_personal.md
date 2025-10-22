# 🧠 Hypergraph-Based Question Answering Model

This repository implements a **Hypergraph Convolutional Network (HyperGCN)** for the **NewsQA** dataset.  
Unlike traditional graph-based models that capture only pairwise relationships between words, this model leverages **hypergraphs** to represent higher-order semantic structures in language.

---

## 📘 1. Why Hypergraphs for This Task?

Traditional graphs in NLP capture **pairwise relationships** between tokens.  
However, language is inherently **multi-relational** — words interact in groups, not just pairs.  
Hypergraphs provide a natural way to model these **higher-order connections**.

### 🔹 Key Advantages

- **Group-Level Semantics**  
  A hyperedge can connect multiple nodes (tokens) at once, enabling the model to treat phrases or entire questions as single semantic units.

- **Superior Context Modeling**  
  Sliding window–based hyperedges capture **local phrasal context** more effectively than word-to-word connections.

- **Enhanced Message Passing**  
  Through hypergraph convolutions, information flows between **individual tokens** and **entire token groups**, producing richer, more context-aware embeddings.

---

## 🏗️ 2. Model Architecture

The model follows a multi-stage pipeline from text encoding to answer span prediction.

### 🔸 Steps Overview

1. **Hypergraph Construction**
   - **Nodes:** Tokens from the passage and question.
   - **Hyperedges:**
     - **Sliding Window:** A moving window of size `3` groups local words into hyperedges.
     - **Question Edge:** All tokens in the question are connected via one hyperedge for global reasoning.

2. **BERT Encoder**
   - A pre-trained `bert-base-uncased` model generates contextualized token embeddings.
   - These embeddings serve as **initial node features** for the hypergraph.

3. **HyperGCN Layers**
   - Four stacked **Hypergraph Convolutional Network (HyperGCN)** layers refine embeddings via message passing through hyperedges.

4. **Prediction Head**
   - Two linear layers output logits for the **start** and **end** positions of the answer span.

---

## 🧩 3. The HyperGCN Convolution Layer Explained

Each HyperGCN layer updates node features using a **two-step message passing** process:

1. **Nodes → Hyperedges (aggregate)**  
2. **Hyperedges → Nodes (distribute)**

### 🧮 Mathematical Formulation

\[
X^{(l+1)} = \sigma \left( D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{(l)} \Theta \right)
\]

**Where:**

| Symbol | Meaning |
|:--------|:---------|
| \( X^{(l)} \) | Node feature matrix at layer *l* |
| \( H \) | Hypergraph incidence matrix (maps nodes to hyperedges) |
| \( W \) | Diagonal matrix of learnable hyperedge weights |
| \( D_v, D_e \) | Node and hyperedge degree matrices (for normalization) |
| \( \Theta \) | Learnable transformation weight matrix |
| \( \sigma \) | Activation function (e.g., ReLU) |

Efficiently implemented with `torch.einsum`, this formulation ensures that each token updates its representation based on all other tokens within the same hyperedge.

---

## 🚀 4. Future Work & Improvements

The current model provides a strong baseline, but several extensions can significantly improve performance and help achieve **state-of-the-art F1 scores**:

- **🔧 Upgrade the Encoder**  
  Replace `bert-base-uncased` with a larger pre-trained model such as **RoBERTa-large** or **DeBERTa** for richer contextual embeddings.

- **🔄 Add Sequential Layers**  
  Integrate **LSTM** or **GRU** layers after the HyperGCN stack to capture sequential dependencies before the final prediction.

- **📈 Scale Up Training**  
  Train on the **full NewsQA dataset** or even train on the qa dataset and then finetune for newsqa to improve generalization and robustness.
