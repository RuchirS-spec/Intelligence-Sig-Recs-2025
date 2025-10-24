# 🧠 Bilingual Hypergraph QA

**Note:**  
The code in `finale.ipynb` was my attempt at writing the code somewhat myself, but I kept running into errors and tried using Colab’s built-in Gemini to solve them, which became messy. I came up with the entire concept myself and explained it to ChatGPT, and the code in `bilingual_hyper_graph` is an entirely non-AI made concept; I generated all the code by ChatGPT just to test if it works.

---

## 🔹 Overview of the Process

- **Goal:** Answer **English questions in French** using English context and its French translation.  
- **Core idea:** Parallel hypergraph processing + bipartite-style cross-attention for bilingual semantic alignment.

---

## 🏗️ Design Choices

### Why Hypergraphs?
- **Higher-order relationships:** Unlike traditional graphs (pairwise edges), hypergraphs connect **multiple nodes with a single hyperedge**, capturing relationships among groups of tokens.  
- **Sentence-to-sentence semantic meaning:** Each sentence can form a hyperedge connecting all its tokens. This allows semantic dependencies to propagate across entire sentences.  
- **Word-to-word semantic meaning:** Sliding window or n-gram hyperedges capture local context relationships between words within phrases.  
- **Question-context interaction:** A special hyperedge can connect all question tokens, allowing the model to focus on relevant parts of the context.

### How Hyperedges are Formed
- **Sentence-level:** Each sentence in the context → hyperedge connecting all tokens in that sentence.  
- **Sliding-window n-grams:** Overlapping sequences of tokens → hyperedges to capture local word relationships.  
- **Question hyperedge:** All question tokens → hyperedge connecting question to context tokens during propagation.  
- **Bipartite alignment (cross-attention):** Hyperedges from English context interact with French context through cross-attention rather than direct edges, achieving a conceptual bipartite graph.

### Semantic Propagation
- **Within-language:** HyperGCN propagates semantic information from words → phrases → sentences via the hyperedges.  
- **Cross-language:** Cross-attention propagates semantic meaning between English and French representations, aligning translations and enabling better answer extraction.  

---

## 🔹 Step-by-Step Process
1. **Tokenization:** English question, English context, French context → token embeddings.  
2. **Hypergraph Construction:** Build hyperedges (sentence-level, n-grams, question-centered).  
3. **HyperGCN Encoding:** Independent propagation in English and French streams.  
4. **Bipartite Cross-Attention:** English ↔ French token interaction; fusion via residual + LayerNorm.  
5. **Answer Prediction:** Start/end logits over French tokens.  
6. **Evaluation:** EM/F1 on predicted French answers.

---

## ⚙️ Key Features
- Captures **sentence-level and word-level semantics** via hyperedges.  
- Parallel processing of bilingual contexts for richer alignment.  
- Cross-attention as a **conceptual bipartite graph** between English and French tokens.  

---

