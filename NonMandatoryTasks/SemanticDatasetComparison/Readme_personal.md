# 🌟 SeSS: Semantic Similarity Score via Scene Graphs (Unsupervised Pipeline)

The **SeSS** project provides an **unsupervised** method for quantifying the **semantic similarity** between two images.  
The core pipeline converts visual content into structured **Scene Graphs** and then uses a simplified **Graph Convolutional Network (GCN)**-like approach to generate comparable graph embeddings.

> **Example Insight:**  
> A similarity score of around `0.5` between two images (e.g., *a woman with her dog on the beach* and *the beach during sunset*) is perfect — it reflects that only the **beach** aspect is semantically common, while everything else differs in meaning.

---

## ⚙️ Pipeline and Architecture

The process is divided into three consecutive stages:

---

### 1️⃣ Image Captioning

- **Model:** [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) (BLIP Base)
- **Purpose:** Converts raw images into concise, descriptive text captions — the linguistic foundation for later stages.
- **Example Output:**  
  🖼️ *"a woman sitting on the beach with her dog"*

---

### 2️⃣ Scene Graph Generation (Linguistic Parsing)

- **Tool:** [spaCy](https://spacy.io) (`en_core_web_sm`)
- **Method:** A rule-based parser (`robust_scene_parse`) extracts **(subject, relation, object)** triples from the caption.
- **Parsing Strategy:**  
  Focuses on identifying relations based on **verbs**, **prepositions**, and **adjectival clauses**.

**Graph Representation:**

| Component | Description |
| :-- | :-- |
| **Nodes** | Entities (e.g., `"woman"`, `"dog"`, `"beach"`) |
| **Edges** | Relations (e.g., `"sitting"`, `"with"`, `"on"`) |

The resulting **directed graph (NetworkX)** encodes how objects interact within the scene.

---

### 3️⃣ Graph Embedding via Message Passing

This stage converts the relational Scene Graph into a dense vector embedding that can be compared across images.

| Component | Model / Method | Description |
| :-- | :-- | :-- |
| **Node Feature Matrix** ($\mathbf{X}$) | [Sentence Transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (`all-MiniLM-L6-v2`) | Encodes each entity (node label) into a semantically rich embedding vector. |
| **Graph Filter** ($\mathbf{A}_{\text{norm}}$) | Degree Normalization | The adjacency matrix ($\mathbf{A}$) is augmented with self-loops ($\mathbf{A} + \mathbf{I}$) and normalized ($\mathbf{D}^{-1}\mathbf{A}$). |
| **Message Passing** | GCN-like Propagation | Information diffuses across connected nodes over $K=2$ iterations:  
  $$\mathbf{H}^{(k)} = \text{Normalize}(\mathbf{A}_{\text{norm}} \mathbf{H}^{(k-1)})$$  
  This allows related entities (e.g., `"dog"` ↔ `"woman"`) to influence each other’s embeddings. |
| **Graph Vector** ($\mathbf{g}_{\text{vec}}$) | Averaging & Normalization | The final graph embedding is computed as:  
  $$\mathbf{g}_{\text{vec}} = \text{Normalize}(\text{Mean}(\mathbf{H}^{(K)}))$$ |

---

## 📊 Semantic Similarity Score

- The **SeSS score** is computed as the **cosine similarity** between two final graph embeddings:  
  $$\text{Sim} = \cos(\mathbf{g}_{\text{vec},1}, \mathbf{g}_{\text{vec},2})$$

- **Core Strength:**  
  By leveraging **scene graphs + message passing**, the system compares images based on the **meaning and context** of their components — not just surface keywords or pixels.

- **Example Result:**  
  ```text
  Semantic Similarity = 0.5336
