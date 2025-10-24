# 🧠 Akinator Clone: Hypergraph Bayesian Inference Engine

The goal was to build a full Akinator website, but due to time constraints, this repository implements only the **core machine learning inference engine**—the "brain" responsible for storing knowledge, asking questions, and calculating the guess. The model uses **Bayesian Inference** and **Hypergraph structures** for robust knowledge representation.

---

## 4. ML Design

### 4.1 Problem Framing

The model’s core function is to find the character with the highest probability given the answers provided.

* **Core Task:** Calculate the **posterior probability** $P(\text{Character} | \text{Answers})$ using probabilistic inference (Bayes' Theorem).
* **Question Selection:** Questions are chosen to maximize **Information Gain (MI)**, aiming to achieve the largest expected reduction in uncertainty (entropy) with each answer.
* **Stopping Condition:** The game ends when one of these conditions is met:
    * Maximum number of questions is reached.
    * Highest probability exceeds a **confidence threshold** ($\tau \approx 0.94$).
    * The **margin** between the top two guesses is sufficient ($\text{Margin} > 0.4$), indicating high separation.

---

### 4.2 Learning & Updates

The architecture supports dynamic learning and improvement of the knowledge base.

* **Knowledge Storage:** New characters and trait corrections are incorporated immediately into the core data structures (the $\mathbf{P}_{\text{yes}}$ probability matrix and hyperfactor definitions).
* **Incremental Updates:** In a production system, a **Naive Bayes count** approach would be used. New games and corrections incrementally update success/failure counts for each character-trait pair, allowing for efficient, ongoing model adjustment without full retraining.
* **Model Versioning:** A robust system would maintain **versioned models** (saving $\mathbf{P}_{\text{yes}}$ and hypergraph structures) and **audit logs** of all game outcomes for periodic batch retraining and quality control.

---

## 5. Model Architecture: Hypergraphs & Bayesian Inference

The model combines a simple, scalable Bayesian framework with Hypergraphs for structured, complex knowledge.

### Bayesian Inference Core

The inference uses log-likelihoods to update probabilities:

1.  **Start:** Begin with uniform **Prior Probabilities** for all characters.
2.  **Update:** For each answer, the **log-likelihood** $P(\text{Answer} | \text{Character})$ is added to the log-prior.
3.  **Result:** The final **Posterior Probability** is found by exponentiating and normalizing the total log-likelihood, providing the model’s current confidence for each character.

### Hypergraph Advantage (Structured Knowledge)

The use of hypergraphs provides a significant advantage over simple CSV or flat-file representations.

| Structure | Definition | Advantage |
| :--- | :--- | :--- |
| **Traits $\rightarrow$ Characters** | Hyperedges are **traits** (e.g., `"is_wizard"`); nodes are the **characters** possessing that trait. | Enables efficient data lookup and **probabilistic filtering** based on single traits. |
| **Hyperfactors $\rightarrow$ Traits** | Hyperedges (Hyperfactors) link **multiple traits** (e.g., `["is_wizard", "has_glasses"]`). | Models **complex, conditional knowledge**. When all traits in a Hyperfactor are 'Yes,' a strong **log-bonus** is applied to specific characters (e.g., Harry Potter), increasing guessing accuracy. |

---

## 6. Optimization: Thompson Sampling (RL)

Thompson Sampling (a Multi-Armed Bandit technique) is used to prevent the model from asking the same, safest questions repeatedly and to encourage the discovery of more effective strategies.

* **Problem:** Maximizing Information Gain on fixed data leads to repetitive questions.
* **TS Method:** Each question has a success history ($\alpha$/$\beta$ counts). At runtime, the model samples a **random weight** from a Beta distribution based on this history.
* **Selection:** The final question choice is based on the Information Gain multiplied by this sampled weight. This balances **exploitation** (choosing a high-gain question) with **exploration** (giving a historically under-used question a chance to prove its long-term value).
