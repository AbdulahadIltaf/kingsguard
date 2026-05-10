---
title: KingsGuard
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🛡️ KingsGuard: 5-Layer Agentic Security Framework

**KingsGuard** is a mathematically rigorous, defense-in-depth security framework designed specifically for Autonomous Agent systems (like CrewAI). It prevents Prompt Injections, Zero-Day Exploits, Sleeper Agents, and Telemetry Poisoning through a unique multi-layered architecture.

---

## 🏛️ Architecture Overview

Rather than relying on simple keyword filters, KingsGuard implements five distinct security layers that gate every agent action:

1.  **Layer 1: Semantic Screener (ProtectAI DeBERTa)**
    *   Analyzes the raw semantic intent of user input.
    *   Detects known prompt injection patterns and malicious jailbreak attempts.
    *   **Adaptive Perplexity Filter**: Detects high-perplexity adversarial camouflage.

2.  **Layer 2: VAE Anomaly Profiler**
    *   Uses a **Variational Autoencoder (VAE)** to build a "benign manifold" of expected agent behaviors.
    *   Detects zero-day anomalies by calculating certified reconstruction errors.

3.  **Layer 3: Causal Arbitrator (Council of Rivals)**
    *   Features a "Council of Rivals" (LLMs like Llama 3.1 and Qwen) that independently judge risky actions.
    *   **Structural Causal Modeling (SCM)**: Uses causal discovery to block inadmissible logical paths.

4.  **Layer 4: Bayesian Watchman**
    *   Monitors agent behavior over time using **Bayesian Online Changepoint Detection (BOCPD)**.
    *   Triggers a "Trust Cliff" revocation if behavior drifts into high-risk territory.

5.  **Layer 5: Warden Sandbox**
    *   Executes all tool actions within an isolated Docker container or restricted subprocess.
    *   Performs **Adversarial Rollback** shadow testing to validate configuration updates.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Docker (Optional, for L5 Warden)
*   Groq API Key (For L3 Council)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AbdulahadIltaf/kingsguard.git
   cd kingsguard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # OR if using uv
   uv sync
   ```

3. Set up environment variables:
   Create a `.env` file:
   ```env
   GROQ_API_KEY=your_key_here
   ```

### Running the Dashboard
Start the real-time security pipeline and dashboard:
```bash
python app.py
```
Visit `http://localhost:5000` to view the real-time security telemetry.

---

## 🛠️ Tech Stack
*   **Core**: Python, Flask, CrewAI
*   **ML/AI**: PyTorch, Transformers (HuggingFace), Scikit-Learn
*   **Security**: ProtectAI DeBERTa, VAE Profiling, Causal Learn
*   **Database**: SQLite

---

## ⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
