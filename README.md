# AI-Powered-Insurance-Claims-Triage-System (Phase 1)

This repository contains the Phase 1 implementation of an AI-driven system for triaging insurance claims. The goal is to automate the classification of claim severity (Low, Medium, High) using a transformer-based language model, while providing a clean and intuitive interface for users via Streamlit. The project focuses on speed, transparency, and scalability by combining a robust backend model with an interactive frontend dashboard.

---

## 📌 What This Phase Includes

Phase 1 focused on establishing the core functionality of the triage system — from dataset preparation and model training to full-stack integration with the user interface. Key accomplishments include:

- Building a **RoBERTa-based transformer model** fine-tuned on synthetic claim narratives.
- Developing a **Streamlit-based frontend** for user login, claim upload, management, and EDA.
- Creating a modular project structure to support future phases like explainability, RAG, and MLOps.
- Implementing the complete AI prediction loop — user inputs a claim, the model evaluates it, and the frontend displays the severity classification.

---

## 🧩 Phase 1 – Core Model Integration

In this phase, we developed a transformer-based classifier (fine-tuned RoBERTa) trained on a custom dataset derived from real insurance claim data. Raw, keyword-style descriptions were enriched using Falcon-7B to simulate natural language narratives. These were combined with structured metadata (e.g., age, salary, accident date) to form rich `InputText` examples for training.

The Streamlit frontend was simultaneously developed to serve as the user-facing layer of the system. Users can log in, upload claims (individually or in batch), view predicted severity classifications, and explore trends across demographic variables using the EDA dashboard. A management interface was also implemented to browse through uploaded claims in a paginated table. The complete flow — from claim upload to risk classification — is now functional and ready for real-world validation.

---

## 📊 Streamlit Pages & Features

| Page             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Login/Register   | Secure user authentication and account management                          |
| Claim Upload     | Upload claims manually or via CSV/XLSX batch files                          |
| Management Page  | Browse uploaded claims and their predicted severities (paginated view)      |
| EDA Dashboard    | Visual insights into claim patterns by age, gender, employment type, etc.   |


## 🧠 Model Details

- **Architecture**: `roberta-base` (fine-tuned)
- **Input**: Combined structured data + enriched textual narratives
- **Output**: Severity classification — `Low`, `Medium`, `High`
- **Accuracy**: ~99% (note: inflated due to synthetic training set, see docs)
- **Data Size**: 21,684 claims


## 📄 Key Reports

- [`Frontend Development Report`](./docs/Frontend_Development_Report.pdf) – UI architecture, page breakdown, workflow
- [`Phase 1 LLM Claims Report`](./docs/MS1_LLM_Triage_Report.pdf) – Dataset prep, model training, performance evaluation

---

