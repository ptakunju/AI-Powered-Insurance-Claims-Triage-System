# AI-Powered-Insurance-Claims-Triage-System

This project aims to automate the risk triage process for insurance claims using a combination of Large Language Models (LLMs), metadata analysis, and explainable AI. Traditional manual claim triaging is labor-intensive and prone to human bias. By introducing a structured, AI-powered solution, we target faster and more accurate claim classification.

## Background and Goals
This solution is tailored to address the need for scalable claim triage by leveraging:
- Fine-tuned transformer models like RoBERTa
- Retrieval-Augmented Generation (RAG) pipelines
- Explainable AI tools (LIME/SHAP)
- A DevOps-ready infrastructure for CI/CD and monitoring
The goal is to produce a real-time, transparent, and scalable classification engine that categorizes insurance claims into Low, Medium, or High severity/risk.

## Approach
The project is being implemented using a phase-wise development strategy, allowing for modularity, rapid iteration, and effective testing of each component. This structured approach not only accelerates development but also facilitates the integration of MLOps and DevOps pipelines in subsequent stages.

## Phase-Wise Development Plan

### 🔹 Phase 1 – Core Model Development
- Build the core LLM-based classifier using a fine-tuned transformer model like RoBERTa, focused on predicting claim severity based on structured data and claim description.
- Build Streamlit pages for user login, claim upload (single and batch), and management; display model-predicted claim severity using the integrated RoBERTa-based classifier.

### 🔹 Phase 2 – Explainability Integration (LIME/SHAP)
- Incorporate explainable AI tools to interpret the model’s decisions, offering transparency to reviewers and ensuring regulatory compliance.
- Implement the Explanation page to visualize SHAP and LIME outputs for each prediction, and integrate database connectivity for storing and retrieving user data, claims, predictions, and explanation metadata.

### 🔹 Phase 3 – RAG Pipeline Integration
- Implement Retrieval-Augmented Generation using policy documents. This grounds model decisions in source documentation for traceability and improved performance.
- Enhance the Explanation page to include retrieved policy document excerpts and rationale generated via the RAG pipeline, providing context and traceability behind AI decisions.

### 🔹 Phase 4 – MLOps and DevOps Pipelines
- Integrate model retraining, monitoring, version control (GitHub), and automated deployment mechanisms using tools like MLflow, Docker, and CI/CD pipelines.

---

## Phase 1 – Core Model Development

This repository contains the Phase 1 implementation of an AI-driven system for triaging insurance claims. The goal is to automate the classification of claim severity (Low, Medium, High) using a transformer-based language model, while providing a clean and intuitive interface for users via Streamlit. The project focuses on speed, transparency, and scalability by combining a robust backend model with an interactive frontend dashboard.

### 📌 What This Phase Includes

Phase 1 focused on establishing the core functionality of the triage system — from dataset preparation and model training to full-stack integration with the user interface. Key accomplishments include:

- Building a **RoBERTa-based transformer model** fine-tuned on synthetic claim narratives.
- Developing a **Streamlit-based frontend** for user login, claim upload, management, and EDA.
- Creating a modular project structure to support future phases like explainability, RAG, and MLOps.
- Implementing the complete AI prediction loop — user inputs a claim, the model evaluates it, and the frontend displays the severity classification.

### 🧩 Phase 1 – Core Model Integration

In this phase, we developed a transformer-based classifier (fine-tuned RoBERTa) trained on a custom dataset derived from real insurance claim data. Raw, keyword-style descriptions were enriched using Falcon-7B to simulate natural language narratives. These were combined with structured metadata (e.g., age, salary, accident date) to form rich `InputText` examples for training.

The Streamlit frontend was simultaneously developed to serve as the user-facing layer of the system. Users can log in, upload claims (individually or in batch), view predicted severity classifications, and explore trends across demographic variables using the EDA dashboard. A management interface was also implemented to browse through uploaded claims in a paginated table. The complete flow — from claim upload to risk classification — is now functional and ready for real-world validation.

### 📊 Streamlit Pages & Features

| Page             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Login/Register   | Secure user authentication and account management                          |
| Claim Upload     | Upload claims manually or via CSV/XLSX batch files                          |
| Management Page  | Browse uploaded claims and their predicted severities (paginated view)      |
| EDA Dashboard    | Visual insights into claim patterns by age, gender, employment type, etc.   |


### 🧠 Model Details

- **Architecture**: `roberta-base` (fine-tuned)
- **Input**: Combined structured data + enriched textual narratives
- **Output**: Severity classification — `Low`, `Medium`, `High`
- **Accuracy**: ~99% (note: inflated due to synthetic training set, see docs)
- **Data Size**: 21,684 claims

---

