⚖️ ClauseWise — Legal Document Analyzer

ClauseWise is an AI-powered platform for analyzing legal documents with speed and accuracy. It converts dense legal text into plain English, extracts entities, classifies documents, and enables powerful search and bulk processing.

Designed with a FastAPI backend and a Streamlit frontend, ClauseWise helps lawyers, businesses, and individuals quickly understand and manage contracts.

🚀 Features
🔹 Core Analysis

AI-Powered Document Insights → Simplify complex legal text into user-friendly language

Document Classification → Detect contract types (NDA, Employment, Lease, Service Agreement, etc.)

Named Entity Recognition (NER) → Extract parties, dates, organizations, amounts, obligations, and key legal terms

Clause Simplification → Translate dense clauses into plain English

Keyword Search → Instantly find clauses or terms within documents

🔹 Enhanced Capabilities

Bulk Simplification → Simplify multiple clauses simultaneously

Multi-Format Support → Process PDF, DOCX, TXT files

Clause Extraction → Break contracts into meaningful clause units for better readability

📁 Project Structure
ClauseWise/
├── backend/
│   ├── main.py              # FastAPI backend
│   ├── requirements.txt     # Backend dependencies
│
├── frontend/
│   ├── app.py               # Streamlit frontend
│   ├── requirements.txt     # Frontend dependencies
│
├── README.md                # Project documentation
└── ...                      # Additional project files

🛠️ Technology Stack

Backend

FastAPI → High-performance API framework

Hugging Face Transformers + PyTorch → AI models for NLP

PyPDF, python-docx, re → Document parsing utilities

Frontend

Streamlit → Interactive user interface

Pandas → Data handling and visualization

🔧 Installation & Setup
Prerequisites

Python 3.8+

pip (Python package manager)

Git

1️⃣ Clone Repository
git clone https://github.com/YOUR-USERNAME/ClauseWise.git
cd ClauseWise

2️⃣ Run Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

3️⃣ Run Frontend (Streamlit)

Open a new terminal:

cd frontend
pip install -r requirements.txt
streamlit run app.py

📖 Usage Guide

📂 Upload Document → Upload a PDF, DOCX, or TXT file

📝 Analyze Document → Extract entities, simplify clauses, and classify contract type

🔍 Search & Simplify → Use keyword search or bulk simplification

📥 Download Results → Export simplified outputs and structured data

🔍 API Endpoints (Backend)

POST /extract → Upload & extract document text

POST /simplify → Simplify legal clauses

POST /bulk_simplify → Simplify multiple clauses

POST /classify → Document type classification

POST /ner → Named Entity Recognition

GET /health → API health check

🎯 Key Components

Clause Simplification → Converts legalese into plain English

Entity Extraction (NER) → Captures parties, dates, monetary values, obligations, organizations

Document Classification → Identifies contract type (e.g., NDA, Service Agreement)

Bulk Processing → Simplifies multiple clauses at once

🔐 Security & Reliability

Input validation for uploaded files

Error handling for corrupted or password-protected documents

Large document support with batching

🤝 Contributing

Fork the repository

Create a feature branch → git checkout -b feature/awesome-feature

Commit changes → git commit -m "Add awesome feature"

Push branch → git push origin feature/awesome-feature

Open a Pull Request

📄 License

Licensed under the MIT License – see the LICENSE file for details.

🙏 Acknowledgments

FastAPI community

Streamlit for elegant frontend

Hugging Face & PyTorch for NLP models

IBM Granite embeddings

Contributors & testers