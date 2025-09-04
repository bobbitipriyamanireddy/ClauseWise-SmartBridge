# ⚖️ ClauseWise — Legal Document Analyzer

## 📖 Overview
ClauseWise is an AI-powered legal document analysis platform that leverages NLP models to simplify, extract, and analyze contracts.  
It provides clause simplification, document classification, entity recognition, keyword search, and bulk processing through an intuitive web interface.

Built with a **FastAPI backend** and a **Streamlit frontend**, ClauseWise helps lawyers, businesses, and individuals quickly understand and manage legal documents.

---

## 🚀 Features

### 🔹 Core Analysis
- **AI-Powered Document Insights** → Simplify complex legal text into plain English  
- **Document Classification** → Detect contract types (NDA, Employment, Lease, Service Agreement, etc.)  
- **Named Entity Recognition (NER)** → Extract parties, dates, organizations, monetary values, obligations, and legal terms  
- **Clause Simplification** → Translate dense clauses into easy-to-read language  
- **Keyword Search** → Instantly find clauses or terms within documents  

### 🔹 Enhanced Capabilities
- **Bulk Simplification** → Simplify multiple clauses at once  
- **Multi-Format Support** → Works with PDF, DOCX, and TXT files  
- **Clause Extraction** → Break down contracts into meaningful clauses for readability  

---

## 📁 Project Structure
ClauseWise/
├── backend/
│ ├── main.py # FastAPI backend
│ ├── requirements.txt # Backend dependencies
├── frontend/
│ ├── app.py # Streamlit frontend
│ ├── requirements.txt # Frontend dependencies
├── README.md # Project documentation
└── ... # Additional project files

yaml
Copy code

---

## 🛠️ Technology Stack

**Backend**
- ⚡ FastAPI → High-performance API framework  
- 🤗 Hugging Face Transformers + PyTorch → NLP models for text analysis  
- 📄 PyPDF, python-docx, re → Document parsing utilities  

**Frontend**
- 🎨 Streamlit → Interactive UI for contract analysis  
- 📊 Pandas → Data handling and visualization  

---

## ⚡ Installation & Setup

### ✅ Prerequisites
- Python 3.8+  
- pip (Python package manager)  
- Git  

### 1️⃣ Clone Repository
```bash
git clone https://github.com/YOUR-USERNAME/ClauseWise.git
cd ClauseWise
2️⃣ Run Backend (FastAPI)
bash
Copy code
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
3️⃣ Run Frontend (Streamlit)
Open a new terminal:

bash
Copy code
cd frontend
pip install -r requirements.txt
streamlit run app.py
📖 Usage Guide
📂 Upload Document → Upload a PDF, DOCX, or TXT file

📝 Analyze Document → Extract entities, simplify clauses, classify contract type

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

Document Classification → Identifies contract type (e.g., NDA, Lease, Service Agreement)

Bulk Processing → Simplifies multiple clauses simultaneously

🔐 Security & Reliability
✅ Input validation for uploaded files

⚠️ Error handling for corrupted or password-protected documents

📑 Supports large document processing with batching

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
