import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, pipeline
from huggingface_hub import login
import uvicorn
from pypdf import PdfReader
from docx import Document
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# HuggingFace login (optional)
# ==========================
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception:
        pass

# ==========================
# Models
# ==========================
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", "google/flan-t5-small")

_textgen = None

# ==========================
# Load Granite Embedding Model
# ==========================
GRANITE_MODEL_NAME = "ibm-granite/granite-embedding-small-english-r2"
granite_tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL_NAME)
granite_model = AutoModel.from_pretrained(GRANITE_MODEL_NAME)
granite_model.eval()

# Predefined categories for classification
CATEGORIES = ["NDA", "Lease", "Employment", "Service Agreement"]
# You can precompute embeddings for these categories if needed; for demo we use random
CATEGORY_EMBEDDINGS = np.random.rand(len(CATEGORIES), 768)  # 768-dim assumed

# ==========================
# Helpers
# ==========================
def _load_textgen():
    global _textgen
    if _textgen is not None:
        return _textgen
    kwargs = {}
    if torch.cuda.is_available():
        kwargs.update(dict(device_map="auto"))
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_ID, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    _textgen = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return _textgen

def _simplify_text(text: str) -> str:
    tg = _load_textgen()
    prompt = f"Simplify this legal text into plain English:\n{text}"
    outputs = tg(prompt, max_new_tokens=128)
    return outputs[0]["generated_text"].strip()

def _chunk_text(text: str, chunk_size: int = 300):
    """Split text into word chunks to avoid model overflow"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def extract_entities(text: str):
    """Use Granite embeddings to produce token embeddings"""
    inputs = granite_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = granite_model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    entities = [{"text": granite_tokenizer.decode(inputs.input_ids[0][i]), "embedding": embeddings[i].tolist()} 
                for i in range(len(inputs.input_ids[0]))]
    return entities

def get_text_embedding(text: str):
    inputs = granite_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = granite_model(**inputs)
    # Mean pooling over tokens
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()
    return embedding

# ==========================
# FastAPI app
# ==========================
app = FastAPI(title="ClauseWise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Request Models
# ==========================
class TextIn(BaseModel):
    text: str

class BulkIn(BaseModel):
    clauses: List[str]

# ==========================
# Endpoints
# ==========================
@app.post("/simplify")
async def simplify(req: TextIn):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    simplified = _simplify_text(req.text)
    return {"simplified": simplified}

@app.post("/bulk_simplify")
async def bulk_simplify(req: BulkIn):
    results = []
    for clause in req.clauses:
        simp = _simplify_text(clause)
        results.append({"original": clause, "simplified": simp})
    return {"results": results}

@app.post("/classify")
async def classify(req: TextIn):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    embedding = get_text_embedding(text)
    similarities = cosine_similarity(embedding, CATEGORY_EMBEDDINGS)
    predicted_label = CATEGORIES[np.argmax(similarities)]
    return {"label": predicted_label}

@app.post("/ner")
async def ner(req: TextIn):
    text = req.text.strip()
    if not text:
        return {"entities": []}
    try:
        entities = extract_entities(text)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER processing failed: {str(e)}")

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    try:
        if file.filename.endswith(".pdf"):
            reader = PdfReader(file.file)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
        elif file.filename.endswith(".docx"):
            doc = Document(file.file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif file.filename.endswith(".txt"):
            text = file.file.read().decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    return {"text": text}

# ==========================
# Run backend
# ==========================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
