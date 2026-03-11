import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from services.email_service import send_project_email

app = FastAPI(title="Portfolio API avec OpenAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://raharison-joshue-agape-folio.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")


@app.get("/")
def root():
    return {"message": "Hello FastAPI"}


@app.get("/files")
def list_files():
    return os.listdir(DOCUMENTS_DIR)


@app.get("/download/{filename}")
def download_file(filename: str):
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Filename invalid")

    file_path = os.path.join(DOCUMENTS_DIR, filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)


class ProjectForm(BaseModel):
    full_name: str
    email: EmailStr
    location: str
    budget: str
    subject: str
    description: str


@app.post("/contact")
async def contact(form: ProjectForm):
    try:
        await send_project_email(form.dict())
        return {"message": "Email envoyé avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class Question(BaseModel):
    question: str


def load_context():
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print("ERROR loading data.txt:", e)
        return "No portfolio information available."


@app.post("/ask")
def ask_ai(q: Question):
    context = load_context()

    prompt = f"""
You are an AI assistant for my portfolio.

Answer any question, but if the question is about me, use ONLY the information below.

Portfolio info:
{context}

Question:
{q.question}
"""

    try:
        payload = {
            "model": "llama-3.1-8b-instant",  # mixtral-8x7b
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant in a developer portfolio.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }

        print("DEBUG: Payload to Groq:", payload)

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()
        print("DEBUG: AI response:", data)

        answer = data.get("choices", [{}])[0].get("message", {}).get("content")

        if not answer:
            raise HTTPException(status_code=500, detail="AI response invalid")

        return {"answer": answer}

    except requests.exceptions.RequestException as e:
        print("ERROR: AI request failed", e)
        raise HTTPException(status_code=500, detail=f"AI request error: {str(e)}")
