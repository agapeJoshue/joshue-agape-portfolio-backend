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
HF_API_KEY = os.getenv("HF_API_KEY")

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


OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "http://ollama:11434"
)
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")


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
        You are an AI assistant integrated into a developer portfolio.

        - Answer naturally.
        - If the question is about the portfolio owner (skills, projects,
        experience, personal info),
        use ONLY the information below.
        - If it is about something else, answer normally.

        Portfolio information:
        {context}

        User question:
        {q.question}

        Answer:
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 400,
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/v1/completions", json=payload, timeout=60
        )
        response.raise_for_status()
        data = response.json()

        answer = (
            data.get("completion")
            or data.get("completions", [{}])[0].get("output")
            or None
        )
        if not answer:
            raise HTTPException(
                status_code=500, detail="Ollama returned empty response"
            )

        return {"answer": answer.strip()}

    except requests.exceptions.RequestException as e:
        print("Ollama request failed:", e)
        fallback_msg = (
            "Merci pour votre message ! "
            "Le serveur AI n'est pas encoredisponible. "
            "Veuillez réessayer plus tard."
        )
        return {"answer": fallback_msg}
