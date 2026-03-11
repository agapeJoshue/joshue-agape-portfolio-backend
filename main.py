import os
import time
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


HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = "sarvamai/sarvam-30b"
MODEL_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"


class Question(BaseModel):
    question: str


def load_context() -> str:
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print("ERROR loading data.txt:", e)
        return "Portfolio information not available."


def query_huggingface(prompt: str, retries: int = 2, delay: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                MODEL_URL,
                headers={
                    "Authorization": f"Bearer {HF_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "temperature": 0.7,
                        "max_new_tokens": 400,
                        "return_full_text": False,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            else:
                print("WARNING: Invalid HF response structure:", data)
        except requests.exceptions.RequestException as e:
            print(f"WARNING: HF request failed (attempt {attempt + 1}): {e}")
            if attempt < retries:
                time.sleep(delay)
    return (
        "Merci pour votre message ! "
        "Le serveur AI n'est pas encore disponible. "
        "Veuillez réessayer plus tard."
    )


@app.post("/ask")
def ask_ai(q: Question):
    context = load_context()

    prompt = f"""
        You are an AI assistant integrated into a developer portfolio.

        Rules:
        - Answer questions naturally and clearly.
        - If the question is about the portfolio owner
            (skills, projects, experience, personal info),
            ONLY use the information in the portfolio context.
        - If the question is about something else, answer normally
            without referencing the portfolio context.

        Portfolio information:
        {context}

        User question:
        {q.question}

        Answer:
    """

    answer = query_huggingface(prompt)
    return {"answer": answer}
