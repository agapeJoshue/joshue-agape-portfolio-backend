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
    with open("data.txt", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
def ask_ai(q: Question):
    context = load_context()

    prompt = f"""
You are an assistant for my portfolio.
Answer only using the information below.

Information:
{context}

Question:
{q.question}
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get("choices", [{}])[0].get("message", {}).get("content")

        if not answer:
            raise HTTPException(status_code=500, detail="AI response invalid")

        return {"answer": answer}

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="AI request timeout")

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Unable to connect to AI service")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"AI request error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
