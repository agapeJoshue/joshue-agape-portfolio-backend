import os
from dotenv import load_dotenv
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from jinja2 import Environment, FileSystemLoader

load_dotenv()

env = Environment(loader=FileSystemLoader("templates"))

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)


async def send_project_email(data: dict):
    try:
        template = env.get_template("project_email.html")
        html_content = template.render(**data)

        message = MessageSchema(
            subject=f"Nouveau projet – {data.get('subject', 'Sans sujet')}",
            recipients=["agapedev.dark@gmail.com"],
            body=html_content,
            subtype="html",
            sender="My Portfolio",
        )

        fm = FastMail(conf)
        await fm.send_message(message)
        print("Email envoyé avec succès !")

    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email : {e}")
        raise e
