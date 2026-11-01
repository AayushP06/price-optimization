import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Set deployment values in the process environment or a local .env file.
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///local.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Server
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    PORT = int(os.environ.get("PORT", 5000))
    SECRET_KEY = os.environ.get("SECRET_KEY")
    ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
