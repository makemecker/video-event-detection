import os
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
CLIENT_ID = os.getenv("CLIENT_ID")
BASE_API = os.getenv("BASE_API")
PROXY_KEY = os.getenv("PROXY_KEY")