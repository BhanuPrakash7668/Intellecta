from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv
import os

load_dotenv()

chroma_host = os.getenv("CHROMA_HOST")
chroma_port = os.getenv("CHROMA_PORT")

client = Client(Settings(
    chroma_server_host=chroma_host,
    chroma_server_http_port=chroma_port
))
