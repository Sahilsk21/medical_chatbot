

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = "medi-chat"

# Validate environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from environment variables.")

# Initialize Pinecone client from LangChain (not from official Pinecone package)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load and process PDF data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Create Pinecone index using LangChain's from_texts method
docsearch = Pinecone.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("Embeddings stored successfully in Pinecone!")
