


from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
api_key = os.getenv("gemini_api_key")

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone client
pinecone = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

index_name = "medi-chat"

# Loading the index without passing 'client'
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Set up the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

# Set up the LLM (Google Gemini in this case)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Set up the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
