import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from flask import Flask, request, jsonify, render_template

# Load environment variables
load_dotenv(dotenv_path="C:/Support-Ticket-Classifier-with-RAG/key.env")

# Retrieve API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Knowledge Base Class
class KnowledgeBase:
    def __init__(self, content: List[str], file_path: str):
        self.content = content
        self.file_path = file_path

    def save_to_file(self):
        with open(self.file_path, "w", encoding="utf-8") as file:
            for entry in self.content:
                file.write(entry + "\n")


# Model Setup
class ModelSetup:
    def __init__(self, api_key: str, model_name: str):
        self.model = ChatGroq(
            api_key=api_key,
            model=model_name
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )


# Vector Store
class VectorStore:
    def __init__(self, file_path: str, embeddings):
        self.file_path = file_path
        self.embeddings = embeddings
        self.vectorstore = None

    def create_vectorstore(self):
        loader = TextLoader(self.file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        splits = splitter.split_documents(docs)

        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

    def get_retriever(self, k: int = 3):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


# -----------------------------
# RAG Chain
# -----------------------------
class ChainSetup:
    def __init__(self, model, retriever):
        self.model = model
        self.retriever = retriever
        self.rag_chain = None

    def create_rag_chain(self):

        system_prompt = """
You are an AI support assistant.

Use the provided background information to generate ONLY a short and helpful solution for the user's issue.

Background Information:
{context}

Rules:
1. Analyze the support ticket carefully.
2. Generate a short practical solution.
3. Keep the response concise (maximum 2-3 lines).
4. Do NOT classify the ticket.
5. Do NOT mention categories.
6. Do NOT add explanations or extra text.
7. If the user input is not a real support issue, random text, greeting, joke, or unrelated message, return:
   Solution: Please enter a valid support issue so I can help you.

Output Format:
Solution: <short solution based on context>

Examples:

Input: "hello"
Output:
Solution: Please enter a valid support issue so I can help you.

Input: "asdfghjkl"
Output:
Solution: Please enter a valid support issue so I can help you.

Input: "The app crashes when I upload a photo."
Output:
Solution: Try updating the application and clearing the app cache before retrying.

Do not add anything else.
"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(
            self.model,
            qa_prompt
        )

        self.rag_chain = create_retrieval_chain(
            self.retriever,
            question_answer_chain
        )

    def invoke_chain(self, input_text: str):
        return self.rag_chain.invoke({"input": input_text})


# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

knowledge_base = None
model_setup = None
vector_store = None
chain_setup = None


# -----------------------------
# Initialize Application
# -----------------------------
def initialize_application():

    global knowledge_base
    global model_setup
    global vector_store
    global chain_setup

    knowledge_base_content = [

"""
Login Issues

Description:
Users may face login problems due to incorrect credentials, expired sessions, or authentication errors.

Solutions:
- Reset the password
- Verify email and username
- Check spam folder for reset email
- Clear browser cache
- Contact support if account remains locked
""",

"""
App Functionality

Description:
Application features may fail due to outdated versions or software issues.

Solutions:
- Update the application
- Restart the app
- Clear app cache
- Reinstall the application
""",

"""
Billing Issues

Description:
Billing problems include duplicate charges, failed payments, or delayed refunds.

Solutions:
- Verify payment history
- Contact bank support
- Retry payment
- Wait for refund processing
""",

"""
Account Management

Description:
Issues related to profile updates, account settings, or linked accounts.

Solutions:
- Refresh and retry changes
- Verify entered information
- Use supported image formats
""",

"""
Performance Issues

Description:
Slow performance may happen because of weak internet connection or device limitations.

Solutions:
- Restart the application
- Close background apps
- Check internet connection
- Update the application
"""
    ]

    file_path = "Data/knowledge_base2.txt"

    knowledge_base = KnowledgeBase(
        knowledge_base_content,
        file_path
    )

    knowledge_base.save_to_file()

    # Setup Model
    model_setup = ModelSetup(
        groq_api_key,
        "llama-3.1-8b-instant"
    )

    # Create Vector Store
    vector_store = VectorStore(
        file_path,
        model_setup.embeddings
    )

    vector_store.create_vectorstore()

    retriever = vector_store.get_retriever()

    # Setup RAG Chain
    chain_setup = ChainSetup(
        model_setup.model,
        retriever
    )

    chain_setup.create_rag_chain()


# Routes
@app.route("/")
def index():
    return render_template("index_2.html")


# Single Ticket Solution
@app.route("/classify", methods=["POST"])
def classify_ticket():

    data = request.json

    if not data or "text" not in data:
        return jsonify({
            "error": "Missing text in request"
        }), 400

    response = chain_setup.invoke_chain(data["text"])

    result_text = response["answer"]

    solution = result_text.replace("Solution:", "").strip()

    return jsonify({
        "solution": solution
    })


# Bulk Ticket Solutions
@app.route("/bulk_classify", methods=["POST"])
def bulk_classify_tickets():

    data = request.json

    if not data or "tickets" not in data:
        return jsonify({
            "error": "Invalid request format"
        }), 400

    results = []

    for ticket in data["tickets"]:

        if "text" in ticket:

            response = chain_setup.invoke_chain(ticket["text"])

            result_text = response["answer"]

            solution = result_text.replace(
                "Solution:",
                ""
            ).strip()

            results.append({
                "text": ticket["text"],
                "solution": solution
            })

    return jsonify({
        "results": results
    })


# Run Server
if __name__ == "__main__":
    initialize_application()
    app.run(debug=True)