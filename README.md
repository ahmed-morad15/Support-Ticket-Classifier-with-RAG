#  Support Ticket Classification using RAG + LLM

##  Overview
This project is an AI-powered Support Ticket Classification system that uses a Retrieval-Augmented Generation (RAG) pipeline to automatically classify support tickets into predefined categories and generate a short solution for each issue.

The system combines:
-  Large Language Model (Llama 3.1 via Groq API)
-  FAISS Vector Database for semantic search
-  HuggingFace Embeddings
-  LangChain RAG pipeline
-  Flask Web Application (Frontend + Backend)

---

# Project Structure

```
Support-Ticket-Classifier-with-RAG/
│
├── server.py                # Flask backend (RAG pipeline + API)
├── templates/
│   └── index.html           # Simple web interface
├── Data/
│   └── knowledge_base.txt   # Knowledge base (categories)
├── images/
|   ├── single_ticket.png    # Screenshot of an example classification result
|   └── bulk_ticket.png      # Screenshot of an example classification result
|
├── requirements.txt         # Dependencies
├── notebook.ipynb           # Experimentation & testing
└── README.md                # Project documentation
```
---

##  Features
- Automatic support ticket classification
- Categorization into predefined issue types:
  - Login Issues
  - App Functionality
  - Billing
  - Account Management
  - Performance Issues
- AI-generated short solution for each ticket
- RAG-based context-aware responses
- Simple web interface (HTML + Flask)
- Bulk ticket classification support

---

##  System Architecture

1. **Knowledge Base Creation**
   - Predefined support categories with descriptions

2. **Document Processing**
   - Text splitting into chunks for better retrieval

3. **Embeddings Generation**
   - Using HuggingFace `all-MiniLM-L6-v2`

4. **Vector Store**
   - FAISS used for similarity search

5. **RAG Pipeline**
   - Retrieves relevant context
   - Sends it to LLM (Llama 3.1 via Groq)
   - Generates:
     - Category
     - Solution

---

##  Tech Stack
- Python 
- Flask 
- LangChain 
- FAISS 
- HuggingFace Transformers 
- Groq API 
- Sentence Transformers

---

##  Installation

### 1. Clone Repository
```bash
git clone <repo-url>
cd Support-Ticket-Classifier-with-RAG
````

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Run the Project

### Start Flask Server

```bash
python server.py
```

Then open in browser:

```
http://localhost:5000/
```

---

##  Usage

###  Single Ticket

* Enter a support ticket
* Click "Classify"
* Get:

  * Category
  * Solution

---

###  Bulk Tickets

* Enter multiple tickets (one per line)
* Click "Classify All"
* Get structured results for each ticket

---

## Example Output

Below is a snapshot of an example classification result:

<p align="center">
  <img src="images\single_ticket.png" width="700"/>
</p>

---

##  Limitations

* Performance depends on quality of knowledge base
* May misclassify ambiguous or unseen cases
* Requires structured categories for best results

---

##  Future Improvements

* Improve knowledge base with real-world tickets
* Add authentication system
* Deploy on cloud (Render / AWS / Railway)
* Replace HTML UI with React or Streamlit
* Store ticket history in database
* Return structured JSON using schema validation

---

##  Author

**Ahmed Morad**<br>
AI/ML Engineer

---
