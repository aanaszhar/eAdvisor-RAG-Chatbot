# eAdvisor: UTP Academic Chatbot
**eAdvisor** is an AI-powered assistant designed to help students navigate Universiti Teknologi PETRONAS (UTP) academic handbooks. Instead of scrolling through 400+ page PDFs, you can ask eAdvisor questions in plain English and get cited, accurate answers instantly.

-------------------------------------------------------------------------
## How It Works
This version uses an advanced Parent-Child Retrieval strategy:* "What are the prerequisites for Object-Oriented Programming?"
* Child Chunks: Small snippets used for high-accuracy semantic search.
* Parent Chunks: Larger context blocks retrieved once a match is found to give the LLM better surrounding info.
* Reranking: A Cross-Encoder model double-checks the results to ensure the most relevant context is prioritized.
* Query Expansion: The system generates multiple versions of your question to improve search coverage.

The system finds the most relevant information from the handbooks and uses a Large Language Model (LLM) to generate a concise, accurate answer.

-------------------------------------------------------------------------
## Quick Start
Prerequisites
* Python 3.10+
* Ollama: (Download here)[https://ollama.com/] and run `ollama pull llama3:8b`

### Step 1: Set Up
```bash
# Clone the project and enter the folder
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt    ```
4.  Activate the environment:
    ```bash
    # On Windows
    .\venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
```
---

### Step 2: Ingest data
Place your UTP handbook PDFs in the `/data` folder, then run:
```bash
python ingest.py
```
This creates the vector database `(faiss_index.faiss)` and the document stores `(child_docs.pkl and parent_docs.pkl)`.

---

### Step 3: Run the app
```bash
streamlit run eadvisor_app.py
```
The interface will open automatically in your browser at `http://localhost:8501`.

---

## Evaluation
To test the accuracy of the system using RAGAs:
* Ensure your test JSON files (e.g., `test_questions_UG.json`) are in the root directory.
* Run: python `evaluate_new_stack.py`.
* Results will be saved as a CSV for analysis.