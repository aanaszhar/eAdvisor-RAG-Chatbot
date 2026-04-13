# 🎓 eAdvisor: UTP Academic Chatbot
**eAdvisor** is a semantic-search enabled chatbot designed to provide fast and accurate answers to questions about Universiti Teknologi PETRONAS (UTP) academic handbooks. This project, part of the "eAdvisor" Final Year Project, addresses the inefficiency of manually searching large PDF handbooks by implementing a Retrieval-Augmented Generation (RAG) system.

-------------------------------------------------------------------------
## 🚀 What It Does
Instead of manually using "CTRL+F" to find information in 400+ page documents, users can ask the chatbot natural language questions like:
* "What are the prerequisites for Object-Oriented Programming?"
* "What is the minimum attendance requirement?"
* "How many credit hours are required for the Dean's List?"

The system finds the most relevant information from the handbooks and uses a Large Language Model (LLM) to generate a concise, accurate answer.

-------------------------------------------------------------------------
## ⚙️ Core Components & Architecture

This system is built entirely on free, open-source tools and runs locally.

* **PDF Extraction:** `PyMuPDF`
* **Text Embedding:** `all-MiniLM-L6-v2` (via Hugging Face)
* **Vector Database:** `FAISS` (Facebook AI Similarity Search)
* **Orchestration:** `LangChain`
* **Local LLM:** `Gemma-3:4B` (via `Ollama`)
* **Backend API:** `Flask`
* **Frontend:** `HTML`, `CSS`, `JavaScript`



The architecture is decoupled into three main parts:
1.  **`ingest.py` (Data Pipeline):** An offline script that reads all PDFs from the `/data` folder, splits them into chunks, generates vector embeddings, and saves them into a local `FAISS` index.
2.  **`app.py` (Backend API):** A Flask server that loads the FAISS index and the RAG chain. It exposes an `/api/query` endpoint that receives a user's question and returns a JSON answer.
3.  **`templates/index.html` (Frontend):** A simple chat interface that sends the user's query to the Flask API and displays the response.

## 🛠️ Setup & Run Instructions
Follow these steps to set up and run the project locally.

### Prerequisites
1.  **Python:** Python 3.10 or newer.
2.  **Ollama:** You must [download and install Ollama](https://ollama.com/).

---

### Step 1: Set Up the Environment

1.  Clone or download this project.
2.  Open a terminal in the project's root directory (`/eAdvisor_Project/`).
3.  Create a Python virtual environment:
    ```bash
    python -m venv venv
    ```
4.  Activate the environment:
    ```bash
    # On Windows
    .\venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```

---

### Step 2: Install Dependencies

1.  Ensure your virtual environment is active.
2.  Install all required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt`, install the libraries manually: `pip install flask langchain langchain-community langchain-huggingface langchain-ollama langchain-text-splitters PyMuPDF faiss-cpu sentence-transformers requests`)*

---

### Step 3: Download the Local LLM

1.  Make sure the Ollama application is running in the background.
2.  Pull the `gemma3:4b` model:
    ```bash
    ollama pull gemma3:4b
    ```

---

### Step 4: Prepare the Data (Ingestion)

1.  Place all your UTP handbook PDFs (e.g., `CFS Handbook.pdf`) inside the `/data` folder.
2.  Run the ingestion script. This will read the PDFs, create the vector index, and save it to a new `faiss_index` folder.
    ```bash
    python ingest.py
    ```
    *(This only needs to be run once, or again if you add/change the PDFs.)*

---

### Step 5: Run the Backend Server

1.  Make sure Ollama is still running.
2.  In the same terminal, run the Flask application:
    ```bash
    python app.py
    ```
3.  Wait for the server to load the model. You will see a message like `eAdvisor RAG chain loaded. Server is ready.`
4.  **Leave this terminal running.**

---

### Step 6: Access the Chatbot

1.  Open your web browser.
2.  Go to the following address:
    **`http://127.0.0.1:5000`**

You should now see the eAdvisor chat interface and can begin asking questions.

## 🚀 How to Run (Simple Steps)

After you have completed the one-time setup, you can run the chatbot anytime with these 3 simple steps:

1.  **Start Ollama:** Make sure the Ollama application is running in the background. (`ollama pull gemma3:4b`)
2.  **Run the Server:** Open your terminal, activate your environment (`.\venv\Scripts\activate`)
3.  **Run the Flask app:** Run in same terminal (`python app.py`)
4.  **Open the Chatbot:** Once the server says `eAdvisor RAG chain loaded. Server is ready.`, open your web browser and go to:
    **`http://127.0.0.1:5000`**

---


## 🧪 How to Evaluate (Optional)

To test the chatbot's accuracy, you can run the evaluation script.

1.  First, manually add your "ground truth" questions and answers to the `test_questions.json` file.
2.  Make sure your `app.py` server is running (Step 5).
3.  Open a **second, new terminal** and activate the `venv`.
4.  Run the evaluation script:
    ```bash
    python evaluate.py
    ```
5.  This will create a new `evaluation_results_...json` file for you to manually review and score the bot's performance.