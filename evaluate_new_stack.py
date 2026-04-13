import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
import time
import sys
import pickle

try:
    from eadvisor_app import (
        retrieve_and_rerank,
        format_prompt_with_context,
        call_ollama,
        EMBEDDING_MODEL,
        LLM_MODEL
    )
except ImportError as e:
    print(f"Error: Could not import from eadvisor_app.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during import: {e}")
    sys.exit(1)

print("--- RAGAs Evaluation Script ---")

# --- 1. Set up Evaluation Models ---
eval_llm = OllamaLLM(model="llama3:8b") 
eval_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)
print("Evaluation models loaded.")

# --- 2. Load Your Test Questions ---
TEST_FILES_MAP = {
    "test_questions_PG.json": "PG Handbook.pdf",
    "test_questions_CFS.json": "CFS Handbook.pdf",
    "test_questions_UG.json": "UG Handbook.pdf"
}

evaluation_data = {
    "question": [],
    "ground_truth": [],
    "answer": [],
    "contexts": []
}

print(f"Loading and processing questions from {len(TEST_FILES_MAP)} test files...")

for test_file, handbook_name in TEST_FILES_MAP.items():
    if not os.path.exists(test_file):
        print(f"Warning: Test file '{test_file}' not found. Skipping.")
        continue
        
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            print(f"  - Loaded {len(test_data)} questions from {test_file} (filtering for '{handbook_name}')")
            
            for item in test_data:
                question = item.get("question")
                ground_truth = item.get("answer")
                
                if not question or not ground_truth:
                    continue
                
                retrieved_docs = retrieve_and_rerank(
                    question, 
                    handbook_name, 
                    chat_history=[]
                )
                
                full_prompt = format_prompt_with_context(question, retrieved_docs)
                response_text = call_ollama(full_prompt, chat_history=[])
                
                answer = response_text
                
                contexts_list = [doc['text'] for doc in retrieved_docs]
                
                evaluation_data["question"].append(question)
                evaluation_data["ground_truth"].append(ground_truth)
                evaluation_data["answer"].append(answer)
                evaluation_data["contexts"].append(contexts_list)

    except Exception as e:
        print(f"Error processing {test_file}: {e}")

print(f"\nTotal questions prepared for evaluation: {len(evaluation_data['question'])}")

# --- 4. Prepare Dataset for RAGAs ---
dataset = Dataset.from_dict(evaluation_data)
print("Dataset prepared for RAGAs.")

# dataset = Dataset.from_dict(evaluation_data)

# SAMPLE_SIZE = 15
# if len(dataset) > SAMPLE_SIZE:
#     dataset = dataset.shuffle(seed=42).select(range(SAMPLE_SIZE))
#     print(f"--- Running evaluation on a random sample of {SAMPLE_SIZE} questions. ---")
# else:
#     print(f"--- Running evaluation on all {len(dataset)} questions. ---")

# --- 5. Run RAGAs Evaluation ---
print("Running RAGAs evaluation... This may take a while.")
start_time = time.time()

metrics_to_run = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
]

result = evaluate(
    dataset=dataset,
    metrics=metrics_to_run,
    llm=eval_llm,
    embeddings=eval_embeddings,
    run_config=RunConfig(max_workers=1)
)

end_time = time.time()
print(f"Evaluation complete in {end_time - start_time:.2f} seconds.")

# --- 6. Display Results ---
print("\n--- RAGAs Evaluation Metrics (Overall) ---")
print(result)

df = result.to_pandas()
print("\n--- Evaluation DataFrame (Per Question) ---")
print(df)

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RESULTS_FILE_PATH = f"ragas_evaluation_results_{TIMESTAMP}.csv"
df.to_csv(RESULTS_FILE_PATH, index=False)
print(f"\nResults saved to '{RESULTS_FILE_PATH}'")