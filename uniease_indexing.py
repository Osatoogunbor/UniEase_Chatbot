"""
# UniEase Pinecone Indexing Code
# This script contains the code for re-indexing the knowledge base into Pinecone.
"""

import json
import os
import time
import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec  # Updated import
from transformers import pipeline

# --------------------------------------------------------------------------
# 1. LOAD ENVIRONMENT & INITIALIZE
# --------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "us-east-1"

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file.")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env file.")
if not PINECONE_ENV:
    raise ValueError("❌ Missing PINECONE_ENV in .env file.")

# Instantiate OpenAI client
openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Instantiate the Pinecone client using the new API
pc = Pinecone(api_key=PINECONE_API_KEY)

# Optionally, create the index if it doesn't exist
existing_indexes = [idx.name for idx in pc.list_indexes()]
if "ai-powered-chatbot" not in existing_indexes:
    # Assuming the embedding dimension is 1536 for text-embedding-ada-002
    pc.create_index(
        name="ai-powered-chatbot",
        dimension=1536,
        metric="cosine",  # or 'euclidean' if that's your metric
        spec=ServerlessSpec(
            cloud="aws",  # update as needed
            region=PINECONE_ENV
        )
    )

# Retrieve the index object
index = pc.Index("ai-powered-chatbot")

print("✅ Pinecone index connected successfully!\n")

# Path to merged knowledge base
MERGED_JSON_PATH = "/mnt/c/Users/osato/openai_setup/merged_knowledge_base.json"

if not os.path.isfile(MERGED_JSON_PATH):
    raise FileNotFoundError("❌ Merged knowledge base file not found.")

# Load merged knowledge base
with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
    knowledgebase = json.load(f)

qa_pairs = knowledgebase.get("qa_pairs", [])
print(f"✅ Loaded {len(qa_pairs)} QA pairs from merged file.")

# Re-index the knowledge base
def index_qa_pairs(pairs):
    for idx, qa in enumerate(pairs):
        question = qa.get("question", "").strip()

        # Ensure "main_points" is a list and join its elements into a single string
        answer_list = qa.get("answer", {}).get("main_points", [])
        if isinstance(answer_list, list):
            answer = " ".join(answer_list).strip()
        else:
            answer = str(answer_list).strip()  # Handle unexpected cases

        full_text = f"Q: {question}\nA: {answer}"

        # Generate embeddings
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=full_text
        )
        vector = response.data[0].embedding

        # Upsert into Pinecone using the index object
        index.upsert(vectors=[
            {
                "id": f"qa_{idx}",
                "values": vector,
                "metadata": {
                    "question": question,
                    "answer": answer
                }
            }
        ])

        time.sleep(0.1)  # Avoid hitting rate limits

if __name__ == "__main__":
    print("🔄 Re-indexing QA pairs...")
    index_qa_pairs(qa_pairs)
    print("✅ Pinecone index updated.")
