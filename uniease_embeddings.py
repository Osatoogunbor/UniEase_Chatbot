#!/usr/bin/env python3

"""
# UniEase Embedding Code
# This script contains the code is for embedding the knowledge base
and storing it into Pinecone.
"""
#!/usr/bin/env python3

import os
import json
import time
import openai
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import tiktoken

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

# Set API key for OpenAI (synchronous calls)
openai.api_key = OPENAI_API_KEY

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

# Paths
MERGED_JSON_PATH = "/mnt/c/Users/osato/openai_setup/merged_knowledge_base.json"
EMBEDDINGS_OUTPUT_PATH = "/mnt/c/Users/osato/openai_setup/knowledgebase_embeddings.json"

# --------------------------------------------------------------------------
# 2. LOAD MERGED KNOWLEDGE BASE
# --------------------------------------------------------------------------
if not os.path.isfile(MERGED_JSON_PATH):
    raise FileNotFoundError(f"❌ File not found: {MERGED_JSON_PATH}")

with open(MERGED_JSON_PATH, "r", encoding="utf-8") as f:
    knowledgebase = json.load(f)

qa_pairs = knowledgebase.get("qa_pairs", [])
num_qas = len(qa_pairs)
print(f"✅ Loaded {num_qas} QA entries from: {MERGED_JSON_PATH}")

if num_qas == 0:
    print("⚠️ No QA pairs found. Nothing to embed. Exiting.")
    raise SystemExit

# --------------------------------------------------------------------------
# 3. TOKENIZER & CHUNKING UTILS
# --------------------------------------------------------------------------
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

def count_tokens(text: str) -> int:
    """Count tokens for text using the text-embedding-ada-002 tokenizer."""
    return len(tokenizer.encode(text))

def split_text_by_tokens(text: str, max_tokens: int = 512) -> list[str]:
    """Naive word-split approach ensuring <= max_tokens in each chunk."""
    words = text.split()
    chunks = []
    current_words = []

    for word in words:
        current_words.append(word)
        if count_tokens(" ".join(current_words)) > max_tokens:
            # Finalize current chunk
            current_words.pop()
            chunk_str = " ".join(current_words).strip()
            if chunk_str:
                chunks.append(chunk_str)
            current_words = [word]

    # Leftover words
    if current_words:
        leftover_str = " ".join(current_words).strip()
        if leftover_str:
            chunks.append(leftover_str)

    return chunks

# --------------------------------------------------------------------------
# 4. EMBEDDING LOGIC (UPDATING PINECONE CORRECTLY)
# --------------------------------------------------------------------------
def embed_qa_pairs(pairs):
    """
    For each QA pair:
      - Embed the text.
      - Ensure answers are structured properly.
      - Store metadata in Pinecone.
    """
    total = len(pairs)
    print(f"🔵 Embedding {total} QA pairs...")

    for idx, qa in enumerate(pairs):
        doc_id = qa.get("id", f"qa_{idx}")
        category = qa.get("category_id", "unknown")
        source = qa.get("source", "unknown")  # "existing" or "extracted"
        is_emergency = qa.get("is_emergency", False)
        question = (qa.get("question") or "").strip()

        # Ensure answer is a structured dictionary
        ans_block = qa.get("answer", {})
        if not isinstance(ans_block, dict):
            ans_block = {
                "main_points": [str(ans_block).strip()],
                "examples": [],
                "tips": [],
                "related_topics": []
            }

        # Combine answer fields into a single string
        main_points = " ".join(ans_block.get("main_points", []))
        examples = " ".join(ans_block.get("examples", []))
        tips = " ".join(ans_block.get("tips", []))
        rel_topics = " ".join(ans_block.get("related_topics", []))
        combined_answer = f"{main_points}\n{examples}\n{tips}\n{rel_topics}".strip()

        if not question or not combined_answer:
            print(f"⚠️ Skipping doc_id='{doc_id}' due to empty question or answer.")
            continue

        full_text = f"Q: {question}\nA: {combined_answer}"
        full_len = count_tokens(full_text)

        # If the text is too long, split it into chunks
        if full_len > 8192:
            text_chunks = split_text_by_tokens(full_text, max_tokens=512)
        else:
            text_chunks = [full_text]

        print(f"   • {idx+1}/{total} => doc_id='{doc_id}', {len(text_chunks)} chunk(s), tokens={full_len}")

        for c_idx, chunk_str in enumerate(text_chunks):
            try:
                # Generate embedding using the new lowercase endpoint
                resp = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk_str
                )
                # Access the embedding using dot notation
                vector = resp.data[0].embedding

                chunk_id = f"{doc_id}_chunk{c_idx}"
                index.upsert(vectors=[
                    {
                        "id": chunk_id,
                        "values": vector,
                        "metadata": {
                            "doc_id": doc_id,
                            "category": category,
                            "source": source,
                            "is_emergency": is_emergency,
                            "text_chunk": chunk_str
                        }
                    }
                ])

                time.sleep(0.2)  # Avoid rate limits

            except Exception as e:
                print(f"❌ Error embedding doc_id={doc_id}, chunk={c_idx}: {e}")

# --------------------------------------------------------------------------
# 5. MAIN - EMBED & SAVE
# --------------------------------------------------------------------------
if __name__ == "__main__":
    embed_qa_pairs(qa_pairs)
    print("✅ Embedding process completed.")
