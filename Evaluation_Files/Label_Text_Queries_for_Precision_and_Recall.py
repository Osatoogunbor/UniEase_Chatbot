import asyncio
import nest_asyncio

nest_asyncio.apply()  # Allow nested event loops in Jupyter

import os
from dotenv import load_dotenv
import pandas as pd
from pinecone import Pinecone
from openai import AsyncOpenAI

# ----------------------------------------------------------------------------
# 1) LOAD ENVIRONMENT VARIABLES
# ----------------------------------------------------------------------------
load_dotenv()  # Ensure API keys are loaded from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env file (or environment).")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY in .env file (or environment).")
if not PINECONE_ENV:
    raise ValueError("❌ Missing PINECONE_ENV in .env file (or environment).")

# ----------------------------------------------------------------------------
# 2) INITIALIZE OPENAI & PINECONE
# ----------------------------------------------------------------------------
client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # ✅ Use AsyncOpenAI
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "ai-powered-chatbot"
index = pc.Index(INDEX_NAME)


# ----------------------------------------------------------------------------
# 3) ASYNC EMBEDDING FUNCTION (FIXED)
# ----------------------------------------------------------------------------
async def get_embedding_async(text, model="text-embedding-ada-002"):
    """
    Fetch an embedding asynchronously using OpenAI's new API.
    """
    response = await client.embeddings.create(
        model=model,
        input=[text]
    )
    return response.data[0].embedding


# ----------------------------------------------------------------------------
# 4) MAIN LABELING LOGIC (ASYNC)
# ----------------------------------------------------------------------------
async def label_queries_async(csv_path="query_only_dataset.csv"):
    """
    Processes queries from a CSV file, retrieves embeddings asynchronously,
    queries Pinecone for matches, and allows manual labeling.
    """
    df = pd.read_csv(csv_path)
    if "RelevantChunkIDs" not in df.columns:
        df["RelevantChunkIDs"] = ""

    for i in range(len(df)):
        query_text = df.loc[i, "Query"]
        print(f"\n=== Query {i + 1}/{len(df)} ===")
        print(f"User Query: {query_text}")

        # 1) Get embedding asynchronously (✅ FIXED)
        try:
            emb = await get_embedding_async(query_text)
        except Exception as e:
            print("❌ Error calling OpenAI Embeddings API:", e)
            df.loc[i, "RelevantChunkIDs"] = "None"
            continue

        # 2) Query Pinecone (synchronous, as Pinecone is not async)
        try:
            res = index.query(vector=emb, top_k=10, include_metadata=True)
        except Exception as e:
            print("❌ Error querying Pinecone:", e)
            df.loc[i, "RelevantChunkIDs"] = "None"
            continue

        # 3) Show results
        if not res or not res.matches:
            print("⚠️ No matches returned.")
            df.loc[i, "RelevantChunkIDs"] = "None"
            continue

        for match_idx, match in enumerate(res.matches):
            chunk_id = match.id
            text_chunk = match.metadata.get("text_chunk", "")
            score = match.score
            snippet = (text_chunk[:200] + "...") if len(text_chunk) > 200 else text_chunk
            print(f"[{match_idx + 1}] Chunk ID: {chunk_id}")
            print(f"     Score: {score:.3f}")
            print(f"     Text:  {snippet}")

        # 4) Prompt user for relevant chunk IDs
        print("Enter the chunk ID(s) that best answer the query.")
        print("If multiple, separate with commas. If none are correct, leave blank.")
        user_input = input("Relevant chunk IDs: ").strip()

        df.loc[i, "RelevantChunkIDs"] = user_input if user_input else "None"

    # 5) Save the updated CSV
    df.to_csv("labeled_test_queries.csv", index=False)
    print("\n✅ All done! 'labeled_test_queries.csv' now has a RelevantChunkIDs column.")


# ----------------------------------------------------------------------------
# 5) RUN THE LABELING
# ----------------------------------------------------------------------------
async def main():
    await label_queries_async("query_only_dataset.csv")


# ✅ Run the async function properly in Jupyter
if __name__ == "__main__":
    asyncio.run(main())  # Ensures proper execution in a Python script

