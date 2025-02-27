import pandas as pd
import asyncio
import nest_asyncio
from openai import AsyncOpenAI
from pinecone import Pinecone
import os

# Allow async execution inside Jupyter
nest_asyncio.apply()

# Load API Keys
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_4MmQQE_QVvEdpRKLtekQvxKHp9P5QUEWZe7NMzRvdZzXXheWoxTEkTYFYTqYQCYmUop3RP"

# Initialize OpenAI and Pinecone clients
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ai-powered-chatbot")

# ✅ Load the merged test dataset
input_csv_path = "C:/Users/osato/Downloads/test_dataset_merged.csv"  # Update with the actual path
df = pd.read_csv(input_csv_path)

# ✅ Function to retrieve relevant context from Pinecone
async def retrieve_chunks(query: str, top_k: int = 5):
    """Retrieve relevant text chunks from Pinecone based on query embeddings."""
    try:
        embedding_resp = await client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_vector = embedding_resp.data[0].embedding

        pinecone_result = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        if not pinecone_result.matches:
            return []

        return [match.metadata.get("text_chunk", "").strip() for match in pinecone_result.matches]

    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        return []

# ✅ Function to generate chatbot responses using GPT-4 with retrieved context
async def generate_response(query: str, top_k: int = 5):
    """Generate chatbot responses from GPT-4 with retrieved context."""
    context_chunks = await retrieve_chunks(query, top_k=top_k)

    if not context_chunks:
        return "I couldn't find relevant information. Could you rephrase your question or provide more details?"

    combined_context = "\n\n---\n\n".join(context_chunks)

    system_message = (
        "You are UniEase, a university chatbot providing accurate and concise answers to student inquiries. "
        "Use retrieved knowledge to generate structured responses. If information is insufficient, acknowledge the limitation."
    )

    user_prompt = (
        f"User Query:\n{query}\n\n"
        "Relevant Context:\n"
        f"{combined_context}\n\n"
        "Generate an appropriate response."
    )

    try:
        chat_response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        return chat_response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Error generating GPT response: {e}")
        return "Oops, something went wrong."

# ✅ Function to process all queries and generate chatbot responses
async def process_queries():
    """Process all queries asynchronously and save results to a CSV file."""
    df["Generated Response"] = await asyncio.gather(*[generate_response(q) for q in df["Query"]])

    # Save output to a CSV file
    output_csv_path = "C:/Users/osato/Downloads/test_dataset_fixed.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"✅ Generated responses saved to: {output_csv_path}")

# ✅ Run the async function properly in Jupyter
if __name__ == "__main__":
    asyncio.run(process_queries())  # Ensures proper execution in a Python script
