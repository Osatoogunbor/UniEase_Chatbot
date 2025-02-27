import openai
import pandas as pd
from pinecone import Pinecone
import matplotlib.pyplot as plt

# 1) Load test dataset (Queries and Generated Responses)
df = pd.read_csv("C:/Users/osato/Downloads/test_dataset_fixed.csv")

# 2) Initialize OpenAI client
OPENAI_API_KEY = "openai_api_key"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 3) Initialize Pinecone client
PINECONE_API_KEY = "pcsk_4MmQQE_QVvEdpRKLtekQvxKHp9P5QUEWZe7NMzRvdZzXXheWoxTEkTYFYTqYQCYmUop3RP"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("ai-powered-chatbot")  # Replace with your actual index name

# 4) Function to get embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# 5) Function to retrieve knowledge chunks
def retrieve_chunks(query: str, top_k=10):
    """
    Retrieve relevant knowledge from Pinecone.
    Increased top_k from 5 to 10 to reduce missing relevant info.
    """
    try:
        query_vector = get_embedding(query)
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        # Join all retrieved chunks
        return "\n".join([match.metadata.get("text_chunk", "").strip() for match in response.matches])
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        return "No knowledge retrieved."

# 6) Hallucination check function
def detect_hallucination(query, retrieved_context, generated_response):
    """
    Checks if the chatbot response contains hallucinated information.
    Returns: "Yes" or "No" and a short explanation.
    """

    prompt = f"""
    You are evaluating a university chatbot's response.
    The chatbot must generate responses based on the retrieved knowledge.
    **Criteria**:
    - If the response is factually correct but reworded, mark "No" (not hallucination).
    - If the chatbot adds incorrect info not in the retrieved context, mark "Yes" (hallucination).
    - Paraphrasing or formatting differences should not count as hallucinations.

    **User Query**: {query}
    **Retrieved Context**: {retrieved_context}
    **Generated Response**: {generated_response}

    Provide your output EXACTLY in the following format (no extra text):

    ```
    Hallucination: [Yes or No]
    Explanation: [One short sentence]
    ```
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert, strict evaluator. Follow the exact format below."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=50,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "Parsing failed"

# 7) Apply retrieval + hallucination test
retrieved_contexts = []
hallucination_labels = []
explanations = []

for _, row in df.iterrows():
    query = row["Query"]
    generated_response = row["Generated Response"]

    # Retrieve knowledge from Pinecone
    retrieved_context = retrieve_chunks(query, top_k=10)

    # Detect hallucination
    eval_result = detect_hallucination(query, retrieved_context, generated_response)

    # Parse the answer
    h_label = "Unknown"
    expl = "No explanation"

    lines = eval_result.splitlines()
    for line in lines:
        line = line.strip().replace("```", "")
        if line.startswith("Hallucination:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                h_label = parts[1].strip()
        elif line.startswith("Explanation:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                expl = parts[1].strip()

    retrieved_contexts.append(retrieved_context)
    hallucination_labels.append(h_label)
    explanations.append(expl)

# 8) Add results to DataFrame
df["Retrieved Context"] = retrieved_contexts
df["Hallucination Detected"] = hallucination_labels
df["Hallucination Explanation"] = explanations

# 9) Save results
df.to_csv("test_dataset_with_hallucination_check.csv", index=False)
print("✅ Hallucination detection completed. Results saved.")

# 10) **Fix summary and create the pie chart**
# Create summary DataFrame by counting hallucination results
summary_df = df["Hallucination Detected"].value_counts().reset_index()
summary_df.columns = ["Hallucination", "Count"]

# Plot pie chart
plt.figure(figsize=(6, 6))
wedges, texts, autotexts = plt.pie(
    summary_df["Count"],
    labels=summary_df["Hallucination"],
    autopct='%1.1f%%',
    startangle=90,
    colors=["red", "green", "orange"]  # Adjust colors as desired
)

# Create a white circle at the center to turn the pie into a donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Donut Chart of Hallucination Detection Distribution")
plt.axis('equal')  # Ensure the chart is drawn as a circle
plt.show()
