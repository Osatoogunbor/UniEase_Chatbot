import pandas as pd
import numpy as np
import statistics  # for median, stdev
from openai import OpenAI
from pinecone import Pinecone
import matplotlib.pyplot as plt

# 1) Load your labeled data
df = pd.read_csv("labeled_test_queries.csv")

# 2) Configure OpenAI & Pinecone
client = OpenAI(
    api_key="openai_api_key"

    )
pc = Pinecone(api_key="pinecone_api_key",
              environment="us-east-1-aws")
index = pc.Index("ai-powered-chatbot")

K = 3

# We'll store per-query precision and recall so we can compute mean/median/stdev
precision_list = []  # binary: 1 if any relevant in top-K, else 0
recall_list = []  # fraction: (# relevant in top-K) / (total relevant labeled)

for i, row in df.iterrows():
    query = str(row["Query"]).strip()
    rel_chunk_str = str(row["RelevantChunkIDs"]).strip()

    # Skip if user labeled 'None' or empty
    if rel_chunk_str.lower() == "none" or not rel_chunk_str:
        continue

    # Possibly multiple relevant IDs, e.g. "doc-123, doc-456"
    relevant_ids = [x.strip() for x in rel_chunk_str.split(",")]
    num_relevant = len(relevant_ids)

    # If there's no relevant chunk, skip
    if num_relevant == 0:
        continue

    # 3) Get embedding (sync call)
    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[query]
        )
    except Exception as e:
        print(f"Error calling OpenAI Embeddings API: {e}")
        continue

    emb = emb_resp.data[0].embedding

    # 4) Query Pinecone
    try:
        result = index.query(vector=emb, top_k=K, include_metadata=True)
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        continue

    # If no matches returned, skip
    if not result or not result.matches:
        continue

    returned_ids = [m.id for m in result.matches]

    # -- Evaluate for this query --
    # A) Precision (binary): 1 if ANY relevant chunk is in top-K
    if any(rid in returned_ids for rid in relevant_ids):
        precision_list.append(1.0)
    else:
        precision_list.append(0.0)

    # B) Recall: fraction of relevant IDs that appear in top-K
    found_count = sum(rid in returned_ids for rid in relevant_ids)
    query_recall = found_count / num_relevant
    recall_list.append(query_recall)

# 5) Compute Statistics
if len(precision_list) == 0:
    print("No labeled queries to evaluate.")
else:
    mean_precision = np.mean(precision_list)
    median_precision = np.median(precision_list)
    std_precision = np.std(precision_list)

    mean_recall = np.mean(recall_list)
    median_recall = np.median(recall_list)
    std_recall = np.std(recall_list)

    print(f"Precision@{K}:")
    print(f"  Mean = {mean_precision:.3f}")
    print(f"  Median = {median_precision:.3f}")
    print(f"  Std Dev = {std_precision:.3f}\n")

    print(f"Recall@{K}:")
    print(f"  Mean = {mean_recall:.3f}")
    print(f"  Median = {median_recall:.3f}")
    print(f"  Std Dev = {std_recall:.3f}")

    # 6) Visualize (Optional)
    # e.g., histogram of recall
    plt.hist(recall_list, bins=10, range=(0, 1), alpha=0.7, color='blue')
    plt.title(f"Recall@{K} Distribution")
    plt.xlabel("Recall Score")
    plt.ylabel("Frequency")
    plt.show()

# Assuming the following variables are defined from your previous code:
# mean_precision, std_precision, mean_recall, std_recall, and K

# Define labels and corresponding values
metrics = [f"Precision@{K}", f"Recall@{K}"]
mean_values = [mean_precision, mean_recall]
std_values = [std_precision, std_recall]

plt.figure(figsize=(6, 4))
bars = plt.bar(metrics, mean_values, yerr=std_values, capsize=10, color=['green', 'blue'])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"Average Precision and Recall at {K}")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Annotate the bars with their mean values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom')

plt.show()
