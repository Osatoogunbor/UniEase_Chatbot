import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the test dataset (Ensuring we use the correct file)
df = pd.read_csv("test_dataset_fixed.csv")

# ✅ Initialize OpenAI client
OPENAI_API_KEY="openai_api_key"

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Function to generate OpenAI embeddings
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",  # Best model for semantic similarity
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# ✅ Function to compute cosine similarity (used for semantic similarity)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ✅ Compute similarity scores
similarity_scores = []
for index, row in df.iterrows():
    expected_text = row["Expected Response"]
    generated_text = row["Generated Response"]

    expected_embedding = get_embedding(expected_text)
    generated_embedding = get_embedding(generated_text)

    if expected_embedding is not None and generated_embedding is not None:
        similarity = cosine_similarity(expected_embedding, generated_embedding)
    else:
        similarity = None

    similarity_scores.append(similarity)

# ✅ Add similarity scores to DataFrame
df["Semantic Similarity Score"] = similarity_scores

# ✅ Compute Cumulative Statistics
average_similarity = np.nanmean(similarity_scores)  # Compute average
min_similarity = np.nanmin(similarity_scores)  # Minimum score
max_similarity = np.nanmax(similarity_scores)  # Maximum score
std_dev_similarity = np.nanstd(similarity_scores)  # Standard deviation
median_similarity = np.nanmedian(similarity_scores)  # Compute median

# Create a histogram with KDE
plt.figure(figsize=(8, 5))
sns.histplot(similarity_scores, bins=10, kde=True, color="blue", alpha=0.7)

# Labels and title
plt.xlabel("Semantic Similarity Score")
plt.ylabel("Frequency")
plt.title("Distribution of Semantic Similarity Scores with KDE")
plt.show()

# ✅ Print Cumulative Results
print("\n📊 **Semantic Similarity Test Summary**")
print(f"✅ **Average Similarity Score:** {average_similarity:.4f}")
print(f"🔻 **Minimum Similarity Score:** {min_similarity:.4f}")
print(f"🔺 **Maximum Similarity Score:** {max_similarity:.4f}")
print(f"📉 **Standard Deviation:** {std_dev_similarity:.4f}")
print(f"📊 **Median Similarity Score:** {median_similarity:.4f}")  # ✅ Correctly added

# ✅ Save results to CSV
df.to_csv("test_dataset_with_semantic_similarity.csv", index=False)
print("\n✅ All scores saved to 'test_dataset_with_semantic_similarity.csv'")
