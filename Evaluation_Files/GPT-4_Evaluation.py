import pandas as pd
import openai
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
#gpt test evaluation

# === 1) Initialize OpenAI client in your style ===
OPENAI_API_KEY="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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)  # <-- Same approach you used before


df = pd.read_csv("C:/Users/osato/Downloads/test_dataset_fixed.csv")
# Ensure "Generated Response" column exists and fill missing
df["Generated Response"] = df["Generated Response"].fillna("No response available")

# === 3) GPT-4 EVALUATION FUNCTION (TWEAKED) ===
def evaluate_response_strict(query, generated):
    """
    - Hides the expected response from GPT-4
    - Encourages stricter, more critical evaluation
    - Score range: 1-5
    """

    prompt = f"""
    You are an expert, STRICT chatbot evaluator. 
    You must be very critical in your judgment of the response.

    Provide your evaluation in THIS exact format (no extra text):

    ```
    Factual Accuracy: [Yes/No]
    Alignment with Query: [Yes/No]
    Tone Appropriateness: [Yes/No]
    Score: [1-5]
    Justification: [Brief justification]
    ```

    Detailed Instructions:
    1. **Factual Accuracy (Yes/No)**: Is the response factually correct based on general knowledge or typical university info?
       - If it contains ANY incorrect claims, say "No".
    2. **Alignment with Query (Yes/No)**: Does it address the user's question?
       - If it partially or fully ignores the query, say "No".
    3. **Tone Appropriateness (Yes/No)**: Is the tone suitable for a university student?
       - If it's rude or overly casual, say "No".
    4. **Overall Score (1-5)**:
       - 5 = Perfect correctness & alignment, very appropriate tone
       - 4 = Mostly correct, minor issues
       - 3 = Some correctness but noticeable flaws
       - 2 = Significant factual or alignment issues
       - 1 = Very poor or completely incorrect
    5. **Justification**: Provide a brief 1-2 sentence justification.

    USER QUERY: {query}
    CHATBOT'S RESPONSE: {generated}

    Important: 
    - DO NOT see or rely on an "Expected Response" because it is hidden. 
    - Evaluate critically based on your general knowledge.
    - Output ONLY in the exact code block format above.
    """

    try:
        # Use your older-style approach: client.chat.completions.create
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert, strict chatbot evaluator. "
                        "Follow the specified format exactly. Be thorough and critical."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "Evaluation failed."

# === 4) APPLY EVALUATION ===
df["GPT-4 Strict Evaluation"] = df.apply(
    lambda row: evaluate_response_strict(row["Query"], row["Generated Response"]),
    axis=1
)

# === 5) PARSE EVALUATION ===
def parse_strict_evaluation(evaluation_text):
    """
    Extract fields from GPT-4's stricter format:
    ```
    Factual Accuracy: [Yes/No]
    Alignment with Query: [Yes/No]
    Tone Appropriateness: [Yes/No]
    Score: [1-5]
    Justification: ...
    ```
    """
    try:
        factual_accuracy = re.search(r"Factual Accuracy: (Yes|No)", evaluation_text).group(1)
        alignment = re.search(r"Alignment with Query: (Yes|No)", evaluation_text).group(1)
        tone = re.search(r"Tone Appropriateness: (Yes|No)", evaluation_text).group(1)
        score = int(re.search(r"Score: (\d+)", evaluation_text).group(1))
        justification_match = re.search(r"Justification:\s*(.*)", evaluation_text, re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else "No justification found"
        return factual_accuracy, alignment, tone, score, justification
    except:
        return "Unknown", "Unknown", "Unknown", None, "Parsing failed"

df[["Factual Accuracy", "Alignment", "Tone", "Score", "Justification"]] = df["GPT-4 Strict Evaluation"].apply(
    lambda x: pd.Series(parse_strict_evaluation(x))
)

# Convert Yes/No to 1/0 (optional)
yes_no_map = {"Yes": 1, "No": 0}
df["Factual Accuracy"] = df["Factual Accuracy"].map(yes_no_map)
df["Alignment"] = df["Alignment"].map(yes_no_map)
df["Tone"] = df["Tone"].map(yes_no_map)

# === 6) SUMMARY STATS & VISUALIZATION ===
evaluation_summary = df[["Factual Accuracy", "Alignment", "Tone", "Score"]].describe()
print("\n📊 **GPT-4 Strict Evaluation Summary:**")
print(evaluation_summary)

df.to_csv("evaluated_responses_strict.csv", index=False)
print("\n✅ Strict evaluation completed and results saved to 'evaluated_responses_strict.csv'.")

# Optional: Plot a histogram of the new 1-5 scores
plt.hist(df["Score"].dropna(), bins=5, range=(1,6), color="blue", alpha=0.7)
plt.xlabel("GPT-4 Strict Evaluation Score (1-5)")
plt.ylabel("Frequency")
plt.title("Distribution of GPT-4 Strict Evaluation Scores")
plt.show()

plt.figure(figsize=(6, 4))
sns.barplot(x=accuracy_counts.index, y=accuracy_counts.values, palette=["red", "green"])
plt.xticks(ticks=[0, 1], labels=["No (Incorrect)", "Yes (Correct)"])
plt.xlabel("Factual Accuracy")
plt.ylabel("Number of Responses")
plt.title("Factual Accuracy Distribution")
plt.show()
