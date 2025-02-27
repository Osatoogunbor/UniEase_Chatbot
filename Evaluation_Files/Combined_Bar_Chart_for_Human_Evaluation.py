import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file (replace 'data.csv' with your actual file name)
data = pd.read_csv("C:/Users/osato/openai_setup/Academic Stress Support Chatbot_ User Feedback Form (Responses) - Form responses 1.csv")

# Define the survey rating columns
rating_cols = [
    'How helpful did you find the chatbot\n(1 = Not helpful at all, 5 = very helpful)',
    "How natural and human-like were the chatbot's responses?\n(1 = Not natural at all, 5 = very natural)",
    "Did the chatbot provide relevant and useful academic stress management advice?\n(1 = Not relevant and not helpful at all, 5 = very relevant and helpful)",
    "Did the chatbot's tone feel supportive and empathetic?\n(1 = Not supportive at all, 5 = very supportive)",
    "Did you feel more relieved or reassured after interacting with the chatbot?\n(1 = Not relieved at all, 5 = very relieved and reassured)"
]

# Convert rating columns to numeric
for col in rating_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Create histograms for each numeric rating column
data[rating_cols].hist(bins=15, figsize=(15, 10))
plt.suptitle("Histograms of Numeric Rating Columns")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("histograms.png")  # Save the figure
plt.show()
