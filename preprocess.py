import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove numbers, punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # collapse multiple spaces
    return text

df['clean_message'] = df['message'].apply(clean_text)

print("=== Before vs After cleaning (first 5) ===")
for i in range(5):
    print(f"Original: {df['message'].iloc[i]}")
    print(f"Cleaned:  {df['clean_message'].iloc[i]}")
    print()

# Convert labels to 0/1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into train/test BEFORE vectorizing (avoids data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'],
    test_size=0.2, random_state=42, stratify=df['label_num']
)

print(f"=== Split sizes ===")
print(f"Train: {X_train.shape[0]} messages")
print(f"Test:  {X_test.shape[0]} messages")

# TF-IDF: converts text into weighted word-importance vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\n=== TF-IDF matrix shapes ===")
print(f"Train matrix: {X_train_tfidf.shape}")
print(f"Test matrix:  {X_test_tfidf.shape}")

print(f"\n=== Sample of vocabulary learned (first 20 words) ===")
print(list(vectorizer.vocabulary_.keys())[:20])