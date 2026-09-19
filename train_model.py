import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# ---- Load and clean  ----
df = pd.read_csv('spam.csv', sep='\t', header=None, names=['label', 'message'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'],
    test_size=0.2, random_state=42, stratify=df['label_num']
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---- Training the model ----
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---- Predicting on test set ----
y_pred = model.predict(X_test_tfidf)

# ---- Evaluating ----
print("=== Accuracy ===")
print(f"{accuracy_score(y_test, y_pred):.4f}")

print("\n=== Precision (of predicted spam, how much was actually spam) ===")
print(f"{precision_score(y_test, y_pred):.4f}")

print("\n=== Recall (of actual spam, how much did we catch) ===")
print(f"{recall_score(y_test, y_pred):.4f}")

print("\n=== F1 Score (balance of precision and recall) ===")
print(f"{f1_score(y_test, y_pred):.4f}")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(f"                Predicted Ham   Predicted Spam")
print(f"Actual Ham      {cm[0][0]:<15} {cm[0][1]}")
print(f"Actual Spam     {cm[1][0]:<15} {cm[1][1]}")

print("\n=== Full Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# ---- Testing on custom messages ----
custom_messages = [
    "Congratulations! You've won a free iPhone, click here to claim now!",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT: Your account will be suspended. Verify now at this link.",
    "Can you send me the notes from class today?"
]
custom_clean = [clean_text(msg) for msg in custom_messages]
custom_tfidf = vectorizer.transform(custom_clean)
custom_pred = model.predict(custom_tfidf)
custom_proba = model.predict_proba(custom_tfidf)

print("\n=== Custom Message Predictions ===")
for msg, pred, proba in zip(custom_messages, custom_pred, custom_proba):
    label = "SPAM" if pred == 1 else "HAM"
    confidence = proba[1] if pred == 1 else proba[0]
    print(f"[{label} - {confidence:.2%} confident] {msg}")

# ---- Save model and vectorizer for later use  ----
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nModel and vectorizer saved to spam_model.pkl and vectorizer.pkl")