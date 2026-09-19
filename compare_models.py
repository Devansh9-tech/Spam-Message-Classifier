import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ---- Load and clean ----
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

# ---- Define models to compare ----
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': LinearSVC(max_iter=2000)
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    })
    trained_models[name] = model

    print(f"=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print()

# ---- Comparison table ----
results_df = pd.DataFrame(results)
print("=== Comparison Table ===")
print(results_df.to_string(index=False))

# ---- Pick best model by F1 score ----
best_row = results_df.loc[results_df['F1'].idxmax()]
best_name = best_row['Model']
best_model = trained_models[best_name]

print(f"\n=== Best Model: {best_name} (F1: {best_row['F1']:.4f}) ===")

# ---- Save best model + vectorizer (overwrites Day 4 files) ----
joblib.dump(best_model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print(f"Saved {best_name} as spam_model.pkl (this replaces the Day 4 Naive Bayes model)")