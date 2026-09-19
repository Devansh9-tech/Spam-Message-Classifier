import streamlit as st
import joblib
import re

# ---- Load saved model and vectorizer ----
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---- Page config ----
st.set_page_config(page_title="Spam Message Classifier", page_icon="🛡️", layout="centered")

st.title("🛡️ Spam/Scam Message Classifier")
st.write("Paste an SMS or text message below to check if it's spam or legitimate.")

# ---- Input ----
user_input = st.text_area("Message text:", height=120, placeholder="e.g. Congratulations! You've won a free prize, click here to claim...")

if st.button("Classify", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # Linear SVM doesn't have predict_proba by default, use decision_function instead
        decision_score = model.decision_function(vectorized)[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 **SPAM DETECTED**")
        else:
            st.success("✅ **LEGITIMATE MESSAGE**")

        st.write(f"**Confidence score:** {abs(decision_score):.2f}")
        st.caption("Higher score = model is more confident in its decision. This is a distance-from-decision-boundary score (SVM), not a direct probability.")

        with st.expander("See cleaned text (what the model actually saw)"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

st.divider()
st.caption("Model: Linear SVM | Trained on SMS Spam Collection dataset (5,572 messages) | Precision: 96.4% | Recall: 90.0%")