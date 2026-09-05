import streamlit as st
import joblib

st.set_page_config(page_title="AI Resume Classifier", page_icon="📄")

@st.cache_resource
def load_model():
    return joblib.load("resume_classifier.joblib")

st.title("AI Resume Classification System")
st.write("Paste resume text below to predict its most likely job category.")

resume_text = st.text_area(
    "Resume text",
    height=260,
    placeholder="Paste resume content here..."
)

if st.button("Classify Resume"):
    if not resume_text.strip():
        st.warning("Please enter resume text before classification.")
    else:
        model = load_model()
        prediction = model.predict([resume_text])[0]
        probabilities = model.predict_proba([resume_text])[0]

        st.subheader(f"Predicted Category: {prediction}")
        st.write(f"Confidence: **{max(probabilities):.2%}**")

        st.subheader("Category Probabilities")
        for label, probability in sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        ):
            st.write(f"**{label}:** {probability:.2%}")
            st.progress(int(probability * 100))
