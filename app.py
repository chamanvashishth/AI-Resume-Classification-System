import streamlit as st

from model_utils import load_or_train_model

st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="centered",
)

st.title("AI Resume Classification System")
st.caption(
    "Paste resume text below to predict the job category it most closely matches."
)

@st.cache_resource(show_spinner=False)
def get_model():
    # The model is cached in memory. The app can train from scratch when no
    # saved .joblib file exists, so deployment does not depend on that file.
    return load_or_train_model(persist=False)


resume_text = st.text_area(
    "Resume text",
    height=260,
    placeholder="Paste the text from a resume here...",
    help="The app currently accepts plain text and supports four demonstration categories.",
)

if st.button("Classify Resume", type="primary"):
    cleaned_text = resume_text.strip()

    if not cleaned_text:
        st.warning("Please enter resume text before classification.")
    elif len(cleaned_text) < 30:
        st.warning("Please provide a little more resume text for a meaningful prediction.")
    else:
        try:
            with st.spinner("Preparing the model and analyzing the resume..."):
                model = get_model()
                prediction = model.predict([cleaned_text])[0]
                probabilities = model.predict_proba([cleaned_text])[0]

            confidence = float(max(probabilities))

            st.success(f"Predicted Category: {prediction}")
            st.metric("Model Confidence", f"{confidence:.2%}")

            st.subheader("Category Probabilities")
            ranked_probabilities = sorted(
                zip(model.classes_, probabilities),
                key=lambda item: float(item[1]),
                reverse=True,
            )

            for label, probability in ranked_probabilities:
                probability = float(probability)
                st.write(f"**{label}:** {probability:.2%}")
                st.progress(int(round(probability * 100)))

            st.caption(
                "The confidence score reflects certainty within this demonstration "
                "model. It should not be used as a hiring or candidate-selection decision."
            )

        except Exception:
            st.error(
                "The classifier could not be started. Please refresh the app and try again."
            )
            st.caption(
                "If the problem continues, check the deployment logs for the full error."
            )
