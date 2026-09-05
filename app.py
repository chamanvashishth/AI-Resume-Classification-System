import streamlit as st

from model_utils import load_or_train_model

st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="centered",
)

st.title("AI Resume Classification System")
st.caption(
    "Paste resume text below and the model will predict the most likely job category."
)

st.info(
    "The first request may take a little longer because the model is trained "
    "automatically if a saved model is not available."
)

@st.cache_resource(show_spinner=False)
def get_model():
    return load_or_train_model()


resume_text = st.text_area(
    "Resume text",
    height=260,
    placeholder="Paste the text from a resume here...",
)

if st.button("Classify Resume", type="primary"):
    if not resume_text.strip():
        st.warning("Please enter resume text before classification.")
    else:
        try:
            with st.spinner("Preparing the model and analyzing the resume..."):
                model = get_model()
                prediction = model.predict([resume_text])[0]
                probabilities = model.predict_proba([resume_text])[0]

            confidence = max(probabilities)

            st.success(f"Predicted Category: {prediction}")
            st.metric("Model Confidence", f"{confidence:.2%}")

            st.subheader("Category Probabilities")

            for label, probability in sorted(
                zip(model.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            ):
                st.write(f"**{label}:** {probability:.2%}")
                st.progress(int(probability * 100))

            st.caption(
                "This confidence score reflects the model's certainty on the "
                "current demonstration dataset. It is not a hiring recommendation."
            )

        except Exception as error:
            st.error("The classifier could not be prepared.")
            st.exception(error)
