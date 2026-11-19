import os

import shap
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline

from text_cleaning import clean_text

# Where the trained model is: a local folder, or a Hugging Face Hub name like "your-username/misinformation-detector".
# Set the MODEL_PATH environment variable (or Streamlit secret) to change it.
MODEL_PATH = os.environ.get("MODEL_PATH", "fake_news_model")
MAX_CHARS = 400      # keep posts short so the explanation stays fast
MAX_TOKENS = 128

st.set_page_config(page_title="Misinformation Detector", page_icon="🔎", layout="centered")

st.title("🔎 Misinformation Detector")
st.markdown("Paste a short social-media post in English. The model gives a label, a confidence score, "
            "and highlights the words that influenced it.")
st.caption("Trained on English COVID-19 posts, so it may not work well on other topics or languages.")
st.caption("This is a demo model. A wrong or uncertain result is possible, so never treat it as a fact-check.")
st.divider()


@st.cache_resource(show_spinner="Loading model... (first time only)")
def load_model():
    # Loaded once and reused, because loading a transformer takes a while
    classifier = pipeline("text-classification", model=MODEL_PATH, top_k=None,
                          truncation=True, max_length=MAX_TOKENS)
    explainer = shap.Explainer(classifier)
    return classifier, explainer


text = st.text_area("Post text", height=140, max_chars=MAX_CHARS,
                    placeholder="Paste a tweet or comment here...")

if st.button("Analyze", type="primary"):
    cleaned = clean_text(text)
    if not cleaned:
        st.warning("Please enter some text first.")
        st.stop()

    classifier, explainer = load_model()

    # 1) Prediction
    scores = sorted(classifier(cleaned)[0], key=lambda s: s["score"], reverse=True)
    top = scores[0]
    st.subheader(f"Prediction: {top['label']}")
    st.metric("Confidence", f"{top['score'] * 100:.1f}%")
    for s in scores:
        st.progress(float(s["score"]), text=f"{s['label']}: {s['score'] * 100:.1f}%")

    with st.expander("Text the model saw (after cleaning)"):
        st.write(cleaned)

    # 2) Explanation with SHAP (if it fails, the prediction above is still shown)
    st.subheader("Which words drove this prediction?")
    try:
        with st.spinner("Explaining..."):
            n_tokens = len(classifier.tokenizer(cleaned, truncation=True, max_length=MAX_TOKENS)["input_ids"])
            shap_values = explainer([cleaned], max_evals=max(300, 4 * n_tokens))
            html = shap.plots.text(shap_values[0, :, top["label"]], display=False)
            html = f'<div style="background:white; padding:10px">{html}</div>'

        st.caption(f"Red words push toward '{top['label']}', blue words push away from it.")
        if hasattr(st, "iframe"):
            st.iframe(html, height=260)
        else:
            components.html(html, height=260, scrolling=True)
    except Exception as error:
        st.info(f"The word-level explanation is not available right now ({type(error).__name__}).")
