
import streamlit as st
import pickle
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# 2. Load saved model, tokenizer and max_len
@st.cache_resource
def load_resources():

    model = load_model(
        str(BASE_DIR / "LSTM Model.h5"),
        compile=False
    )

    with open(BASE_DIR / "Tokenizer for Next Word Prediction.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(BASE_DIR / "max len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

# 3. Create reverse word mapping
index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}

# 4. Prediction function
def predict_next_word(text):

    sequence = tokenizer.texts_to_sequences([text])[0]

    if len(sequence) == 0:
        return "No known words found. Try different text."

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre",
        truncating="pre"
    )

    preds = model.predict(sequence, verbose=0)
    predicted_index = int(np.argmax(preds[0]))

    return index_to_word.get(
        predicted_index,
        "Predicted word not found"
    )


# 5. Streamlit UI
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the next word.")

user_input = st.text_input(
    "✍️ Enter text:",
    placeholder="Type a sentence here..."
)

if st.button("Predict Next Word", type="primary"):

    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        st.success(f"Predicted Next Word: **{next_word}**")

# 6. Footer
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")
