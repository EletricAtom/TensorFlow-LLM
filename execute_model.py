import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import numpy as np

# **Load the trained model**
model = load_model("llm_model.h5")
print("Model loaded!")

# **Load the vocabulary**
with open("vocabulary.pkl", "rb") as f:
    vocabulary = pickle.load(f)
print("Vocabulary loaded!")

# Recreate the TextVectorization layer using the loaded vocabulary
vectorizer = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=10)
vectorizer.set_vocabulary(vocabulary)  # Set the vocabulary manually using set_vocabulary()

# **Function to generate text**
def generate_text(seed_text, model, vectorizer, vocabulary, max_length=2):
    for _ in range(max_length):
        # Convert seed_text into tokens using the vectorizer
        tokens = vectorizer([seed_text])
        tokens = tf.expand_dims(tokens, axis=0)  # Expand dimensions to match model input shape
        prediction = model.predict(tokens)  # Predict the next token
        next_word_index = np.argmax(prediction, axis=-1)  # Find the index of the next token
        next_word = vocabulary[next_word_index[0]]  # Retrieve the word from the vocabulary
        seed_text += " " + next_word  # Add the predicted word to the seed text
    return seed_text

# **Test text generation**
seed_text = "Name"
generated_text = generate_text(seed_text, model, vectorizer, vocabulary)
print("Generated text:", generated_text)
