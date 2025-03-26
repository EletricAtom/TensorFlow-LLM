import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
import numpy as np
import pickle

# Open the file 'files.txt' for reading, with error handling for encoding issues
with open('files.txt', 'r', errors="replace") as file:
    dataset = file.read()

# Create an example dataset from the text file
dataset = tf.data.Dataset.from_tensor_slices([dataset])

# Configure the word vectorizer
vocab_size = 10000  # Vocabulary size (maximum number of unique words)
embedding_dim = 128  # Dimensionality of the word embeddings
sequence_length = 10  # Maximum length of the sequences (input sequence length)

# Create and train the vectorizer
vectorizer = TextVectorization(output_mode='int', output_sequence_length=sequence_length)
vectorizer.adapt(dataset.batch(32))  # Adapt the vectorizer to the dataset

# Save the vocabulary to a file
vocabulary = vectorizer.get_vocabulary()
print(f"Vocabulary size: {len(vocabulary)}")

# Save the vocabulary as a pickle file
with open("vocabulary.pkl", "wb") as f:
    pickle.dump(vocabulary, f)
print("Vocabulary saved as 'vocabulary.pkl'")

# Create the language model using a Sequential model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(256, return_sequences=True),  # First LSTM layer with 256 units
    LSTM(256),  # Second LSTM layer with 256 units
    Dense(128, activation="relu"),  # Fully connected layer with 128 units
    Dense(vocab_size, activation="softmax")  # Output layer with softmax for probability distribution
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Simulate some training data (tokens of sentences)
X_train = np.array([[1, 5, 8, 2, 0, 0, 0, 0, 0, 0], 
                    [4, 3, 9, 6, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

y_train = np.array([2, 6], dtype=np.int32)  # Ensure the target labels are in the correct format

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2)

# Save the trained model to a file
model.save("llm_model.h5")
print("Model saved as 'llm_model.h5'")
