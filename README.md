# Complete-Sentence-Prediction-using-LSTM-
LSTM-based language model for next-word prediction using Keras Sequential architecture and word embeddings, trained on public-domain text from A room with a view
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import requests
url = "https://www.gutenberg.org/cache/epub/2641/pg2641.txt"
response = requests.get(url)
text = response.text.lower()
corpus = text.split('\n')[:1000]
tokenizer = Tokenizer()  # Initialize the tokenizer
tokenizer.fit_on_texts(corpus)  # Learn the vocabulary from the corpus
total_words = len(tokenizer.word_index) + 1  # Total number of unique words
print(f"Total Unique Words: {total_words}")

# Create input-output pairs
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]  # Convert line to sequence of IDs
    for i in range(1, len(token_list)):  # Generate n-grams
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Expected output:
# - A list of n-gram sequences
# Example (for the line "the rabbit ran"):
# Pad sequences to ensure all inputs have the same length
max_len = max([len(seq) for seq in input_sequences])  # Maximum sequence length
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Expected output:
# - Padded sequences with zeros at the start
# Example:
# Original sequence: [1, 123]
# Padded sequence: [0, 0, 1, 123]

# Split data into input (X) and output (y)
X = input_sequences[:, :-1]  # All but the last word
y = input_sequences[:, -1]   # Last word as the output

# One-hot encode the output labels
y = np.eye(total_words)[y]  # Convert word IDs to one-hot vectors

# Expected output:
# X => Padded input sequences (2D array)
# y => One-hot encoded output (2D array)
# Example:
# X[0] => [0, 0, 1]
# y[0] => [0, 0, 0, 1, 0, ...]
# Step 3: Build the Model
model = Sequential([
    Embedding(input_dim=total_words, output_dim=100, input_length=max_len-1),  # Word embedding
    LSTM(150, return_sequences=False),  # Learn patterns in sequences
    Dense(total_words, activation='softmax')  # Predict next word
])


#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()



# Compile with gradient clipping
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#TRAIN THE MODEL
model.fit(X, y,epochs=100, batch_size=128,verbose=1)

Step 5: Sentence Completion Function
def complete_sentence(seed_text, num_words):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_probs[0][0] = 0   # avoid padding token only

        predicted_word_index = np.argmax(predicted_probs)
        output_word = tokenizer.index_word.get(predicted_word_index)

        if output_word is None:
            print("Unknown index:", predicted_word_index)
            break

        seed_text += " " + output_word

    return seed_text

Step 6: Test the Model for Sentence Completion
seed_text = "the ladies"
completed_sentence = complete_sentence(seed_text, num_words=10)

print(f"Seed: {seed_text}\nCompleted Sentence: {completed_sentence}")

