import tensorflow as tf

class Model:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def create(self, num_words, max_seq_len, num_classes):
        # Define the model architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(num_words, 128, input_length=max_seq_len),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

    def load(self, model_dir='./models'):
        # Load the tokenizer
        tokenizer_path = f'{model_dir}/tokenizer'
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.word_index = tf.keras.preprocessing.text.tokenizer_from_json(open(f'{tokenizer_path}.json').read())

        # Load the model architecture and weights
        model_path = f'{model_dir}/model.h5'
        self.model = tf.keras.models.load_model(model_path)

    def save(self, model_dir='./models'):
        # Save the tokenizer
        tokenizer_path = f'{model_dir}/tokenizer'
        tokenizer_json = self.tokenizer.to_json()
        with open(f'{tokenizer_path}.json', 'w', encoding='utf-8') as f:
            f.write(tokenizer_json)

        # Save the model architecture and weights
        model_path = f'{model_dir}/model.h5'
        self.model.save(model_path)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        # Fit the model to the training data
        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_val, y_val),
                                 epochs=epochs,
                                 batch_size=batch_size)

        return history

    def predict(self, message):
        # Tokenize the message
        sequence = self.tokenizer.texts_to_sequences([message])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=self.tokenizer.max_len)

        # Make the prediction
        prediction = self.model.predict(padded_sequence)
        return prediction