
import tensorflow as tf

from src.data.load_data import prepare_dataset

X_train, Y_train, X_test, Y_test, X_val, Y_val = prepare_dataset("../data/raw/abalone.data")


# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1280, activation="relu"),
    tf.keras.layers.Dense(400, activation="relu"),
    tf.keras.layers.Dense(30, activation="softmax")
])

# Compile
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# Feed dataset to TensorFlow
model.fit(X_train, Y_train, epochs=10, batch_size=32)

print("Evaluate123")
# Evaluate
model.evaluate(X_test, Y_test)

