import tensorflow as tf
import numpy as np

from src.data.load_data import prepare_dataset

X_train, Y_train, X_test, Y_test, X_val, Y_val = prepare_dataset(data="../../data/raw/abalone.data")




baseline = np.mean(Y_train)
baseline_preds = np.full_like(Y_test, baseline)

mae_baseline = np.mean(np.abs(Y_test - baseline_preds))
print("Baseline: ", mae_baseline)


# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='mae',
    patience=10,
    restore_best_weights=True
)

# Compile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# Feed dataset to TensorFlow
model.fit(X_train, Y_train, epochs=100, batch_size=32, callbacks=[early_stop])

print("Evaluate123")
# Evaluate
model.evaluate(X_test, Y_test)

