import tensorflow as tf
import time


batch_size = 32
input_shape = (batch_size, 224, 224, 3)
x = tf.random.normal(input_shape)
y = tf.random.uniform((batch_size,), maxval=1000, dtype=tf.int32)

model = tf.keras.applications.ResNet50(weights=None)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = loss_fn(y, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Warm-up
for _ in range(10):
    train_step(x, y)

# Benchmark
steps = 100
start = time.time()

for _ in range(steps):
    train_step(x, y)

end = time.time()

total_time = end - start
throughput = (steps * batch_size) / total_time

print(f"Throughput: {throughput:.2f} samples/sec")
print(f"Time per step: {total_time/steps:.4f} sec")

start = time.time()
for _ in range(100):
    model(x, training=False)
end = time.time()

latency = (end - start) / 100
print(f"Latency per batch: {latency:.6f} sec")