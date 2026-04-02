import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data.load_data import prepare_dataset

X_train, Y_train, X_test, Y_test, X_val, Y_val = prepare_dataset(data="../../data/raw/abalone.data")

# --- Callback для автоматической остановки ---
early_stop = EarlyStopping(
    monitor='mae',
    patience=10,
    restore_best_weights=True
)

# --- Callback для динамического изменения LR ---
reduce_lr = ReduceLROnPlateau(
#    monitor='val_loss',
#    monitor='loss',
    monitor='mae',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)


# --- Функция для построения модели ---
def build_model(hp):
    model = keras.Sequential()

    # Подбор числа нейронов
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'
    ))

    # Еще один скрытый слой (по желанию)
    model.add(keras.layers.Dense(
        units=hp.Int('units2', min_value=32, max_value=256, step=32),
        activation='relu'
    ))

    model.add(keras.layers.Dense(1))

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dense(1)
    # ])


    # Подбор learning rate
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    return model


# --- Настройка Keras Tuner ---
tuner = kt.RandomSearch(
    build_model,
#    objective='mae',
    objective='val_accuracy',
    max_trials=10,  # Количество комбинаций гиперпараметров
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='auto_epochs_tuning'
)

# --- Поиск оптимальных гиперпараметров ---
tuner.search(
    X_train, Y_train,
    validation_split=0.2,
    epochs=1000,  # Можно задать очень большое число
    batch_size=32,  # Можно пробовать также через hp
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# --- Лучшие гиперпараметры ---
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]

print("Лучшие гиперпараметры:")
print(f"units: {best_hp.get('units')}")
print(f"units2: {best_hp.get('units2')}")
print(f"learning_rate: {best_hp.get('learning_rate')}")


best_model.fit(X_train, Y_train, epochs=1000, batch_size=32, callbacks=[early_stop])

print("Evaluation:")
# Evaluate
best_model.evaluate(X_test, Y_test)

#20/20 ━━━━━━━━━━━━━━━━━━━━ 0s 820us/step - loss: 4.3961 - mae: 1.5058
