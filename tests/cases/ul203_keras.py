from sklearn.model_selection import train_test_split
from tensorflow import keras

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = keras.Sequential([keras.layers.Dense(64, activation="relu"), keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, callbacks=[stop])  # expect: UL203
print(model.evaluate(X_test, y_test))
