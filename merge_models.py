import numpy as np
import os
from tensorflow import keras
from keras import layers

# Get the list of models in the models folder.
models = []
for file in os.listdir("models"):
    if file.endswith(".h5"):
        models.append(keras.models.load_model("models/" + file))

# Create a meta-model.
meta_model = keras.models.Sequential()
meta_model.add(layers.Dense(128, activation="relu"))
meta_model.add(layers.Dense(1, activation="sigmoid"))

# Train the meta-model.
meta_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# meta_model.fit(np.array(predictions), y_test, epochs=100)

# Save the meta-model.
# Save the new model object to a file.
meta_model.build(input_shape=models[0].input_shape)
meta_model.save("meta_model.h5")
print("Merged models and Saved It has 'meta_model.h5' to disk")
