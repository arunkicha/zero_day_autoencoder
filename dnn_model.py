from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_dnn(input_dim):

    input_layer = Input(shape=(input_dim,))

    x = Dense(128, activation="relu")(input_layer)
    x = Dropout(0.3)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation="relu")(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model