from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_autoencoder(input_dim, latent_dim=24):

    input_layer = Input(shape=(input_dim,))

    # ----- Encoder -----
    encoded = Dense(128, activation="relu")(input_layer)
    encoded = BatchNormalization()(encoded)

    encoded = Dense(64, activation="relu")(encoded)
    encoded = BatchNormalization()(encoded)

    encoded = Dense(32, activation="relu")(encoded)
    encoded = Dropout(0.2)(encoded)

    # Bottleneck (LATENT SPACE)
    bottleneck = Dense(latent_dim, activation="relu")(encoded)

    # ----- Decoder -----
    decoded = Dense(32, activation="relu")(bottleneck)
    decoded = Dense(64, activation="relu")(decoded)
    decoded = Dense(128, activation="relu")(decoded)

    output_layer = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    autoencoder.compile(
        optimizer=Adam(learning_rate=0.0008),
        loss="mse"
    )

    return autoencoder