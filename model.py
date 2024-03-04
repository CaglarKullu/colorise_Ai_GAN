from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def build_generator():
    inputs = Input(shape=(32, 32, 1))

    # Encoder
    c1 = Conv2D(32, kernel_size=3, strides=1, padding="same")(inputs)
    c1 = LeakyReLU(alpha=0.2)(c1)
    c1 = BatchNormalization()(c1)
    
    c2 = Conv2D(64, kernel_size=3, strides=2, padding="same")(c1)
    c2 = LeakyReLU(alpha=0.2)(c2)
    c2 = BatchNormalization()(c2)
    
    # Decoder
    c3 = Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(c2)
    c3 = LeakyReLU(alpha=0.2)(c3)
    c3 = BatchNormalization()(c3)
    
    # Skip Connection
    c3 = Concatenate()([c3, c1])
    
    # Output layer
    outputs = Conv2D(3, kernel_size=3, padding="same", activation="tanh")(c3)  # Ensure this outputs 32x32x3
    
    model = Model(inputs, outputs)
    return model


def build_discriminator():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    # Ensure the number of units in the first Dense layer matches the flattened output
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model