from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input, Flatten, Dense
from tensorflow.keras.models import Model

def build_generator():
    inputs = Input(shape=(32, 32, 1))
    # Encoder starts
    x = Conv2D(32, kernel_size=3, strides=1, padding="same")(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    # Adjust or remove upsampling here to ensure output is 32x32x3
    # For example, you might skip the upsampling or use a different strategy
    # Output layer
    outputs = Conv2D(3, kernel_size=3, padding="same", activation="tanh")(x)  # Ensure this outputs 32x32x3
    
    
    model = Model(inputs, outputs)
    print(model.summary())
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