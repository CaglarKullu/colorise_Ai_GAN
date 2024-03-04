import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from model import build_generator, build_discriminator
from data_preparation import load_and_preprocess_data


def train_gan(epochs=1000, batch_size=128, save_interval=100):
    K.clear_session()  # Clearing the session to start fresh
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Combined model (stacked generator and discriminator)
    discriminator.trainable = False  # For the combined model, we will only train the generator
    gan_input = Input(shape=(32, 32, 1))
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    
    # Load data
    x_train, _, x_train_gray, _ = load_and_preprocess_data()

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
        gray_imgs = x_train_gray[idx]
        fake_imgs = generator.predict(gray_imgs)
        
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        discriminator.trainable = False
        
        #  Train Generator
        g_loss = gan.train_on_batch(gray_imgs, valid)
        
        # Save progress
        if epoch % save_interval == 0:
            # Ensure the checkpoint directory exists
            save_dir = 'checkpoints'
            os.makedirs(save_dir, exist_ok=True)  # This line avoids the FileNotFoundError
            
            save_path = f'{save_dir}/gan_epoch_{epoch}'
            generator.save_weights(f'{save_path}_generator.h5')
            discriminator.save_weights(f'{save_path}_discriminator.h5')
            print(f"Saved model at epoch {epoch}")
        
        print(f"{epoch} [D loss: {0.5 * np.add(d_loss_real, d_loss_fake)[0]} | D acc: {0.5 * np.add(d_loss_real, d_loss_fake)[1]}] [G loss: {g_loss}]")
