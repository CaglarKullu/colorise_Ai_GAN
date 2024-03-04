from train import train_gan
from data_preparation import load_and_preprocess_data
from model import build_generator
from evaluate import evaluate_performance
from display import display_images

def main():
    train_gan(epochs=10000, batch_size=128, save_interval=200) 
    
    # Load the latest generator for evaluation
    generator = build_generator()
    generator.load_weights('path_to_latest_generator.h5')
    
    # Load data
    _, x_test, _, x_test_gray = load_and_preprocess_data()
    
    # Generate images
    predicted_images = generator.predict(x_test_gray[:10])
    
    # Evaluate and display
    evaluate_performance(x_test[:10], predicted_images)
    display_images(x_test_gray[:10], x_test[:10], predicted_images)

if __name__ == "__main__":
    main()
