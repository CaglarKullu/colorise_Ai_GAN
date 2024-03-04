import matplotlib.pyplot as plt

def display_images(grayscale, original, predicted, n=10):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Display grayscale image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(grayscale[i].reshape(32, 32), cmap='gray')
        plt.title("Grayscale")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Display original image
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(original[i])
        plt.title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display GAN colorized image
        ax = plt.subplot(3, n, i + 1 + n * 2)
        plt.imshow(predicted[i])
        plt.title("Colorized")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
