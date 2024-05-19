import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(image):
    # Normalize and resize the image if needed
    image = np.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, 28, 28)  # Reshape to (1, 28, 28) to match the model's input shape

def predict(model, image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    return np.argmax(predictions)  # Return the index of the highest probability

def main():
    # Load the trained model
    # model = load_model('mnist_model.h5')
    model = load_model('mnist-new-model.keras')

    # Load or input a new image (here you would replace this with actual image data)
    # For demonstration, let's use a random image from the MNIST dataset itself.
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    random_idx = np.random.randint(0, len(x_train))
    image = x_train[random_idx]
    print("^^^^^^^^^^^^^^^")
    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title(f"Original Label: {y_train[random_idx]}")
    plt.show()
    print("************")
    # Predict the digit
    prediction = predict(model, image)
    print(f"Predicted Label: {prediction}")

if __name__ == "__main__":
    main()
    print("------------------")