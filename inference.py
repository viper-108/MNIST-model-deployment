import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(image):
    
    image = np.resize(image, (28, 28))
    image = image / 255.0
    return image.reshape(1, 28, 28)  

def predict(model, image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    return np.argmax(predictions) 
    
def main():

    model = load_model('mnist_model.h5')
    # model = load_model('mnist-new-model.keras')

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    random_idx = np.random.randint(0, len(x_train))
    image = x_train[random_idx]

    plt.imshow(image, cmap='gray')
    plt.title(f"Original Label: {y_train[random_idx]}")
    plt.show()

    prediction = predict(model, image)
    print(f"Predicted Label: {prediction}")

if __name__ == "__main__":
    main()
