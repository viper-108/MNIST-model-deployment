import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    return (x_train, y_train), (x_test, y_test)

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

def main():
    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        (x_train, y_train), (x_test, y_test) = load_data()

        model = build_model()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
        print(f"Test accuracy: {test_accuracy:.4f}")

        model.save('mnist-new-model.keras')
        
if __name__ == "__main__":
    main()
