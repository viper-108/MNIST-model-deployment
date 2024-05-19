from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('mnist_model.h5')

def preprocess_image(image):
    image = np.array(image).astype(np.float32) / 255.0
    return image.reshape(1, 28, 28)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image = data['image']
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return jsonify({'prediction': int(np.argmax(predictions))})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9201)
