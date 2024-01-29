from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import model

app = Flask(__name__)
cnn_model = model.create_model()
cnn_model.load_weights('optimized_emnist_model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    img = Image.open(file.stream).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)


    prediction = cnn_model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Map the predicted class to the corresponding alphabet (you need to customize this based on your classes)
    alphabet = chr(ord('A') + predicted_class)

    return render_template('index.html', prediction=alphabet)

