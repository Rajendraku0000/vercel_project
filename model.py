import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from emnist import extract_training_samples, extract_test_samples
from sklearn.model_selection import train_test_split

# # Load EMNIST data
# train_images, train_labels = extract_training_samples('letters')
# test_images, test_labels = extract_test_samples('letters')

# # Preprocess data
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# # Add channel dimension for CNN input
# train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
# test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# # Convert labels to one-hot encoding
# train_labels = tf.keras.utils.to_categorical(train_labels, 27)
# test_labels = tf.keras.utils.to_categorical(test_labels, 27)


# # Split training data into training and validation sets
# train_images, val_images, train_labels, val_labels = train_test_split(
#     train_images, train_labels, test_size=0.1, random_state=42
# )

# Build CNN model
# Build a more complex CNN model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(27, activation='softmax'))  # Change to 27 classes
    return model



# model=create_model()
# # Compile the model with a lower learning rate
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Implement learning rate reduction and early stopping callbacks
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model with the callbacks
# history = model.fit(
#     train_images, train_labels,
#     epochs=30,
#     validation_data=(val_images, val_labels),
#     callbacks=[reduce_lr, early_stopping]
# )

# # Evaluate the model on test data
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(f'Test Accuracy: {test_acc}')

# # Save the optimized model
# model.save('optimized_emnist_model.h5')


import cv2
import numpy as np
from tensorflow.keras.models import load_model


def preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    # Resize the image to 28x28 (assuming EMNIST-like size)
    image = cv2.resize(image, (28, 28))

    # Normalize pixel values to the range [0, 1]
    image_normalized = image / 255.0

    # Add channel dimension for grayscale
    image_normalized = image_normalized.reshape((1, 28, 28, 1))

    return image_normalized

def load_and_predict_alphabet(image_path, model_path="optimized_emnist_model.h5", mapping_file_path="C:/Users/rajen/OneDrive/Desktop/assin/emnist-letters-mapping.txt"):
    # Load the trained model
    loaded_model = tf.keras.models.load_model(model_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    if preprocessed_image is None:
        return None

    # Use the trained model to make predictions
    predictions = loaded_model.predict(preprocessed_image)

    # Get the predicted alphabet (assuming labels are integers 0-26)
    predicted_alphabet_index = np.argmax(predictions)

    # Convert the index to the corresponding letter using a mapping file
    with open(mapping_file_path, 'r') as file:
        mappings = file.read().splitlines()

    letter_mapping = {int(mapping.split()[0]): mapping.split()[1] for mapping in mappings}
    predicted_alphabet = letter_mapping[predicted_alphabet_index]

    return predicted_alphabet


