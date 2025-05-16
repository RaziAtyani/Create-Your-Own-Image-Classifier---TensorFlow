import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image

# Constants for processing
BATCH_SIZE = 32
INPUT_IMAGE_SIZE = 224
LABEL_MAP = {}


def preprocess_image(image):
    """
    Preprocesses the input image for model inference:
    - Casts image to float32
    - Resizes to required dimensions
    - Normalizes pixel values to [0, 1]
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
    image /= 255.0  # Normalize pixel values
    return image.numpy()


def load_and_predict(image_path, trained_model, top_k=5):
    """
    Loads an image, preprocesses it, runs prediction, and retrieves top-K classes.
    :param image_path: Path to input image file
    :param trained_model: Loaded TensorFlow model
    :param top_k: Number of top predictions to return
    :return: Tuple of probabilities and class names
    """
    # Open and preprocess the image
    img = Image.open(image_path)
    img_array = np.expand_dims(np.array(img), axis=0)
    processed_img = preprocess_image(img_array)

    # Run model prediction
    predictions = trained_model.predict(processed_img)

    # Sort predictions and fetch top-K results
    sorted_indices = np.argsort(predictions[0])[::-1]
    top_probs = [predictions[0][i] for i in sorted_indices[:top_k]]
    top_classes = [LABEL_MAP[str(i + 1)] for i in sorted_indices[:top_k]]

    return top_probs, top_classes


def main():
    """Main function to parse arguments, load model, and predict classes."""
    print("Starting prediction script...\n")

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Image Classifier Script")
    parser.add_argument('image_path', type=str, help="Path to input image file")
    parser.add_argument('model_path', type=str, help="Path to saved TensorFlow model")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to return")
    parser.add_argument('--category_names', type=str, help="Path to JSON file for class label mapping")
    args = parser.parse_args()

    # Print arguments
    print(f"Image path: {args.image_path}")
    print(f"Model path: {args.model_path}")
    print(f"Top-K predictions: {args.top_k}")
    print(f"Category names file: {args.category_names}\n")

    # Load label mapping
    global LABEL_MAP
    if args.category_names:
        with open(args.category_names, 'r') as json_file:
            LABEL_MAP = json.load(json_file)

    # Load the pre-trained model
    print("Loading the model...")
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    print("Model loaded successfully.\n")

    # Perform prediction
    probabilities, predicted_classes = load_and_predict(args.image_path, model, args.top_k)

    # Display results
    print("Top predictions:")
    for prob, cls in zip(probabilities, predicted_classes):
        print(f"Class: {cls}, Probability: {prob:.4f}")


if __name__ == '__main__':
    main()
