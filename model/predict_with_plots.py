import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

MODEL_PATH = "dog_breed_classifier.keras"
IMAGE_PATH = "golden_test.jpeg"
IMAGE_SIZE = 224

class_names = [
    "Beagle", "Boxer", "Bulldog", "Dachshund", "German_Shepherd",
    "Golden_Retriever", "Labrador_Retriever", "Poodle", "Rottweiler", "Yorkshire_Terrier"
]

def predict_breed():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    
    img = keras.utils.load_img(IMAGE_PATH, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_breeds = [class_names[i] for i in top_indices]
    top_confidences = [predictions[0][i] for i in top_indices]
    
    # The highest confidence prediction
    predicted_breed = top_breeds[0]
    confidence = top_confidences[0]
    
    print(f"\nPredicted breed: {predicted_breed}")
    print(f"Confidence: {confidence:.4f}")
    
    print("\nTop 3 predictions:")
    for breed, conf in zip(top_breeds, top_confidences):
        print(f"{breed}: {conf:.4f}")
    
    plt.figure(figsize=(12, 5))
    
    # Left subplot is image with prediction
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_breed}", fontsize=14)
    plt.axis('off')
    
    # Right subplot is confidence bars
    plt.subplot(1, 2, 2)
    colors = ['#1f77b4'] * len(top_breeds)
    colors[0] = '#ff7f0e'
    
    plt.barh(top_breeds, top_confidences, color=colors)
    plt.xlabel('Confidence', fontsize=12)
    plt.title('Top Predictions', fontsize=14)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    
    plt.savefig('prediction_result.png')
    plt.show()

if __name__ == "__main__":
    predict_breed()
