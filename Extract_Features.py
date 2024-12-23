import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import os

# Load the ResNet50 model with weights pre-trained on ImageNet
image_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
os.chdir("C:/Users/seema/Desktop/Pratyush/Food Vision 2/")

def extract_features(img_path):
    """Extract features from an image using a pre-trained model."""
    # Load image and resize it to the size expected by ResNet50
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    
    # Expand the dimensions to fit the model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image for the model
    img_array = preprocess_input(img_array)
    
    # Extract features using the pre-trained model
    features = image_model.predict(img_array)
    
    return features

def extract_features_from_directory(dataset_dir, output_file='dataset/extracted_features.npy', output_paths_file='dataset/image_paths.npy'):
    """Extract features from all images in the dataset directory."""
    all_features = []
    image_paths = []

    # Walk through the dataset directory and get all image files
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            # Check if the file is an image (you can add other formats like .jpeg if needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                print(f"Processing image: {img_path}")
                # Extract features for the image
                features = extract_features(img_path)
                all_features.append(features)
                image_paths.append(img_path)

    # Convert list of features to numpy array
    all_features = np.array(all_features)

    # Save the features and image paths to numpy files
    np.save(output_file, all_features)
    np.save(output_paths_file, image_paths)
    print(f"Features saved to '{output_file}'")
    print(f"Image paths saved to '{output_paths_file}'")

    return all_features, image_paths

# Main section to execute the feature extraction for all images in the dataset directory
if __name__ == "__main__":
    # Path to the dataset directory containing images
    dataset_dir = input("Enter the path to your dataset directory: ")

    # Check if the directory exists
    if not os.path.exists(dataset_dir):
        print("The directory does not exist!")
    else:
        # Extract features from all images in the dataset directory
        features, image_paths = extract_features_from_directory(dataset_dir)

        # Print the extracted features shape
        print(f"Extracted features shape: {features.shape}")
