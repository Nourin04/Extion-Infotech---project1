# Image Classification: Dogs vs Cats

## Objective
The goal of this project is to develop a Convolutional Neural Network (CNN) for classifying images of dogs and cats using a deep learning framework, specifically TensorFlow. The project follows these steps:
1. Choose a dataset (Kaggle’s Dogs vs Cats).
2. Build and train a CNN model.
3. Evaluate the model's accuracy on a test dataset.
4. Fine-tune the model for improved performance.

## Highlights of the Project
- **Dataset**: 
  - Installed Kaggle's API and imported the "Dogs vs Cats" dataset.
  - Extracted the dataset and preprocessed the images.

- **Preprocessing**:
  - Resized all images to a uniform size for input into the CNN.
  - Created labels for each image: `1` for dogs and `0` for cats.
  - Converted the resized images into NumPy arrays for efficient processing.

- **Model Development**:
  - Performed Train-Test Split to prepare the data for model training and evaluation.
  - Built a **Convolutional Neural Network (CNN)** using TensorFlow to classify images as either a cat or a dog.

- **Model Training**:
  - Trained the CNN model on the preprocessed images and evaluated its performance.

- **Prediction System**:
  - Developed a predictive system capable of identifying whether a given image is a dog or a cat with high accuracy.

## Key Libraries and Tools Used
- **Kaggle API**: For importing the dataset.
- **NumPy**: For efficient array manipulation.
- **PIL (Python Imaging Library)**: For image resizing and processing.
- **Matplotlib**: For visualizing the data.
- **Scikit-learn**: For splitting the dataset into training and testing sets.
- **OpenCV**: For displaying images using `cv2_imshow`.
- **TensorFlow**: For building and training the CNN.

