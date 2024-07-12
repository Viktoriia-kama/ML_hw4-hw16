import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load models
cnn_model = load_model('cnn_model.h5')
vgg16_model = load_model('vgg16_model.h5')

# Define class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Helper function to preprocess images
def preprocess_image(image, model_type='cnn'):
    if model_type == 'cnn':
        image = image.convert('L')
        image = image.resize((28, 28))
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
    else:  # model_type == 'vgg16'
        image = image.convert('RGB')
        image = image.resize((32, 32))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
    return image

# Streamlit app
st.title('Neural Network Image Classifier')

# Model selection
model_choice = st.sidebar.selectbox('Choose model', ('CNN', 'VGG16'))

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Preprocess and classify
    model_type = 'cnn' if model_choice == 'CNN' else 'vgg16'
    processed_image = preprocess_image(image, model_type=model_type)
    model = cnn_model if model_choice == 'CNN' else vgg16_model
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Display results
    st.write(f'Predicted Class: {class_names[predicted_class]}')
    st.write('Probabilities:')
    for i, class_name in enumerate(class_names):
        st.write(f'{class_name}: {predictions[0][i]:.4f}')

# Show training history
st.sidebar.subheader('Training History')
if st.sidebar.button('Show Loss and Accuracy'):
    history_file = 'cnn_history.npy' if model_choice == 'CNN' else 'vgg16_history.npy'
    history = np.load(history_file, allow_pickle=True).item()  # Load the training history

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax[0].plot(history['loss'], label='Training Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(history['accuracy'], label='Training Accuracy')
    ax[1].plot(history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].legend()

    st.pyplot(fig)
