import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cnn import CNN, load_model_weights

# ========================================================================
# Functions
# ========================================================================

def model_predict():

    classes = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'Inside city', 'Kitchen', 'Living room', 'Mountain', 'Office', 'Open country', 'Store', 'Street', 'Suburb', 'Tall building']
    model_weights = load_model_weights('resnet50-1epoch')
    my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), len(classes))
    my_trained_model.load_state_dict(model_weights)
    predicted_label = predict(my_trained_model, img)
    return classes[predicted_label]

def predict(my_trained_model, img):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])

    tensor = transform(img)
    my_trained_model.eval()
    with torch.no_grad():
        output = my_trained_model(tensor.unsqueeze(0))
        _, predicted = torch.max(output, 1)
    return predicted

def transform_image(img):
    # Convert the image to grayscale
    img_gray = transforms.Grayscale(num_output_channels=1)(img)
    # Convert the grayscale image to a tensor
    img_tensor = transforms.ToTensor()(img_gray)
    # Normalize the tensor for plotting
    img_tensor = (img_tensor - torch.min(img_tensor)) / (torch.max(img_tensor) - torch.min(img_tensor))
    return {"img": img, "im_gray": img_gray, "img_tensor": img_tensor}


# ========================================================================
# Application Config
# ========================================================================
st.set_page_config(layout="wide", page_title="Image prediction with CNNs")
col01, col02, col03 = st.columns(3)
with col01:
    st.write('# Image prediction with CNNs')
with col03:
    st.image("C:/Users/LENOVO/Desktop/MásterBD/ML2/Practicas_DeepLearning_2024/streamlit/img/icai.png")

st.write("### By: Fernando Carballeda, Lucas Justo, Gonzalo Barderas and Manuel Oliveira")


col21, col22, col23 = st.columns(3)
with col21:
    st.write("#### Drag and Drop Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        fig, ax = plt.subplots(figsize=(12, 6))
        img = transform_image(img)
        plt.imshow(img["img_tensor"].squeeze(), cmap='gray')
        plt.colorbar()
        st.pyplot(fig)
    

col41, col42, col43 = st.columns(3)
with col41:
    if st.button("Predict"):
        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            predicted_class = model_predict()  # Llama a la función de predicción
            st.write("##### Predicted Class:", predicted_class)


