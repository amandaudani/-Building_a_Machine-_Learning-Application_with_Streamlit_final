import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import time

st.sidebar.title("Options")
task = st.sidebar.selectbox("Choose a machine learning task", ["Image Classification", "Sentiment Analysis"])

with st.container():
    st.title("Machine Learning Application Flower Analysis")
    st.write("This application allows you to upload and display images, videos, or audio files, and perform machine learning tasks.")

st.write("### Media Display")
media_file = st.file_uploader("Upload an image, video, or audio file", type=["jpg", "jpeg", "png", "mp4", "mp3", "wav"])

if media_file is not None:
    file_type = media_file.type.split('/')[0]
    if file_type == 'image':
        image = Image.open(media_file)
        st.image(image, caption='Uploaded Media.', use_column_width=True)
        if task == "Image Classification":
            st.write("Classifying...")

            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            # Load pre-trained model
            model = tf.keras.applications.MobileNetV2(weights='imagenet')

            # Preprocess the image
            img_array = np.array(image.resize((224, 224)))
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # Make predictions
            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

            # Display predictions
            st.write("Predictions:")
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
                st.write(f"{i + 1}: {label} ({score * 100:.2f}%)")

    elif file_type == 'video':
        st.video(media_file)
    elif file_type == 'audio':
        st.audio(media_file)

if task == "Sentiment Analysis":
    st.write("Sentiment Analysis feature coming soon!")

# Input widgets
with st.container():
    st.write("")
    slider_val = st.slider("Select a value", 0, 100)
    st.write(f"Slider value: {slider_val}")

    text_input_val = st.text_input("Enter details here")
    st.write(f"Text input: {text_input_val}")

    button_clicked = st.button("Click me")
    if button_clicked:
        st.write("Button clicked!")

# Graph
with st.container():
    st.write("### Graph")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

# Link to GitHub repository
st.write("### GitHub Repository")
st.write("[Link to the GitHub repository](https://github.com/your-repo-link)")

# Deployment status
st.write("### Deployment Status")
st.write("Deployed on Streamlit Cloud: [Streamlit App](https://share.streamlit.io/your-app-link)")
