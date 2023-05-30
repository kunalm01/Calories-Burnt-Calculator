import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model.pkl')
    return model


def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((180, 180))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_class(image, model):
    prediction = model.predict(image)
    return prediction


model = load_model()
st.title('Vehicle Classifier')

file = st.file_uploader("Upload an image of a vehicle", type=["jpg", "jpeg", "png"])

if file is None:
    st.text('Waiting for upload....')
else:
    slot = st.empty()
    slot.text('Running inference....')

    test_image = Image.open(file)
    st.image(test_image, caption="Input Image", width=400)

    preprocessed_image = preprocess_image(test_image)

    pred = predict_class(preprocessed_image, model)

    class_names = ['car', 'motorcycle', 'bus', 'truck']

    result = class_names[np.argmax(pred)]

    output = 'The image is a ' + result

    slot.text('Done')
    st.success(output)
