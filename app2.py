# Libraries
import pandas as pd
import numpy as np
import os
import pickle
from tqdm.notebook import tqdm
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import streamlit as st
import unicodedata
from tensorflow.keras.models import load_model
# Libraries
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.layers import Input,Dense, LSTM, Embedding,Dropout,add


# Load the model and tokenizer
BASE_DIR = "D:\\Data_science\\Mini Project\\image caption generater"
WORKING_DIR = 'C:\\Users\\mades\\Documents\\Image Caption Generater'

# Load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()


# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text
#model
model = load_model(os.path.join(WORKING_DIR, "best200_model.keras"))
#tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


def main():
    st.title("Image Caption Generator")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Generate and display caption on button click
        if st.button('Generate Caption'):
            # Resize the image to match the input shape of the VGG16 model
            image = image.resize((224, 224))

            # Convert the image to array and preprocess it
            image_array = img_to_array(image)
            image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
            image_array = preprocess_input(image_array)

            # Extract features from the image using the VGG16 model
            vgg_model = VGG16()
            vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
            feature = vgg_model.predict(image_array, verbose=0)

            # Predict caption using the captioning model
            caption = predict_caption(model, feature, tokenizer, max_length=35)

            # Print the caption without startseq and endseq
            # Print the caption
            st.markdown(f"<div style='border: 2px solid red; border-radius: 5px; padding: 10px;'><b>Generated Caption:</b> {' '.join(caption.split()[1:-1])}</div>", unsafe_allow_html=True)




# Run the app
if __name__ == "__main__":
    main()
