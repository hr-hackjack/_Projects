#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf


# In[3]:


from keras.models import load_model, Model

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# In[4]:


# Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index
word_to_index = {}
with open ("data/textFiles/word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file)

index_to_word = {}
with open ("data/textFiles/idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file)


# In[5]:


resnet50_model = ResNet50 (weights = 'imagenet', input_shape = (224, 224, 3))
resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)


# In[6]:


def preprocess_image (img):
    img = tf.keras.utils.load_img(img, target_size=(224, 224))
    img = tf.keras.utils.img_to_array(img)

    # Convert 3D tensor to a 4D tendor
    img = np.expand_dims(img, axis=0)

    #Normalize image accoring to ResNet50 requirement
    img = preprocess_input(img)

    return img


# In[7]:


# Encoding (feature vector)
def encode_image (img):
    img = preprocess_image(img)

    feature_vector = resnet50_model.predict(img)
    return feature_vector


# In[8]:


# Generate Captions for an image
def predict_caption(photo):

    inp_text = "startseq"

    for i in range(38):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=38, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]

        inp_text += (' ' + word)

        if word == 'endseq':
            break

    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[ ]:


print("Loading the Neural_Image_Caption_Model...\n")
model = load_model('Models/model_19.h5')



print("Encoding the image ...")
img_name = "image.jpg"
photo = encode_image(img_name).reshape((1, 2048))



print("\n\nRunning model to generate the caption...")
caption = predict_caption(photo)

img_data = plt.imread(img_name)
plt.imshow(img_data)
plt.axis("off")

plt.show()
print(caption)


# In[ ]:





# In[ ]:




