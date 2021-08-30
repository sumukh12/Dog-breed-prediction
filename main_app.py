import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
#loading the model
model=load_model('dog_breed.h5')
#name of classes
CLASS_NAMES=['scottish_deerhound','maltese_dog','bernese_mountain_dog']
#setting title of app
st.title('Dog breed Prediction')
st.markdown('upload an image of the Dog')
#Uploading the Dog image
dog_image=st.file_uploader('choose an image...',type=png)
submit=st.button('Predict')
#On predict button click
if submit:

      if dog_image is not None:
        #Convert the file in opencv dog_image
        file_bytes=np.asarray(bytearray(dog_image.read()),dtype=np.unit8)
        opencv_image=cv2.imdecode(file_bytes,1)


      #Displaying the image
      st.image(opencv_image,channels='BGR')
      #Resizeing the image
      opencv_image=cv2.resize(opencv_image ,(224,224))
      #convert image in 4 dimension
      opencv_image.shape= (1,224,244,3)
      #make Prediction
      Y_pred=model.predict(opencv_image)
      st.title(str('The Dog Breed is'+CLASS_NAMES[np.argmax(Y_pred)]))
