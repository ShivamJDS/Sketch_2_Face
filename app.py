import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from IPython.display import Image
from tqdm import tqdm

import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

from PIL import Image

import os
import shutil
import time
import cv2

from matplotlib import pyplot as plt

import streamlit as st

import Generator_Rough_To_Good
import Generator_Good_To_Celeb





@st.cache(allow_output_mutation=True)
def load_gan_model():

    generator_r2g = Generator_Rough_To_Good.Generate_Model_Structure()
    generator_g2c = Generator_Good_To_Celeb.Generate_Model_Structure()

    generator_r2g = tf.keras.models.load_model("RoughSketchToGoodSketchGenerator" , compile=False)
    generator_g2c = tf.keras.models.load_model("GoodSketchToCelebImageGenerator" , compile=False)
    
    
    return generator_r2g, generator_g2c


def main():

    st.title("Sketch To Face")

    """
	## Today you will just draw a rough sketch and, it will be converted into real face

	You should be excited to try it, so just scroll down , draw sketch and have fun!!

	"""

    st.video("Tuitorial.mkv", format='video/mkv', start_time=0)

    loading = st.text("gan is loading....")
	
    generator_r2g,generator_g2c = load_gan_model()

    loading.text("gan is loaded")

    st.write("")
    st.write("")



    st.title("Let's have some fun!!")
    st.write("")

    """
	## Draw rough sketch on this canvas and get its good sketch and real image
	"""

    st.write("")

	# Add a selectbox to the sidebar:
    add_selectbox = st.sidebar.selectbox(
		'What would you like to Draw?',
		('Sketch Of Girl', 'Sketch Of Boy'))

    if add_selectbox == "Sketch Of Girl":

        st.sidebar.header("Configuration")

        tools = st.sidebar.selectbox("Pencil or Eraser?",("Pencil","Eraser"))

		# Specify canvas parameters in application
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        
        if tools == "Pencil":
            stroke_color = "#000000"
        else:
            stroke_color = "#FFFFFF"
        
        bg_color = "#FFFFFF"
        bg_image = "girl.jpg"
        drawing_mode = st.sidebar.selectbox(
		    "Drawing tool:", ("freedraw", "transform")
		)
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        

		# Create a canvas component
        canvas_result = st_canvas(
		    fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with some opacity
		    stroke_width=stroke_width,
		    stroke_color=stroke_color,
		    background_color="" if bg_image else bg_color,
		    background_image=Image.open(bg_image) if bg_image else None,
		    update_streamlit=realtime_update,
		    height=512,
		    width =512,
		    drawing_mode=drawing_mode,
		    key="canvas")

        if canvas_result.image_data is not None:
            
            #st.image(canvas_result.image_data , caption="Your Drawing")
            gray_inverted = canvas_result.image_data[:,:,3]
            cv2.imwrite("drawing.jpg",gray_inverted)
            gray_inverted = cv2.imread("drawing.jpg")
            gray = cv2.bitwise_not(gray_inverted)
            cv2.imwrite("drawing.jpg",gray)

            Rough_Sketch = tf.io.read_file("drawing.jpg")
            Rough_Sketch = tf.io.decode_jpeg(Rough_Sketch,channels=3)

            Rough_Sketch = tf.cast(Rough_Sketch, tf.float32)

            Rough_Sketch = (Rough_Sketch/127.5) - 1
		    
            Rough_Sketch = Rough_Sketch[tf.newaxis,...]

            Good_Sketch  = generator_r2g(Rough_Sketch,training=True)[0]
		    
            Good_Sketch_Image = np.array((Good_Sketch + 1)*127.5)

            cv2.imwrite("Good_Sketch.jpg",Good_Sketch_Image)
		    
            Real_Face = generator_g2c(Good_Sketch[tf.newaxis,...],training=True)
            Real_Face = Real_Face[0]
            Real_Face_Image = np.array((Real_Face + 1)*127.5)

            Real_Face_Image = cv2.cvtColor(Real_Face_Image, cv2.COLOR_RGB2BGR)

            cv2.imwrite("Real_Face.jpg",Real_Face_Image)

            st.image("Good_Sketch.jpg" , caption="Generated Good Sketch")
            st.image("Real_Face.jpg" , caption="Generated Real Face")
            
            
    elif add_selectbox == "Sketch Of Boy":

        st.sidebar.header("Configuration")

        tools = st.sidebar.selectbox("Pencil or Eraser?",("Pencil","Eraser"))

		# Specify canvas parameters in application
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        
        if tools == "Pencil":
            stroke_color = "#000000"
        else:
            stroke_color = "#FFFFFF"
        
        bg_color = "#FFFFFF"
        bg_image = "boy.jpg"
        drawing_mode = st.sidebar.selectbox(
		    "Drawing tool:", ("freedraw", "transform")
		)
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        

		# Create a canvas component
        canvas_result = st_canvas(
		    fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with some opacity
		    stroke_width=stroke_width,
		    stroke_color=stroke_color,
		    background_color="" if bg_image else bg_color,
		    background_image=Image.open(bg_image) if bg_image else None,
		    update_streamlit=realtime_update,
		    height=512,
		    width =512,
		    drawing_mode=drawing_mode,
		    key="canvas")

        if canvas_result.image_data is not None:
            
            #st.image(canvas_result.image_data , caption="Your Drawing")
            gray_inverted = canvas_result.image_data[:,:,3]
            cv2.imwrite("drawing.jpg",gray_inverted)
            gray_inverted = cv2.imread("drawing.jpg")
            gray = cv2.bitwise_not(gray_inverted)
            cv2.imwrite("drawing.jpg",gray)

            Rough_Sketch = tf.io.read_file("drawing.jpg")
            Rough_Sketch = tf.io.decode_jpeg(Rough_Sketch,channels=3)

            Rough_Sketch = tf.cast(Rough_Sketch, tf.float32)

            Rough_Sketch = (Rough_Sketch/127.5) - 1
		    
            Rough_Sketch = Rough_Sketch[tf.newaxis,...]

            Good_Sketch  = generator_r2g(Rough_Sketch,training=True)[0]
		    
            Good_Sketch_Image = np.array((Good_Sketch + 1)*127.5)

            cv2.imwrite("Good_Sketch.jpg",Good_Sketch_Image)
		    
            Real_Face = generator_g2c(Good_Sketch[tf.newaxis,...],training=True)
            Real_Face = Real_Face[0]
            Real_Face_Image = np.array((Real_Face + 1)*127.5)

            Real_Face_Image = cv2.cvtColor(Real_Face_Image, cv2.COLOR_RGB2BGR)

            cv2.imwrite("Real_Face.jpg",Real_Face_Image)

            st.image("Good_Sketch.jpg" , caption="Generated Good Sketch")
            st.image("Real_Face.jpg" , caption="Generated Real Face")


    st.write("")
    st.write("")
    st.write("")
    st.write("")



    st.image("Image1.png",caption="Our generated image from rough sketch is way more similar to real image")

    st.write("")
    st.write("")

    """

	From rough sketch you will be making good sketch and real face image

	If you are good at sketching , you could make real face image directly from your good sketch
	"""

    st.write("")
    st.write("")

    st.image("Image2.png",caption="Our generated image from good sketch is way more similar to real image")

    st.write("")
    st.write("")
	



if __name__ == "__main__":
    main()
