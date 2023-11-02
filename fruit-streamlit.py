import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

background_image_url = "https://img.freepik.com/free-vector/summer-background-videocalls_23-2148969179.jpg?size=626&ext=jpg"

# Function to set the background image
def set_background_image(image_url):
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Use the set_background_image function to set the background
set_background_image(background_image_url)

hide_menu_tools = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibilty: hidden; }
    header {background-color: rgba(0,0,0,0);}
    </style>
    """
st.markdown(hide_menu_tools,unsafe_allow_html=True)

st.markdown('''
            <h1 style="text-align:center;color: BLACK;">FRUIT DETECT</h1>
            ''',unsafe_allow_html=True)
st.markdown('''
            <p style="text-align:center;font-family:'Arial';color: black;">Detect Fruit in your Image using Deep Learning Techniques
            ''',unsafe_allow_html=True)

# Create a button with a link
button_html = """
<a href="#" class="btn">Detect</a>
"""

# Markdown CSS for the button
button_markdown = """
<style>
.btn {
    font-size: 17px;
    text-transform: uppercase;
    text-decoration: none;
    padding: 1em 2.5em;
    display: inline-block;
    border-radius: 6em;
    transition: all .2s;
    border: none;
    font-family: inherit;
    font-weight: 500;
    color: white;
    background-color: purple;
    display: block;
    margin: 0 auto;
    text-align: center;
}

.btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
}

.btn:active {
    transform: translateY(-1px);
    box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}

.btn::after {
    content: "";
    display: inline-block;
    height: 100%;
    width: 100%;
    border-radius: 100px;
    position: absolute;
    top: 0;
    left: 0;
    z-index: -1;
    transition: all .4s;
    background-color: #fff;
}

.btn:hover::after {
    transform: scaleX(1.4) scaleY(1.6);
    opacity: 0;
}
</style>
"""

# Create a Streamlit button with the Markdown CS
col1, col2, col3 = st.columns([4, 1.5, 4])

# Place the button in the centered column (col2)
with col2:
    detect=st.button("Detect", key="custom_button")
if detect:
        st.markdown(button_markdown, unsafe_allow_html=True)
        model = tf.keras.models.load_model('new_model')
        cap = cv2.VideoCapture(0)
        count = 0
        while True:
            ret, frame = cap.read()
            count += 1
            image = cv2.resize(frame, (400, 400))
            image = np.array(image) / 255.0
            predictions = model.predict(np.expand_dims(image, axis=0))
            class_names = ['Apple', 'Banana', 'Orange', 'Pitaya', 'Pomegranate', 'Tomato']
            predicted_class = np.argmax(predictions)
            print(f"Predicted Class: {class_names[predicted_class]}")
            print(f"Confidence: {predictions[0][predicted_class]:.2%}")
            if count == 1:
                break
        cap.release()
        cv2.destroyAllWindows()

        nutritional_data = {
            'Fruit': ['Apple', 'Banana', 'Orange', 'Pitaya', 'Pomegranate', 'Tomato'],
            'Calories (kcal)': [52, 105, 62, 60, 83, 18],
            'Carbohydrates (g)': [14, 27, 15, 9, 19, 4],
            'Protein (g)': [0.3, 1.3, 1.2, 2.2, 1.7, 0.9],
            'Fat (g)': [0.2, 0.3, 0.2, 0.4, 1.2, 0.2],
            'Fiber (g)': [2.4, 3.1, 3.1, 1.9, 4, 1.5]
        }

        nutritional_df = pd.DataFrame(nutritional_data)
        nutritional_df.set_index('Fruit')
        fruit_index = predicted_class  # Index of the fruit you want to display

        st.markdown(
            f'<h2 style="color:black; text-align:center;">Nutritional Information for {nutritional_df.loc[fruit_index, "Fruit"]}</h2>',
            unsafe_allow_html=True
        )

        for key, value in nutritional_df.iloc[fruit_index, 1:].items():
            st.markdown(
                f'<p style="color:black; text-align:center;">{key}: {value}</p>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<p style="color:black; text-align:center;">**Note**: Values are approximate and may vary depending on variety and preparation.</p>',
            unsafe_allow_html=True
        )






 
    


