import ast
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import urllib.request
from fastai.vision import open_image, load_learner
from PIL import Image

classes = [
    "banh-mi",
    "banh-xeo",
    "bubble-tea",
    "bun-bo-hue",
    "bun-bo-nam-bo",
    "bun-cha",
    "bun-dau-mam-tom",
    "che",
    "hu-tieu",
    "pho",
    "spring-rolls",
]

with open("info.txt") as f:
    info = ast.literal_eval(f.read())


def open_image_url(url):
    urllib.request.urlretrieve(url, "./img/test.jpg")
    return open_image("./img/test.jpg")


def plot_probs(outputs):
    probs = pd.Series(np.round(outputs.numpy() * 100, 2), classes)
    probs = probs.sort_values(ascending=False).reset_index()
    probs.columns = ["Class", "Probability"]
    fig = px.bar(probs, x="Class", y="Probability")
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.markdown("<h1 style='text-align: center;'>What is this Vietnamese food?🍜</h1>", unsafe_allow_html=True)
    st.markdown("<center><img src='https://www.google.com/logos/doodles/2020/celebrating-banh-mi-6753651837108330.3-2xa.gif' width='500'></center>", unsafe_allow_html=True)
    learn = load_learner("models/")

    # Input URL
    st.write("")
    url = st.text_input(
        "URL: ",
        "https://cuisine-vn.com/wp-content/uploads/2020/03/google-first-honors-vietnamese-bread-promoting-more-than-10-countries-around-the-world-2.jpg",
    )

    if url:
        # Get and show image
        img_input = open_image_url(url)
        st.markdown("<h2 style='text-align: center;'>Image📷</h2>", unsafe_allow_html=True)
        st.markdown(f"<center><img src='{url}' width='500'></center>", unsafe_allow_html=True)

        # Predict
        st.write("")
        st.markdown("<h2 style='text-align: center;'>Output🍲</h2>", unsafe_allow_html=True)
        pred_class, pred_idx, outputs = learn.predict(img_input)
        st.markdown(info[str(pred_class)])
        st.markdown(f"**Probability:** {outputs[pred_idx] * 100:.2f}%")

        # Plot
        plot_probs(outputs)

    # Reference
    st.markdown(
"""## Resources
[![](https://img.shields.io/badge/GitHub-View_Repository-blue?logo=GitHub)](https://github.com/chriskhanhtran/vn-food-app)
- [How the Vietnamese Food Classifier was trained](https://github.com/chriskhanhtran/vn-food-app/blob/master/notebook.ipynb)
- [Fast AI: Lesson 1 - What's your pet](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)
- [Fast AI: Lesson 2 - Creating your own dataset from Google Images](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb)
- [PyImageSearch: How to (quickly) build a deep learning image dataset](https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)
""")



if __name__ == "__main__":
    main()