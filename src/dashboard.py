import os
import sys
import time
import base64
import requests
from io import BytesIO

import cv2
import boto3
import numpy as np
from PIL import Image
import streamlit as st
from loguru import logger
from PIL.PngImagePlugin import PngImageFile
from streamlit.delta_generator import DeltaGenerator


title = "Image segmentation"
st.markdown(
    f"<h1 style='text-align: center; color: red;'> " f"{title} </h1>", unsafe_allow_html=True
)

logger.configure(
    handlers=[
        {"sink": sys.stderr, "level": "DEBUG"},
        dict(
            sink="logs/debug.log",
            format="{time} {level} {message}",
            level="DEBUG",
            rotation="1 weeks",
        ),
    ]
)


def bytes_to_image(file: bytes, log=None) -> PngImageFile:
    bytes_data = file.getvalue()
    np_array = np.fromstring(bytes_data, np.uint8)

    image_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    if log:
        log.info(f"image.shape = {image_array.shape}")

    return Image.fromarray(image_array)


def vis_image(image: PngImageFile, col: DeltaGenerator, text: str) -> None:
    with col:
        st.markdown(f'<p style="text-align: center;">{get_image_download_link(image, text)}</p>', unsafe_allow_html=True)
        st.image(image, width=350)


def get_predict_by_image_file(ip: str, port: str) -> PngImageFile:
    response = requests.post(f"http://{ip}:{port}/api/v1/predict/file", files={"file": uploaded_file})
    return Image.open(BytesIO(response.content))


def get_predict_by_file_name(ip: str, port: str, file_name: str) -> str:
    params = {
        'file_name': file_name,
    }

    response = requests.get(f'http://{ip}:{port}/api/v1/predict/file_name', params=params)
    return response.json()


def get_image_download_link(img: PngImageFile, filename: str) -> str:
    with BytesIO() as buf:
        img.save(buf, format="JPEG")
        img_str = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}.jpg">{filename}</a>'
    return href


if __name__ == "__main__":
    INPUT_FOLDER = "Inputs"
    OUTPUT_FOLDER = "Outputs"

    BUCKET = os.getenv("BUCKET")

    ip = os.getenv("IP")
    port = os.getenv("PORT")

    client = boto3.client('s3')

    uploaded_file = st.file_uploader("", type="jpg")
    before_col, after_col = st.columns(2)

    if uploaded_file is not None:
        logger.info("Load image")

        input_image = bytes_to_image(uploaded_file, logger)
        vis_image(input_image, before_col, "Before")

        out_img = BytesIO()
        input_image.save(out_img, format='PNG')
        out_img.seek(0)  # Without this line it fails

        file_name = hash(time.time())
        client.put_object(Body=out_img, Bucket=BUCKET,
                          Key=f'{INPUT_FOLDER}/{file_name}.jpg')

        file_name_predict = get_predict_by_file_name(ip, port, file_name).get("file_name")

        if file_name_predict:
            try:
                buffer = BytesIO()
                client.download_fileobj(BUCKET, f'{OUTPUT_FOLDER}/{file_name_predict}.jpg', buffer)

                output_image = bytes_to_image(buffer, logger)
                vis_image(output_image, after_col, "After")
            except Exception as e:
                st.error(e)
        else:
            st.warning("Invalid prediction host")
