import streamlit as st
import cv2
import numpy as np
from sam_utils import *
from inpainting_utils import lama_inpaint, sd_inpaint

st.title("🧠 AI Object Removal & Editing")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded:
    file_path = "input/uploaded.png"
    with open(file_path, "wb") as f:
        f.write(uploaded.read())

    st.image(file_path)

    x = st.number_input("Click X coordinate", value=256)
    y = st.number_input("Click Y coordinate", value=256)

    if st.button("Generate Mask"):
        predictor = initialize_sam_model()
        img = load_image(file_path)
        mask = generate_mask(img, predictor, [(x, y)])
        mask = expand_and_feather_mask(mask)
        save_mask(mask, "input/mask.png")
        st.image("input/mask.png", caption="Generated Mask")

    if st.button("Remove Object (LaMa)"):
        lama_inpaint(file_path, "input/mask.png", "output/lama.png")
        st.image("output/lama.png")

    prompt = st.text_input("Diffusion Prompt")

    if st.button("Replace Object (Stable Diffusion)"):
        sd_inpaint(file_path, "input/mask.png", prompt, "output/sd.png")
        st.image("output/sd.png")
