### AI Object Removal, Replacement & Image Blending System

An end-to-end AI image editing pipeline that combines:

Meta Segment Anything Model (SAM) for object segmentation

Mask refinement (expansion + feathering)

LaMa Inpainting for object removal

Stable Diffusion Inpainting for object replacement

Poisson Blending for realistic image compositing

Streamlit UI for interactive use

This project demonstrates a complete AI-powered image manipulation workflow, from segmentation to generative editing and seamless blending.

### Features

✔ Click-based object selection using Segment Anything
✔ Smart mask expansion & feathering for smoother edits
✔ AI object removal (LaMa)
✔ AI object replacement with text prompts (Stable Diffusion)
✔ Photorealistic blending using Poisson Image Editing
✔ Interactive Streamlit web app
✔ Modular, production-style Python code

### Project Structure
sam_inpainting_app/
│
├── app.py                  # Streamlit UI
├── sam_utils.py            # Segmentation + mask utilities
├── inpainting_utils.py     # LaMa & Stable Diffusion inpainting
├── blending_utils.py       # Poisson blending functions
├── requirements.txt
├── models/                 # SAM model weights
├── input/                  # Uploaded & intermediate images
├── output/                 # Results
└── README.md

### System Pipeline
User Click → SAM Segmentation → Mask Refinement
        ↓
   Inpainting Choice
   ├── LaMa → Object Removal
   └── Stable Diffusion → Object Replacement
        ↓
Optional Poisson Blending for compositing

### Capabilities Demonstrated
1️⃣ Object Segmentation

Uses Meta's Segment Anything Model to extract precise object masks from simple user clicks.

2️⃣ Mask Refinement

Masks are:

Dilated to cover edges

Feathered using Gaussian blur for soft transitions

3️⃣ Object Removal (LaMa)

AI reconstructs background realistically after object deletion.

4️⃣ Object Replacement (Stable Diffusion)

Generates new objects based on text prompts inside selected regions.

5️⃣ Image Blending

Foreground objects can be blended into new backgrounds using:

Histogram matching

Poisson seamless cloning

### Streamlit App

The UI allows users to:

Upload an image

Provide object click coordinates

Generate segmentation mask

Remove object (LaMa)

Replace object using text prompt (Stable Diffusion)

### Installation
git clone https://github.com/yourusername/sam-inpainting-app.git
cd sam-inpainting-app
pip install -r requirements.txt

### Download SAM Model

Download the SAM checkpoint and place inside models/:

sam_vit_h_4b8939.pth

### Run the App
streamlit run app.py


Open in browser:

http://localhost:8501

### Example Use Cases

Remove unwanted objects from photos

Replace objects with AI-generated ones

Create composite scenes

Background cleanup

Photo editing assistance tools

AI-powered design workflows

### Technologies Used
Area	Technology
Segmentation	Meta Segment Anything
Inpainting	LaMa, Stable Diffusion
Image Processing	OpenCV, NumPy
Blending	Poisson Image Editing
UI	Streamlit
Deep Learning	PyTorch, Diffusers

### Why This Project is Strong

This project showcases:

Computer Vision

Generative AI

Image Processing

Diffusion Models

Model Integration

UI deployment

It bridges research models and real-world applications.

### Future Improvements

Interactive click selection inside Streamlit canvas

Multi-object editing

Automatic mask suggestions

Style transfer blending

Deploy as web app
