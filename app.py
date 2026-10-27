import json
from pathlib import Path
import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="ISL Gesture Recognition", page_icon="🤟")
st.title("AI-Based Indian Sign Language Recognition")
st.write("Prototype: image classification only. This does not yet assess movement or facial features.")

def build_model(n):
    return nn.Sequential(
        nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64*16*16,128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128,n)
    )

checkpoint_path=Path("artifacts/best_model.pt")
classes_path=Path("artifacts/classes.json")
if not checkpoint_path.exists() or not classes_path.exists():
    st.warning("No trained model found yet. Add your dataset and run train.py first.")
    st.stop()
classes=json.loads(classes_path.read_text(encoding="utf-8"))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=build_model(len(classes)).to(device)
ckpt=torch.load(checkpoint_path,map_location=device,weights_only=False)
model.load_state_dict(ckpt["state_dict"]); model.eval()
uploaded=st.file_uploader("Upload a labelled-dataset-style gesture image", type=["png","jpg","jpeg"])
if uploaded:
    image=Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)
    tf=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),
                           transforms.Normalize((0.5,)*3,(0.5,)*3)])
    with torch.no_grad():
        logits=model(tf(image).unsqueeze(0).to(device))
        probs=torch.softmax(logits,dim=1)[0]
        idx=int(probs.argmax().item())
    st.subheader(f"Predicted class: {classes[idx]}")
    st.metric("Model confidence", f"{float(probs[idx])*100:.1f}%")
    st.caption("Confidence is the model's score, not a guarantee that the prediction is correct.")
    st.write("Class scores")
    st.bar_chart({classes[i]:float(probs[i]) for i in range(len(classes))})
