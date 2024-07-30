import requests
import torch
from PIL import Image
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1

#
# Load the aesthetics predictor
#
model_id = "shunk031/aesthetics-predictor-v1-vit-large-patch14"

predictor = AestheticsPredictorV1.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

#
# Download sample image
#
url = "/home/azureuser/images/c9a433358305f5aa3644830eb08c8e133fccafe1d685ab4be32fe04a1929ba2b.jpg"
image = Image.open(url)

#
# Preprocess the image
#
inputs = processor(images=image, return_tensors="pt")

#
# Move to GPU
#
device = "cuda"
predictor = predictor.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

#
# Inference for the image
#
with torch.no_grad(): # or `torch.inference_model` in torch 1.9+
    outputs = predictor(**inputs)
prediction = outputs.logits

print(f"Aesthetics score: {prediction}")