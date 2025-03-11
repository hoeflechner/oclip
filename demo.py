import random
import numpy as np
import requests
import json
import torch


# This sample code will use the oclip server to compare one image to
# an array of texts and print out the probapliity that each text 
# matches the image


### set model (from https://huggingface.co/models?library=open_clip)
model="hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
### set image
image = "bus.jpg"
### set texts to compare image to
texts=["a bus in a garage","a cat","two men in front of a blue bus", "a bus on a busy street"]


### shuffle the texts
random.shuffle(texts)

### get embeddings for image
file = open(image, 'rb')
data = {"model": model},
files = [
    ('image', (image, file, 'application,octet')),
    ('data', ('data', json.dumps(data), 'application/json'))
]
response = requests.post(
    "http://localhost:11435/api/embed",
    files=files,
    timeout=20
)
file.close()
embeddings = response.json()['embeddings']
image_features=torch.tensor(embeddings[0])
print(f"image embeddings: {image_features.shape}")

### get embeddings for texts
response = requests.post(
    "http://localhost:11435/api/embed",
    json={'model': model,
          "input": texts}
)
embeddings = response.json()['embeddings']
text_features=torch.tensor(embeddings)
print(f"texts embeddings: {text_features.shape}")

### calculate probailitys
probs = (100 * image_features @ text_features.T).softmax(dim=-1)

### print table
print("the image shows:")
for i, t in enumerate(texts):
    print("{:<40}: {:.1f}%".format(t,probs[i]*100))
