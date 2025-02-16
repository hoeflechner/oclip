import numpy as np
import requests
import json

### get embeddings for image
image = "bus.jpg"
file = open(image, 'rb')
data = {"model": "hf-hub:apple/MobileCLIP-B-OpenCLIP"},
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

print(response)
embeddings = response.json()['embeddings']
print(f"embeddings: {np.asarray(embeddings).shape}")

### get embeddings for string
response = requests.post(
    "http://localhost:11435/api/embed",
    json={'model': "hf-hub:apple/MobileCLIP-B-OpenCLIP",
          "input": "Sample Document goes here"}
)
print(response)
embeddings = response.json()['embeddings']
print(f"embeddings: {np.asarray(embeddings).shape}")

### get embeddings for multiple strings
response = requests.post(
    "http://localhost:11435/api/embed",
    json={'model': "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
          "input": ["line one", "line two"]}
)
print(response)
embeddings = response.json()['embeddings']
print(f"embeddings: {np.asarray(embeddings).shape}")
# print(json.dumps(response.json(), indent=2))
