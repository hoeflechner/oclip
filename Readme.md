# oclip

is a small service that provides the embed endpoint that [ollama](https://ollama.com/blog/embedding-models) does but it will use the [open_clip models](https://huggingface.co/models?library=open_clip) and download these automatically.
 It also work for images - a feature that is currently missing in [ollama](https://github.com/ollama/ollama/issues/5304)

 It is intended to be used until this functionality is available in ollama...

Oclip will unload any models after 300s (default) if they are not used.

It should also be possible to run queries to different models in parallel (as long as (v)ram is available)



## Installation

create an environment, install necessary packages and run

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install Pillow
python src/app.py
```

### Docker

Use the [docker-compose.yml](docker-compose.yml) file in the repository as a starting point.

### GPU 

to utilize a nvidia-gpu start it with:
```
DEVICE=cuda python src/app.py
```

#### Vram usage

```
loading model hf-hub:apple/MobileCLIP-B-OpenCLIP
loaded on NVIDIA GeForce RTX 4060 Ti, 11551 MB left
loading model hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K
loaded on NVIDIA GeForce RTX 4060 Ti, 10875 MB left
```
The process now uses 1636 MB Vram. After 5 min idle:
```
unloading model hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K
unloading model hf-hub:apple/MobileCLIP-B-OpenCLIP
unloaded from NVIDIA GeForce RTX 4060 Ti, 11352 MB left
unloaded from NVIDIA GeForce RTX 4060 Ti, 11966 MB left
```
The process then idles at 260 MB Vram usage



## Usage

See [demo.py](demo.py)

or use curl:

```
curl http://localhost:11435/api/embed -H 'Content-Type: application/json' -d '{
  "model": "hf-hub:apple/MobileCLIP-B-OpenCLIP", 
  "input": "Clip is cool"
}' 
```

returns:

```
{"embeddings":[[-0.048919677734375,0.004100799560546875,-0.006267547607421875,-0.0008993148803710938,0.031524658203125,0.0262908935546875...
```
