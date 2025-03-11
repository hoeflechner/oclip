import json
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import os
import torch
import open_clip
import threading
import time

PORT = int(os.getenv("PORT", "11435"))
MODELNAME = os.getenv(
    "MODELNAME", "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
DEVICE = os.getenv("DEVICE", "cpu")
TIMEOUT = os.getenv("TIMEOUT", "300")

app = Flask(__name__)

class Clip:
    device = 'cpu'
    modelName = ""
    models = {}
    model = None
    preprocess = None
    tokenizer = None
    lastAccessed = time.time()
    background_thread = None

    def __init__(self, modelName):
        self.modelName = modelName

    def load(self):
        if None == Clip.models.get(self.modelName):
            Clip.models[self.modelName] = {}

        modelNode = Clip.models.get(self.modelName)
        modelNode['lastAccessed'] = time.time()
        while (True==modelNode.get('loading',False)):
            time.sleep(1)
        if 'model' in modelNode and 'device' in modelNode and 'tokenizer' in modelNode and 'preprocess' in modelNode:
            return

        modelNode['loading']=True
        print(f"loading model {self.modelName}")
        modelNode['model'], Clip.models[self.modelName]['preprocess'] = open_clip.create_model_from_pretrained(
            self.modelName)
        modelNode['tokenizer'] = open_clip.get_tokenizer(self.modelName)
        if modelNode.get('backgroundThread') == None:
            modelNode['background_thread'] = threading.Thread(
                target=self.background_task)
            modelNode['background_thread'].daemon = True
            modelNode['background_thread'].start()
        try:
            devcount = torch.cuda.device_count()
            if devcount > 0 and DEVICE != "cpu":
                device = DEVICE
                modelNode['model'].to(device)
                freemem = torch.cuda.mem_get_info()[0]
                devicename = torch.cuda.get_device_name(device)
                print(
                    f"loaded on {devicename}, {freemem/1024/1024:.0f} MB left")
            else:
                device = "cpu"
                print(f"loaded on CPU")
        except Exception as e:
            device = 'cpu'
            print(e)
        modelNode['lastAccessed'] = time.time()
        modelNode['device'] = device
        modelNode['loading']=False

    def getDevice(self):
        self.load()
        return Clip.models[self.modelName]['device']

    def getModel(self):
        self.load()
        return Clip.models[self.modelName]['model']

    def getPreprocess(self):
        self.load()
        return Clip.models[self.modelName]['preprocess']

    def getTokenizer(self):
        self.load()
        return Clip.models[self.modelName]['tokenizer']

    def unload(self):
        if None != Clip.models[self.modelName]:
            Clip.models[self.modelName]['model'] = None
            Clip.models[self.modelName]['device'] = None
            Clip.models[self.modelName]['tokenizer'] = None
            Clip.models[self.modelName]['preprocess'] = None
        Clip.models[self.modelName] = {}
        torch.cuda.empty_cache()

    def background_task(self):
        while True:
            modelNode = Clip.models.get(self.modelName)
            if None != modelNode and None != modelNode.get('device') and time.time()-modelNode['lastAccessed'] > int(TIMEOUT):
                device = modelNode['device']
                print(f"unloading model {self.modelName}")
                self.unload()
                if device != 'cpu':
                    freemem = torch.cuda.mem_get_info()[0]
                    devicename = torch.cuda.get_device_name(device)
                    print(
                        f"unloaded from {devicename}, {freemem/1024/1024:.0f} MB left")
                Clip.background_thread = None
                return
            time.sleep(1)


@app.route("/api/embed", methods=['POST'])
def embed():
    startTime = time.time()
    if 1 < len(request.files):
        data = json.load(request.files['data'])
        modelname = data[0]['model']
        clip = Clip(modelname)
        file = request.files['image']
        rawimage = Image.open(file)
        print(f"generating image embeddings {rawimage.filename}")
        image = clip.getPreprocess()(rawimage).unsqueeze(0).to(clip.getDevice())
        with torch.no_grad(), torch.autocast(device_type=clip.getDevice()):
            embeddings = clip.getModel().encode_image(image)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        file.close()
    else:
        data = request.json
        modelname = data['model']
        input = data['input']
        clip = Clip(modelname)
        print(f"generating embeddings for {input}")
        text = clip.getTokenizer()(input).to(clip.getDevice())
        with torch.no_grad(), torch.autocast(device_type=clip.getDevice()):
            embeddings = clip.getModel().encode_text(text)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
    runTime = (time.time()-startTime)*1000
    print(f"embedding done in {runTime} ms")
    ret = {"embeddings": embeddings.cpu().to(torch.float32).numpy().tolist()}
    return ret


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=PORT)
