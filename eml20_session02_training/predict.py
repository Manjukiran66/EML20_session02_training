from typing import Any
from cog import BasePredictor, Input, Path
from eml20_session02_training.model.cfar_module import LitResnet
import json
import torch
import numpy as np

from PIL import Image


 
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        PATH = "/workspace/EML20_session02_training/EML20_session02_training/eml20_session02_training/logs/train/runs/2022-09-09_17-52-44/checkpoints/epoch_000.ckpt"
        self.model = LitResnet.load_from_checkpoint(PATH)
        self.model.eval()
        self.transform = transforms_imagenet_eval()

        with open("imagenet_1k.json", "r") as f:
            self.labels = list(json.load(f).values())

    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model"""
        # Preprocess the image
        img = Image.open(image).convert('RGB')
        img = self.transform(img)

        # Run the prediction
        with torch.no_grad():
            labels = self.model(img[None, ...])
            labels = labels[0] # we'll only do this for one image

        # top 5 preds
        topk = labels.topk(5)[1]
        output = {
            # "labels": labels.cpu().numpy(),
            "topk": [self.labels[x] for x in topk.cpu().numpy().tolist()],
        }

        return output