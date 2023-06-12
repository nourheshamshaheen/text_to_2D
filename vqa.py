from transformers import pipeline
from transformers import ViltProcessor, ViltForQuestionAnswering
from flamingo import flamingo_setup, flamingo_predict
from PIL import Image
import requests
import subprocess
import torch
import os
from LLaVA.llava.eval.run_llava import eval_model
from argparse import Namespace

class VQA:
    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.path = args.path_to_celeba
        self.llava_weights = args.llava_weights

    def setup(self) -> None:
        if self.model_name == "vilt":
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        elif self.model_name == "flamingo":
            self.model, self.processor, self.tokenizer = flamingo_setup()
    
    def answer(self, image_path, text) -> str:
        image = Image.open(image_path)
        if self.model_name == "vilt":
            encoding = self.processor(image, text, return_tensors="pt")

            # forward pass
            outputs = self.model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            return self.model.config.id2label[idx]
    
        elif self.model_name == "flamingo":
            return flamingo_predict(image, self.path, self.model, self.processor, self.tokenizer)
        
        elif self.model_name == "llava":
            args = Namespace(model_name=self.llava_weights, image_file=image_path, query=text, conv_mode=None)
            answer = eval_model(args)
            return answer
            # command = f"python -m llava.eval.run_llava --model-name {self.llava_weights} --image-file {image_path} --query {text}"
            # with os.popen(command) as f:
            #     answer = f.readlines()
            #     print("Answer is:", answer)
            #     return answer
        return None


