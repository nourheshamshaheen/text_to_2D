from transformers import pipeline
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests



class VQA:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def setup(self) -> None:
        if self.model_name == "vilt":
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        else:
            self.processor = None
            self.model = None
        assert self.processor is not None
        assert self.model is not None

    
    def answer(self, image_path, text) -> str:
        image = Image.open(image_path)
        encoding = self.processor(image, text, return_tensors="pt")

        # forward pass
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.model.config.id2label[idx]
