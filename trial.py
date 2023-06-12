import torch
from promptcap import PromptCap_VQA, PromptCap

# QA model support all UnifiedQA variants. e.g. "allenai/unifiedqa-v2-t5-large-1251000"
vqa_model = PromptCap_VQA(promptcap_model="vqascore/promptcap-coco-vqa", qa_model="allenai/unifiedqa-t5-base")

if torch.cuda.is_available():
  vqa_model.cuda()

prompt = "Describe this person."
image = "/home/nour.shaheen/Documents/vqa/text_to_2D/trial.jpeg"

print(vqa_model.vqa(prompt, image))


# model = PromptCap("vqascore/promptcap-coco-vqa")  # also support OFA checkpoints. e.g. "OFA-Sys/ofa-large"

# if torch.cuda.is_available():
#   model.cuda()

# prompt = "What is this person's ethnicity?"
# image = "/home/nour.shaheen/Documents/vqa/text_to_2D/trial.jpeg"

# print(model.caption(prompt, image))