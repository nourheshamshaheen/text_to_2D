from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests


# Initialize model

def flamingo_setup():
    model, processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="aleksickx/llama-7b-hf",
        tokenizer_path="aleksickx/llama-7b-hf",
        cross_attn_every_n_layers=4
    )
    # grab model checkpoint from huggingface hub
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    return model, processor, tokenizer

def flamingo_predict(query_image, path, model, image_processor, tokenizer):
    # Steps
    """
    Step 1: Load 10 images
    """
    # here, path is the path to celeba which will differ from computer to computer

    image1 = Image.open(path + "032774.jpg")
    image2 = Image.open(path + "032775.jpg")
    image3 = Image.open(path + "032776.jpg")
    image4 = Image.open(path + "032777.jpg")
    image5 = Image.open(path + "032778.jpg")
    image6 = Image.open(path + "032779.jpg")
    image7 = Image.open(path + "032780.jpg")
    image8 = Image.open(path + "032781.jpg")
    image9 = Image.open(path + "032782.jpg")
    image10 = Image.open(path + "032783.jpg")
    # image11 = Image.open(path + "032784.jpg")
    # image12 = Image.open(path + "032785.jpg")
    # image13 = Image.open(path + "032786.jpg")
    # image14 = Image.open(path + "032787.jpg")
    # image15 = Image.open(path + "032788.jpg")

    images = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, query_image]

    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = 1, num_media = 3, num_frames = 1 
    (this will always be one expect for video which we don't support yet), 
    channels = 3, height = 224, width = 224.
    """

    vision_x = [image_processor(image).unsqueeze(0) for image in images]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)


    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = tokenizer(
        ["<image>The person in the image is a young woman with a warm and friendly smile. She has dark hair and is wearing a strapless dress, which adds a touch of elegance to her appearance. The woman is also wearing red lipstick, which complements her dark hair and enhances her overall look. She is not wearing any other noticeable makeup or accessories. Her expression and demeanor suggest a positive and approachable individual.<|endofchunk|> \
         <image>The person in the image is a young woman with dark hair, possibly of Hispanic origin. She has a serious look on her face and is wearing minimal makeup, which includes dark eyebrows and a dark line around her lips. The woman is not wearing any accessories, such as earrings or a necklace, and she has a natural, unadorned appearance. Her gaze is directed towards the camera, capturing the viewer's attention.<|endofchunk|> \
         <image>The person in the image is a middle-aged Asian man wearing glasses, a suit, and a blue tie. He has a professional appearance and is smiling, which suggests a friendly and approachable demeanor. The man is posing for a picture, and it seems like he is dressed for a formal or business event. There is no additional information about any other accessories or makeup he might be wearing.<|endofchunk|> \
         <image>The person in the image is a middle-aged man with a beard, who is wearing glasses. He is of Mediterranean descent and appears to be a professional singer, as he is holding a microphone and singing into it. The man is dressed in black, and his attire suggests that he might be performing on stage or in a recording studio. He is the focal point of the image, and his expression and posture convey his passion and dedication to his music.<|endofchunk|> \
         <image>The person in the image is a middle-aged man with a beard and glasses. He is wearing a suit and tie, giving him a professional appearance. The man is sitting down and talking to the camera, possibly giving a speech or presenting information. There is a flag in the background, suggesting the setting could be a formal event or an important location. The man's attire and demeanor indicate that he might be a public figure or an expert in his field.<|endofchunk|> \
         <image>The person in the image is a young woman with a tan complexion. She is wearing a blue shirt and has a headband on her head. She is posing for the camera and smiling, showing off her pearly whites. The woman is also wearing earrings, which adds to her stylish appearance. Her expression and posture convey a friendly and approachable demeanor.<|endofchunk|> \
         <image>The person in the image is a woman with long, dark hair, and she is smiling at the camera. She has a warm and inviting demeanor. The woman is wearing earrings, which adds a touch of elegance to her appearance. Additionally, she is wearing a white shirt, which complements her overall look. The image is a headshot, capturing her from the shoulders up, and it is framed in a square format.<|endofchunk|> \
         <image>The person in the image is a young Asian woman. She is wearing a coat and a scarf, which suggests that she is dressed for colder weather or wants to keep warm. The woman is also wearing a jacket, further emphasizing her attire's focus on warmth and protection. Additionally, she is wearing a brown jacket, which complements her overall outfit. The woman is posing for the camera, and her expression appears calm and composed. She is not wearing any makeup or accessories that are particularly noticeable in the image.<|endofchunk|> \
         <image>The person in the image is a young woman of Asian descent, wearing a black hat and glasses. She has a hipster-style appearance and is striking a pose for the camera. Her outfit includes a black top, which complements her accessories. The woman's makeup includes dark eyeliner and pink lips, adding to her stylish and fashionable look. Overall, she exudes a confident and trendy vibe.<|endofchunk|> \
         <image>The person in the image is a young, black man with a warm and friendly smile. He is clean-shaven and wearing a black shirt. His facial expression is joyful, and he appears to be in a good mood. There are no notable accessories or makeup visible on the man in the image.<|endofchunk|> \
         <image>The person in the image is "],
        return_tensors="pt",
    )

    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=128,
        num_beams=3,
    )

    print("Generated text: ", tokenizer.decode(generated_text[0]))



m, p, t = flamingo_setup()
path = "/home/nour.shaheen/Downloads/img_align_celeba/img_align_celeba/"
image_path = "/home/nour.shaheen/Documents/imgs/000001.jpg"
query_image = Image.open(image_path)
flamingo_predict(query_image, path, m, p, t)