import argparse
from generate import Generator

def main(args):
    gen = Generator(args)
    if args.model_name == "vilt":
        filename = args.question_list
        with open(filename) as file:
            questions = [line.rstrip() for line in file]
    elif args.model_name == "flamingo": 
        questions = [0] # here, question is a list because generator deals with questions as multiple ones
    elif args.model_name == "llava":
        questions = [args.llava_prompt] # here, question is a list because generator deals with questions as multiple ones   
    all = args.all
    if all:
        gen.generate_all(image_folder=args.images, questions=questions, output_file=args.output_file)
    else:
        gen.generate(image_path=args.images, questions=questions, output_file=args.output_file)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model name you want to use", type=str, choices=["vilt", "flamingo", "llava"])
parser.add_argument("--images", help="Path to your image or image folder", type=str)
parser.add_argument("--llava_prompt", help="(Optional) Prompt for your LLava model, if model is LLava", default="", type=str)
parser.add_argument("--llava_weights", help="(Optional) Path to Llava Weights, if model is LLava", default="/home/nour.shaheen/Documents/vqa/text_to_2D/LLaVA-7B-Lightening-v1-1/", type=str)
parser.add_argument("--question_list", help="(Optional) List of questions you want to ask to your VQA model, if model is vilt", default ="", type=str)
parser.add_argument("--output_file", help="Output CSV file to add your dataset to", type=str)
parser.add_argument("--path_to_celeba", type=str, default="/home/nour.shaheen/Downloads/img_align_celeba/img_align_celeba/")
parser.add_argument("--all", help="If you want to generate one image or an entire folder", action='store_true')
args = parser.parse_args()

main(args)


