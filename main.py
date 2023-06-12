import argparse
from generate import Generator

def main(args):
    model_name = args.model_name
    path_to_celeba = args.path_to_celeba
    gen = Generator(model_name=model_name, path_to_celeba=path_to_celeba)
    filename = args.question_list
    if args.model_name == "vilt":
        with open(filename) as file:
            questions = [line.rstrip() for line in file]
    else: 
        questions = [0]
    all = args.all
    if all:
        gen.generate_all(image_folder=args.images, questions=questions, output_file=args.output_file)
    else:
        gen.generate(image_path=args.images, questions=questions, output_file=args.output_file)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model name you want to use", type=str, choices=["vilt", "flamingo", "llava"])
parser.add_argument("--images", help="Path to your image or image folder", type=str)
parser.add_argument("--question_list", help="List of questions you want to ask to your VQA model", type=str)
parser.add_argument("--output_file", help="Output CSV file to add your dataset to", type=str)
parser.add_argument("--api_key", help="GPT4's api key, optional", type=str, default=None)
parser.add_argument("--path_to_celeba", type=str, default="/home/nour.shaheen/Downloads/img_align_celeba/img_align_celeba/")
parser.add_argument("--all", help="If you want to generate one image or an entire folder", action='store_true')
args = parser.parse_args()

main(args)


