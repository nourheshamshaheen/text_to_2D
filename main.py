import argparse
from generate import Generator

def main(args):
    model_name = args.model_name
    gen = Generator(model_name=model_name)
    filename = args.question_list
    with open(filename) as file:
        questions = [line.rstrip() for line in file]
    all = args.all
    if all:
        gen.generate_all(image_folder=args.images, questions=questions, output_file=args.output_file)
    else:
        gen.generate(image_path=args.images, questions=questions, output_file=args.output_file)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model name you want to use", type=str, choices=["vilt", "v_chatgpt"])
parser.add_argument("--images", help="Path to your image or image folder", type=str)
parser.add_argument("--question_list", help="List of questions you want to ask to your VQA model", type=str)
parser.add_argument("--output_file", help="Output CSV file to add your dataset to", type=str)
parser.add_argument("--all", help="If you want to generate one image or an entire folder", action='store_true')
args = parser.parse_args()

main(args)


