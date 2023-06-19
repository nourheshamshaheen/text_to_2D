from paraphraser import Paraphraser
import argparse
import pandas as pd
import os
from csv import writer


def fix(df):
    for i in range(len(df)):
        t = df.iloc[i, 0]
        df.iloc[i, 0] = os.path.basename(t)   
    return df

def main(args):
    parr = Paraphraser(args)
    parr.setup()

    in_df = pd.read_csv(args.input_file)

    if args.fix_path:
        out_df = fix(in_df)
    else:
        out_df = in_df

    if args.all:
        for i in range(len(out_df)):
            img_id = out_df.iloc[i, 0]
            text = out_df.iloc[i, 1]
            query = args.query + text
            ans = parr.paraphrase(query)
            append(img_id, ans, args.output_file)
    else:
        img_id = out_df.iloc[0, 0]
        text = out_df.iloc[0, 1]
        query = args.query + text
        ans = parr.paraphrase(query)
        append(img_id, ans, args.output_file)
        print(ans)


def append(image_id, answer, output_file) -> None:
    with open(output_file, 'a') as f_object:
        row = [image_id, answer]
        writer_object = writer(f_object)
        writer_object.writerow(row)
        f_object.close()


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model name you want to use", type=str, choices=["llama", "gpt", "llava"])
parser.add_argument("--llava_weights", help="(Optional) Path to Llava Weights, if model is LLava", default="/home/nour.shaheen/Documents/vqa/text_to_2D/LLaVA-7B-Lightening-v1-1/", type=str)
parser.add_argument("--api_key", help="(Optional) API key if model is GPT", default="", type=str)
parser.add_argument("--temp_img", help="black image to input to llava", default="/home/nour.shaheen/Documents/vqa/text_to_2D/black.png", type=str)
parser.add_argument("--input_file", help="Input CSV file that has your unprocessed dataset", type=str)
parser.add_argument("--output_file", help="Output CSV file to add your dataset to", default="", type=str)
parser.add_argument("--query", help="Query to input to the summarizer/text generation model", default="Summarize without losing any important information, in 50 words or less: ", type=str)
parser.add_argument("--all", help="If you want to summarize one line or the entire csv", action='store_true')
parser.add_argument("--fix_path", help="If you want to cut the path of the image to be only the image ID", action='store_true')
args = parser.parse_args()

main(args)
