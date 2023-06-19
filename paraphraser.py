from LLaVA.llava.eval.run_llava import eval_setup_text, eval_actual_text
from argparse import Namespace
import openai

class Paraphraser:

    def __init__(self, args) -> None:
        self.model_name = args.model_name
        self.llava_weights = args.llava_weights
        self.img = args.temp_img
        self.open_ai_key = args.api_key


    def setup(self):
        if self.model_name == "gpt":
            openai.api_key = self.open_ai_key
        elif self.model_name == "llava":
            print("Setting up LLAVA for summarization...")
            args = Namespace(model_name=self.llava_weights, image_file=self.img, conv_mode=None)
            self.tokenizer, self.image_tensor, self.model, self.conv_mode, self.mm_use_im_start_end, self.image_token_len = eval_setup_text(args)
            print("LLAVA SETUP DONE!")

    def paraphrase(self, text):
        if self.model_name == "llava":
            args = Namespace(query=text, conv_mode=self.conv_mode)
            answer = eval_actual_text(args, self.tokenizer, self.image_tensor, self.model, self.mm_use_im_start_end, self.image_token_len)
            return answer
        elif self.model_name == "gpt":
            completion0 = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                               messages=[
                                                   {
                                                       "role": "user",
                                                       "content": f"Summarize without losing any important information, in between 45 to 50 words: {text}. "
                                                   }]
            )
            result0 = ''
            for choice in completion0.choices:
                result0 += choice.message.content
            return result0


