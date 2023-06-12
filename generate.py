from vqa import VQA
import os
from csv import writer

class Generator:
    def __init__(self, args) -> None:
        self.vqa_object = VQA(args)
        self.vqa_object.setup()

    def generate_all(self, image_folder, questions, output_file) -> None:
        image_paths = os.listdir(image_folder)
        for image_path in image_paths:
            self.generate(image_path=image_path, questions=questions, output_file=output_file)
            
    def generate(self, image_path, questions, output_file) -> None:
        answers = []
        for question in questions:
            answer = self.vqa_object.answer(image_path, question)
            answers.append(answer)
        self.append(image_path, answers, output_file)

    def append(self, image_path, answers, output_file) -> None:
        with open(output_file, 'a') as f_object:
            description = " ".join(answers)
            row = [image_path, description]
            writer_object = writer(f_object)
            writer_object.writerow(row)
            f_object.close()