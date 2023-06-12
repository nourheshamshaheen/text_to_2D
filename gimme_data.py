import os
import argparse
from generate import Generator
from csv import writer


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

class TextAreaTextStable:
    def __init__(self, locator):
        self.locator = locator
        self.previous_text = None

    def __call__(self, driver: webdriver.Firefox):
        try:
            current_text = driver.find_element(*self.locator).text

        except Exception:
            return False
        except StaleElementReferenceException as e:
            print(f"An error occurred: {e}")
            return False
        if current_text == self.previous_text:
            return current_text
        self.previous_text = current_text

        
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web automation script for LLaVA')
    #parser.add_argument('--img', type=str, help='Path to the image file')
    parser.add_argument('--prompt', type=str, help='Prompt for VQA', default='What is the eye shape and any other relevant eye details, nose size, nose width and height, nasal shape, lip thickness, lip shape, facial shape and width, ear shape and position, eyebrow shape, eyebrow density, accessories worn, facial hair, teeth arrangement (if shown), hair type, hair style, hair color, race, gender of the person, and is this person a child, a teenager, a young adult, an adult, or an elderly?'

)
    parser.add_argument('--output', type=str, help='Path to the output file')
    args = parser.parse_args()

    
            

    for i in range(2835,20000):
        with open(args.output, 'a') as f_object:

            i_str = str(i+1).zfill(5)

            data_path: str = os.path.abspath("flickr_faces/"+i_str+".png")
            print(data_path)
            prompt: str = args.prompt

            # DO NOT CHANGE
            submit_btn_id = 'component-22'
            output_selector = 'div.message:nth-child(2) > p:nth-child(1)'
            prompt_selector = '.scroll-hide'
            model_selector = '.single-select'
            loaded_image_selector = '#component-8 img'


            # Set the path to your Chrome WebDriver executable
            # webdriver_path = 'chromedriver_linux64/chromedriver'

            # Create a new instance of the Chrome driver

            driver = webdriver.Chrome()
            # options = webdriver.FireFoxOptions()
            # options.headless = True

            # Load the web page
            driver.get('https://llava.hliu.cc/')


            # Wait for the DOM to finish generating
            WebDriverWait(driver, 20).until(
                EC.text_to_be_present_in_element((By.CSS_SELECTOR, model_selector), "LLaVA") # This is the model selector, wait for it to show currently selected model
            )

            # Find the input element and set the file path
            img_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
            # img_input.clear()
            img_input.send_keys(data_path)

            # waiting for the image to upload
            try:
                WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, loaded_image_selector)))
            except TimeoutError as e:
                print(f"An error occurred: {e}")

            # waiting for the textarea to be visible
            try:
                textarea = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, prompt_selector)))
            except TimeoutError as e:
                print(f"An error occurred")
                
            textarea.clear()
            textarea.send_keys(prompt)

            # Find the submit button and click it
            submit_button = driver.find_element(By.ID, submit_btn_id)
            submit_button.click()
            try: 
                output_paragraph = WebDriverWait(driver, 60).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, output_selector)) # This is the output paragraph
                )
            except TimeoutError as e:
                print(f"An error occurred: {e}")

            wait = WebDriverWait(driver, 90).until(TextAreaTextStable((By.CSS_SELECTOR, output_selector)))
            
            # frontend keeps updating UI elements, that's why references get stale, need to search for them again and again
            # JavaScript exists to spite God
            output_paragraph = driver.find_element(By.CSS_SELECTOR, output_selector)
            

            #description = " ".join(answers)
            #row = [image_path, description]
            writer_object = writer(f_object)
            f_object.write('{} '.format(i+1))
            try:
                f_object.write(output_paragraph.text)
            except StaleElementReferenceException as e:
                print(f"An error occurred: {e}")

            f_object.write('\n')
            #f_object.close()

            # div.message:nth-child(2) > p:nth-child(1)
            # img.svelte-rlgzoo
            # Close the browser
            driver.quit()
