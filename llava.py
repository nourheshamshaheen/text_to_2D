import os
import argparse

from selenium import webdriver
import selenium
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TextAreaTextStable:
    def __init__(self, locator):
        self.locator = locator
        self.previous_text = None

    def __call__(self, driver: webdriver.Firefox):
        try:
            current_text = driver.find_element(*self.locator).text
        except selenium.common.exceptions.StaleElementReferenceException:
            return False
        if current_text == self.previous_text:
            return current_text
        self.previous_text = current_text
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web automation script for LLaVA')
    parser.add_argument('--img', type=str, help='Path to the image file')
    parser.add_argument('--prompt', type=str, help='Prompt for VQA', default='Describe this person to me. What is their ethnicity, gender, general temperament, age? Are they wearing any accessories or makeup? Describe their appearance in detail.')
    args = parser.parse_args()

    img_path: str = os.path.abspath(args.img)
    print(img_path)
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
    WebDriverWait(driver, 10).until(
        EC.text_to_be_present_in_element((By.CSS_SELECTOR, model_selector), "LLaVA") # This is the model selector, wait for it to show currently selected model
    )

    # Find the input element and set the file path
    img_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"]')
    # img_input.clear()
    img_input.send_keys(img_path)

    # waiting for the image to upload
    WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CSS_SELECTOR, loaded_image_selector)))

    # waiting for the textarea to be visible
    textarea = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, prompt_selector)))
    textarea.clear()
    textarea.send_keys(prompt)

    # Find the submit button and click it
    submit_button = driver.find_element(By.ID, submit_btn_id)
    submit_button.click()

    output_paragraph = WebDriverWait(driver, 30).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, output_selector)) # This is the output paragraph
    )

    wait = WebDriverWait(driver, 10).until(TextAreaTextStable((By.CSS_SELECTOR, output_selector)))
    
    # frontend keeps updating UI elements, that's why references get stale, need to search for them again and again
    # JavaScript exists to spite God
    output_paragraph = driver.find_element(By.CSS_SELECTOR, output_selector)
    print(output_paragraph.text)

    # div.message:nth-child(2) > p:nth-child(1)
    # img.svelte-rlgzoo
    # Close the browser
    driver.quit()