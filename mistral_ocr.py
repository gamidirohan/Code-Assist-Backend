import os
import dotenv
from mistralai import Mistral
from mistralai import ImageURLChunk, TextChunk
import json
import base64
import httpx

dotenv.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

def process_images(imageDataList, language):
    """
    Processes multiple base64-encoded images with Mistral OCR.
    Returns a list of Python dicts containing extracted info.
    """
    results = []
    for base64_data in imageDataList:
        # Retry up to 3 times for each image
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                image_response = client.ocr.process(
                    document=ImageURLChunk(image_url=f"data:image/jpeg;base64,{base64_data}"),
                    model="mistral-ocr-latest"
                )
                image_ocr_markdown = image_response.pages[0].markdown

                prompt_text = (
                    f"This image's OCR in markdown:\n<BEGIN_IMAGE_OCR>\n{image_ocr_markdown}\n<END_IMAGE_OCR>.\n"
                    f"Language expected: {language}\n"
                    "Extract and convert the given text into a well-structured JSON object. Ensure proper formatting, correct parsing of lists, and maintain the integrity of the original content. Strictly return a JSON object with no additional commentary or explanations."
                )

                chat_response = client.chat.complete(
                    model="pixtral-12b-latest",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                ImageURLChunk(image_url=f"data:image/jpeg;base64,{base64_data}"),
                                TextChunk(text=prompt_text)
                            ],
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )

                response_dict = json.loads(chat_response.choices[0].message.content)
                results.append(response_dict)
                break  # Success, exit the retry loop

            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
                attempt += 1
                if attempt == max_retries:
                    raise e  # Reraise if all attempts fail

    return results


def main():
    with open("leetcode_test.png", "rb") as image_file:
        encoded_screenshot = base64.b64encode(image_file.read()).decode("utf-8")
    print(process_images([encoded_screenshot], "python"))

if __name__ == "__main__":
    main()

