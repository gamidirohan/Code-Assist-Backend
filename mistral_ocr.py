import time
import os
import dotenv
import json
import httpx
from fastapi import FastAPI, Request
from mistralai import Mistral, ImageURLChunk, TextChunk
from mistralai.models.sdkerror import SDKError

dotenv.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

app = FastAPI()

@app.post("/api/extract")
async def extract_route(request: Request):
    """
    Receives base64-encoded images (imageDataList) and a language.
    Runs Mistral OCR to extract text and returns structured JSON data.
    """
    body = await request.json()
    imageDataList = body.get("imageDataList", [])
    language = body.get("language", "python")
    results = []
    for base64_data in imageDataList:
        max_retries = 3
        attempt = 0
        backoff_time = 1  # Initial backoff time in seconds
        while attempt < max_retries:
            try:
                image_response = client.ocr.process(
                    document=ImageURLChunk(image_url=f"data:image/jpeg;base64,{base64_data}"),
                    model="mistral-ocr-latest"
                )
                image_ocr_md = image_response.pages[0].markdown
                prompt_text = (
                    f"This image's OCR in markdown:\n<BEGIN_IMAGE_OCR>\n{image_ocr_md}\n<END_IMAGE_OCR>.\n"
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
                break  # Exit the retry loop on success
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
                attempt += 1
                if attempt == max_retries:
                    raise e
                time.sleep(backoff_time)
            except SDKError as e:
                if "Requests rate limit exceeded" in str(e):
                    attempt += 1
                    if attempt == max_retries:
                        raise e
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    raise e

        time.sleep(1)  # Delay to respect the rate limit of 1 request per second

    print("OCR Results:", json.dumps(results, indent=4))  # Print the OCR results in the terminal
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mistral_ocr:app", host="0.0.0.0", port=3000)