import time
import os
import dotenv
import json
import uvicorn
import httpx
from fastapi import FastAPI, Request, HTTPException
from mistralai import Mistral, ImageURLChunk, TextChunk
from mistralai.models.sdkerror import SDKError
from langchain_groq import ChatGroq

dotenv.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

groq_llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}},
    groq_api_key=os.getenv("GROQ_API_KEY")
)

app = FastAPI()

@app.post("/api/extract")
async def extract_route(request: Request):
    """
    Receives a base64-encoded image (first entry in imageDataList) and a language.
    Runs Mistral OCR to extract text and returns a single JSON object.
    """
    body = await request.json()
    imageDataList = body.get("imageDataList", [])
    language = body.get("language", "python")

    if not imageDataList:
        raise HTTPException(status_code=400, detail="imageDataList cannot be empty.")

    base64_data = imageDataList[0]  # Only process the first image

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
                "Extract and convert the given text into a well-structured JSON object. Strictly return a JSON object with 'problemInfo' and 'language' as the only two keys. The value of 'problemInfo' must be a string that contains a JSON-like structure with 'title', 'problem', and 'code' as its components. These components should be simple strings, not nested JSON objects. Ensure proper formatting, correct parsing of lists, and maintain the integrity of the original content. The entire response must be a valid JSON object with no additional text or explanations."
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
            return response_dict  # Return a single JSON object

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

    raise HTTPException(status_code=500, detail="Failed to process image.")

@app.post("/api/generate")
async def generate_route(request: Request):
    """
    Expects a JSON body with:
      "problemInfo": The full text of the coding problem
      "language": The desired solution language (defaults to Python)
    Returns generated code in JSON format.
    """
    print("Incoming request object:", request)  # Print the request object for debugging

    body = await request.json()
    print("Full request body:", body)  # Print the parsed JSON body

    # Ensure problemInfo exists and is a non-empty string
    if not isinstance(body.get("problemInfo"), str) or not body["problemInfo"].strip():
        print("Invalid problemInfo in request body")
        raise HTTPException(status_code=400, detail="problemInfo must be a non-empty string.")
    
    language = body.get("language", "python")
    prompt_text = (
        f"Generate a complete solution in {language}. Return a JSON object with 'Explanation', 'Code', 'Time Complexity', 'Space Complexity', 'complexity_explanation', and 'Problem Information' keys. 'Explanation' should cover brute force, better, and optimal approaches in short paragraphs. 'Code' should be formatted within triple backticks. 'Time Complexity' and 'Space Complexity' should be in Big O notation. 'complexity_explanation' should explain the complexities. 'Problem Information' should include the title and problem description. Ensure proper JSON formatting and no additional commentary. Problem Information: {body['problemInfo']}."
    )
    
    try:
        # Call the ChatGroq LLM
        groq_response = groq_llm.invoke(prompt_text)  # Use .invoke() instead of calling directly
        print("Full Groq Response:", groq_response)  # Print entire response for debugging

        generated_code = groq_response.content  # Extract content correctly
        print("Generated Code:", generated_code)  # Log extracted code

        return {"code": generated_code}
    except Exception as e:
        print("Error generating code:", str(e))  # Log error
        raise HTTPException(status_code=500, detail=str(e))


import socket

if __name__ == "__main__":
    import uvicorn

    # Get the local IP address of the device
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"Backend is running on: http://{local_ip}:3000")

    uvicorn.run("code-assist-backend:app", host="0.0.0.0", port=3000)