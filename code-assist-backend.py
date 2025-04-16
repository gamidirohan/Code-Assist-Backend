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

    max_retries = 5  # Increased from 3
    attempt = 0
    backoff_time = 2  # Increased from 1

    while attempt < max_retries:
        try:
            # First try OCR
            try:
                image_response = client.ocr.process(
                    document=ImageURLChunk(image_url=f"data:image/jpeg;base64,{base64_data}"),
                    model="mistral-ocr-latest"
                )
                image_ocr_md = image_response.pages[0].markdown
            except SDKError as ocr_err:
                if "Requests rate limit exceeded" in str(ocr_err):
                    print(f"OCR rate limit exceeded, attempt {attempt+1}/{max_retries}")
                    attempt += 1
                    if attempt == max_retries:
                        return {
                            "problemInfo": "Unable to process image due to API rate limits. Please try again later.",
                            "language": language
                        }
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue
                else:
                    raise ocr_err
            
            # Then try chat completion
            prompt_text = (
                f"This image's OCR in markdown:\n<BEGIN_IMAGE_OCR>\n{image_ocr_md}\n<END_IMAGE_OCR>.\n"
                f"Language expected: {language}\n"
                "Carefully extract ALL information from the image, including the complete problem statement, examples, constraints, and any other details. Pay special attention to code examples, input/output formats, and edge cases mentioned or implied in the problem. Take your time to think all the possible edge cases.\n\n"
                "Return a valid, properly formatted JSON object with exactly two root keys: 'problemInfo' and 'language'. The 'problemInfo' key should contain a detailed string with ALL the problem information including examples and constraints exactly as given in the image. The 'language' key should contain a string with the programming language. Do not include nested JSON objects or arrays. Ensure all quotes are properly escaped. Return only the JSON object with no additional text."
            )

            try:
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
            except SDKError as chat_err:
                if "Requests rate limit exceeded" in str(chat_err):
                    print(f"Chat rate limit exceeded, attempt {attempt+1}/{max_retries}")
                    attempt += 1
                    if attempt == max_retries:
                        if 'image_ocr_md' in locals():
                            return {
                                "problemInfo": f"API rate limited. Raw OCR text: {image_ocr_md[:500]}...",
                                "language": language
                            }
                        else:
                            return {
                                "problemInfo": "Unable to process image due to API rate limits. Please try again later.",
                                "language": language
                            }
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue
                else:
                    raise chat_err
            
            try:
                response_dict = json.loads(chat_response.choices[0].message.content)
                return response_dict
            except json.JSONDecodeError as json_err:
                print(f"JSON decode error: {json_err}")
                print(f"Raw content: {chat_response.choices[0].message.content}")
                fallback_content = chat_response.choices[0].message.content
                return {
                    "problemInfo": f"Error parsing response. Raw content: {fallback_content[:500]}...",
                    "language": language
                }

        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
            attempt += 1
            if attempt == max_retries:
                raise HTTPException(status_code=503, detail="Network error when connecting to API services.")
            time.sleep(backoff_time)
            backoff_time *= 2

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    raise HTTPException(status_code=500, detail="Failed to process image after maximum retries.")

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