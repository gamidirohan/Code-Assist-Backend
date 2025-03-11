import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# --- Dummy External API Clients ---

class MistralOCRClient:
    def __init__(self, api_key, timeout=20):
        self.api_key = api_key
        self.timeout = timeout

    async def ocr_process(self, image_url: str):
        # Simulate an asynchronous OCR API call
        await asyncio.sleep(1)  # simulate network delay
        # Return dummy OCR response with markdown content
        return {
            "pages": [{
                "markdown": (
                    f"Extracted text from {image_url}\n"
                    "function: dummy_function()\n"
                    "```python\nprint('Hello, World!')\n```"
                )
            }]
        }

class GroqClient:
    def __init__(self, api_key):
        self.api_key = api_key

    async def generate_code(self, prompt: str, model: str = "qwen-2.5-coder-32b"):
        await asyncio.sleep(1)  # simulate API call delay
        return {"code": f"# Generated code based on prompt:\n# {prompt}"}

    async def debug_code(self, prompt: str, model: str = "llama-3.2-11b-vision-preview"):
        await asyncio.sleep(1)  # simulate API call delay
        return {"debuggedOutput": f"# Debugged output for prompt:\n# {prompt}"}

# Instantiate clients using environment variables (or dummy keys)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not GROQ_API_KEY or not MISTRAL_API_KEY:
    logging.warning("Missing API keys for GROQ or MISTRAL.")
groq_client = GroqClient(GROQ_API_KEY)
mistral_client = MistralOCRClient(MISTRAL_API_KEY, timeout=20)

# --- Helper Functions for Text Extraction ---

# Define schema for structured extraction
schema = {
    "question": str,
    "code_syntax": str,
    "example_test_cases": str,
    "constraints": str
}

# Initialize Groq model
llm = ChatGroq(
    model_name="qwen-2.5-coder-32b",
    temperature=0.0,
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Create parser
parser = JsonOutputParser(pydantic_object=schema)

# Define the prompt template for extraction
extraction_prompt = ChatPromptTemplate.from_template(
    """This is an image's OCR output in markdown format:
<BEGIN_IMAGE_OCR>
{image_markdown}
<END_IMAGE_OCR>

Convert this into a structured JSON object with the following fields:
- question
- code_syntax
- example_test_cases
- constraints

Ensure the output is strictly JSON with no extra text."""
)

# Create processing pipeline
chain = extraction_prompt | llm | parser

def extract_problem_info(image_markdown):
    structured_data = chain.invoke({"image_markdown": image_markdown})
    return structured_data  # Returns structured JSON

# --- Request Body Models ---

class ExtractRequest(BaseModel):
    imageDataList: list
    language: str

class GenerateRequest(BaseModel):
    problemInfo: str
    language: str

class DebugRequest(BaseModel):
    code: str

# --- Endpoints ---

@app.post("/api/extract")
async def extract(body: ExtractRequest, request: Request):
    logging.info(f"Received /api/extract request: {body}")
    if not body.imageDataList or not body.language:
        logging.error("Missing imageDataList or language")
        raise HTTPException(status_code=400, detail="imageDataList and language are required.")

    async def process_image(image_data):
        # Convert to URL format
        image_url = f"data:image/jpeg;base64,{image_data}"
        ocr_response = await mistral_client.ocr_process(image_url)
        markdown = ocr_response.get("pages", [{}])[0].get("markdown", "")
        # Use structured extraction via chain
        return extract_problem_info(markdown)

    try:
        results = await asyncio.gather(*(process_image(img) for img in body.imageDataList))
        final_strings = []
        for res in results:
            parts = [
                res.get("question", ""),
                res.get("code_syntax", ""),
                f"Example Test Cases: {res.get('example_test_cases', '')}" if res.get("example_test_cases", "") else "",
                f"Constraints: {res.get('constraints', '')}" if res.get("constraints", "") else ""
            ]
            final_strings.append("\n".join(part for part in parts if part))
        problem_info = "\n\n".join(final_strings)
        logging.info(f"Final problem info: {problem_info}")
        return JSONResponse(content={"problemInfo": problem_info})
    except Exception as e:
        logging.error(f"Error in /api/extract: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate(body: GenerateRequest, request: Request):
    logging.info(f"Received /api/generate request: {body}")
    if not body.problemInfo:
        logging.error("Missing problemInfo")
        raise HTTPException(status_code=400, detail="problemInfo is required.")

    try:
        gen_prompt = (
            f"Generate a complete solution in {body.language}.\n"
            "Include a brief explanation and time-space complexity analysis.\n"
            f"Problem Information:\n{body.problemInfo}"
        )
        groq_response = await groq_client.generate_code(gen_prompt)
        generated_code = groq_response.get("code", "")
        logging.info(f"Generated code: {generated_code}")
        return JSONResponse(content={"code": generated_code})
    except Exception as e:
        logging.error(f"Error in /api/generate: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/debug")
async def debug(body: DebugRequest, request: Request):
    logging.info(f"Received /api/debug request: {body}")
    if not body.code:
        logging.error("Missing code for debugging")
        raise HTTPException(status_code=400, detail="code string is required.")

    try:
        debug_prompt = (
            "Analyze and debug the following code. Identify errors and suggest improvements.\n"
            "Provide a corrected version if necessary.\n"
            f"Code:\n{body.code}"
        )
        debug_response = await groq_client.debug_code(debug_prompt)
        debugged_output = debug_response.get("debuggedOutput", "")
        logging.info(f"Debugged output: {debugged_output}")
        return JSONResponse(content={"debuggedOutput": debugged_output})
    except Exception as e:
        logging.error(f"Error in /api/debug: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Server Startup ---
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG
    logging.info("Starting server on port 3000...")
    uvicorn.run(app, host="0.0.0.0", port=3000)
