from mistralai import Mistral, ImageURLChunk, TextChunk
from pathlib import Path
import base64
import json
from pydantic import BaseModel
from typing import Optional, List

api_key = "in2lIL3vAHbGppv1zDg8Ps9hEQM28aat"
client = Mistral(api_key=api_key)

class CodingQuestionInfo(BaseModel):
    coding_question: Optional[str] = None
    basic_code: Optional[str] = None
    constraints: Optional[str] = None
    example_test_cases: Optional[str] = None

def extract_coding_question_info(image_path: str) -> CodingQuestionInfo:
    """
    Extracts coding question information from an image, focusing on relevant details.

    Args:
        image_path: Path to the image file.

    Returns:
        CodingQuestionInfo: A structured object containing extracted information.
    """
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"The provided image path '{image_path}' does not exist.")

    encoded_image = base64.b64encode(image_file.read_bytes()).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

    image_response = client.ocr.process(document=ImageURLChunk(image_url=base64_data_url), model="mistral-ocr-latest")
    image_ocr_markdown = image_response.pages[0].markdown

    chat_response = client.chat.completions.create(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(
                        text=(
                            "This is the image's OCR in markdown:\n"
                            f"<BEGIN_IMAGE_OCR>\n{image_ocr_markdown}\n<END_IMAGE_OCR>.\n"
                            "Extract the Coding question, basic code (if given), constraints, and Example test cases. "
                            "Return this information in a structured JSON format."
                            "If any field is not present, return null for that field."
                        )
                    ),
                ],
            },
        ],
        response_format={"type": "json_object", "schema": CodingQuestionInfo.schema_json()},
        temperature=0,
    )

    try:
        parsed_response = CodingQuestionInfo.model_validate_json(chat_response.choices[0].message.content)
        return parsed_response
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return CodingQuestionInfo()