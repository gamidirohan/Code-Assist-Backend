import time
import os
import dotenv
import json
import uvicorn
import httpx
import base64
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from mistralai import Mistral, ImageURLChunk, TextChunk
from mistralai.models.sdkerror import SDKError
from langchain_groq import ChatGroq

# Import Instructor (successor to Outlines) for structured generation
# Note: If instructor is not installed, we'll handle it gracefully
try:
    import instructor
    from pydantic import BaseModel, Field
    from typing import Dict, List, Optional

    # Define the Pydantic model for structured output
    class CodeSolution(BaseModel):
        """Model for code generation output"""
        # Using valid Python identifiers with aliases for the frontend
        problem_information: str = Field(alias="Problem Information", description="A summary of the problem")
        Explanation: str = Field(
            description="Detailed explanation with three sections clearly labeled and separated by newlines",
            json_schema_extra={
                "format": "BRUTE FORCE APPROACH:\n[detailed explanation]\n\nBETTER APPROACH:\n[detailed explanation]\n\nOPTIMAL APPROACH:\n[detailed explanation]"
            },
            examples=["BRUTE FORCE APPROACH:\nThis is the brute force approach.\n\nBETTER APPROACH:\nThis is a better approach.\n\nOPTIMAL APPROACH:\nThis is the optimal approach."]
        )
        Code: str = Field(description="Implementation in the specified language wrapped in triple backticks")
        time_complexity: str = Field(alias="Time Complexity", description="Big O analysis of the solution")
        space_complexity: str = Field(alias="Space Complexity", description="Big O analysis of the solution")
        complexity_explanation: str = Field(description="Detailed explanation of both time and space complexity")

        model_config = {
            # This ensures the field names are exactly as specified
            "populate_by_name": True,
            # Use aliases in the output JSON
            "json_schema_extra": {"by_alias": True}
        }

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    CodeSolution = None

dotenv.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

# Configure the LLM for code generation
groq_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",  # Using an available model on Groq
    temperature=0.1,  # Slight temperature for creativity while maintaining reliability
    max_tokens=4096,  # Ensure we have enough tokens for comprehensive solutions
    model_kwargs={
        "response_format": {"type": "json_object"},
        "top_p": 0.95  # High-quality sampling
    },
    groq_api_key=os.getenv("GROQ_API_KEY")
)

app = FastAPI()

@app.post("/api/extract")
async def extract_route(request: Request):
    """
    Receives multiple base64-encoded images (in imageDataList) and a language.
    Runs Mistral OCR on all images to extract text and returns a single coherent JSON object.
    Handles overlapping content between multiple screenshots of a long problem.
    """
    body = await request.json()
    imageDataList = body.get("imageDataList", [])
    language = body.get("language", "python")

    if not imageDataList:
        raise HTTPException(status_code=400, detail="imageDataList cannot be empty.")

    max_retries = 5
    backoff_time = 2

    # We'll collect OCR results from all images
    all_ocr_results = []

    # Process each image in the list
    for i, base64_data in enumerate(imageDataList):
        print(f"Processing image {i+1} of {len(imageDataList)}")

        # For rate limit errors, we'll be more patient and retry with longer waits
        max_rate_limit_retries = 10  # More retries for rate limits
        initial_wait_time = 1  # Start with 1 second wait

        # Process this image with persistent retries
        image_processed = False
        rate_limit_attempt = 0
        general_attempt = 0

        while not image_processed:
            try:
                # Run OCR on this image
                try:
                    # Step 1: Process the image with Mistral OCR
                    image_response = client.ocr.process(
                        document=ImageURLChunk(image_url=f"data:image/jpeg;base64,{base64_data}"),
                        model="mistral-ocr-latest"
                    )

                    # Get the OCR markdown result
                    image_ocr_md = image_response.pages[0].markdown
                    print(f"OCR markdown for image {i+1}: {image_ocr_md[:100]}...")

                    # Step 2: If the OCR result is just an image reference, use Mistral's vision model
                    if image_ocr_md.startswith("![") and image_ocr_md.endswith(")") and len(image_ocr_md.split()) <= 2:
                        print(f"Mistral OCR returned only an image reference for image {i+1}. Using Mistral vision model...")

                        try:
                            # Use Mistral's vision model to extract text from the image
                            vision_response = client.chat.complete(
                                model="pixtral-12b-latest",
                                messages=[
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{base64_data}"
                                                }
                                            },
                                            {
                                                "type": "text",
                                                "text": "This is a screenshot of a coding problem. Extract ALL text from this image with perfect accuracy, preserving the exact formatting. Pay special attention to:\n\n1. Problem title and difficulty level\n2. Complete problem description with ALL details\n3. ALL examples with their inputs, outputs, and explanations\n4. ALL constraints and edge cases (e.g., array size limits, value ranges)\n5. Time and space complexity requirements\n6. Any code templates or function signatures\n\nFormat your response as plain text without any additional commentary. Include EVERY number, symbol, and special character exactly as shown. Do not summarize or paraphrase anything - extract the EXACT text as it appears."
                                            }
                                        ]
                                    }
                                ],
                                temperature=0
                            )

                            # Extract the text from the vision model response
                            extracted_text = vision_response.choices[0].message.content

                            if extracted_text and len(extracted_text) > 20:
                                print(f"Successfully extracted text using vision model for image {i+1}")
                                image_ocr_md = extracted_text
                            else:
                                print(f"Vision model returned insufficient text for image {i+1}")
                        except Exception as vision_err:
                            print(f"Error using vision model for image {i+1}: {str(vision_err)}")

                    print(f"Successfully processed image {i+1}")
                    all_ocr_results.append({
                        "index": i,
                        "text": image_ocr_md
                    })
                    # Successfully processed this image
                    image_processed = True

                except SDKError as ocr_err:
                    if "Requests rate limit exceeded" in str(ocr_err):
                        rate_limit_attempt += 1
                        wait_time = initial_wait_time * (2 ** (rate_limit_attempt - 1))  # Exponential backoff
                        print(f"OCR rate limit exceeded for image {i+1}, attempt {rate_limit_attempt}/{max_rate_limit_retries}. Waiting {wait_time} seconds...")

                        if rate_limit_attempt >= max_rate_limit_retries:
                            print(f"Failed to process image {i+1} after {max_rate_limit_retries} rate limit retries.")
                            # Even after many retries, we'll still try to continue with what we have
                            if i > 0 and all_ocr_results:
                                print(f"Moving on with {len(all_ocr_results)} successfully processed images")
                                image_processed = True  # Mark as processed to exit the loop
                            else:
                                # If this is the first image, we need to succeed
                                raise HTTPException(
                                    status_code=429,
                                    detail=f"Rate limit exceeded after {max_rate_limit_retries} retries. Please try again later."
                                )

                        # Wait and retry
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ocr_err

            except HTTPException:
                # Re-raise HTTP exceptions
                raise

            except Exception as e:
                general_attempt += 1
                print(f"Error processing image {i+1}: {str(e)}")

                if general_attempt >= max_retries:
                    print(f"Failed to process image {i+1} after {max_retries} retries due to error: {str(e)}")
                    # After max general retries, continue with what we have if possible
                    if i > 0 and all_ocr_results:
                        print(f"Moving on with {len(all_ocr_results)} successfully processed images")
                        image_processed = True  # Mark as processed to exit the loop
                    else:
                        # If this is the first image, we need to succeed
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to process the first image after {max_retries} retries: {str(e)}"
                        )

                # Wait and retry for general errors
                wait_time = backoff_time * (2 ** (general_attempt - 1))
                time.sleep(wait_time)
                continue

    # If we couldn't process any images, return an error
    if not all_ocr_results:
        raise HTTPException(status_code=500, detail="Failed to process any images after maximum retries.")

    # Combine all OCR results, preserving order
    all_ocr_results.sort(key=lambda x: x["index"])
    combined_ocr_text = "\n\n---IMAGE BOUNDARY---\n\n".join([result["text"] for result in all_ocr_results])

    # Now process the combined OCR text
    print(f"Processing combined OCR text from {len(all_ocr_results)} images")

    # For chat completion, we'll also use a more patient retry approach
    max_chat_rate_limit_retries = 10
    initial_chat_wait_time = 1

    chat_attempt = 0
    chat_rate_limit_attempt = 0
    chat_completed = False

    # Prepare the prompt for chat completion
    prompt_text = (
        f"I have OCR text from multiple screenshots of a coding problem:\n<BEGIN_COMBINED_OCR>\n{combined_ocr_text}\n<END_COMBINED_OCR>\n"
        f"Language expected: {language}\n\n"
        "These screenshots are multiple captures of a long problem and likely contain overlapping content. "
        "Your task is to create a COMPLETE and EXACT reproduction of the problem statement by carefully analyzing all the text.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "1. Extract EVERY detail from the screenshots, including title, difficulty level, full description, ALL examples, ALL constraints, and ANY follow-up questions\n"
        "2. Preserve the EXACT formatting, including newlines, indentation, and code blocks\n"
        "3. Include ALL code snippets exactly as shown, including comments and class definitions\n"
        "4. Pay special attention to numerical details like constraints (e.g., array size limits, value ranges)\n"
        "5. Merge overlapping content carefully to avoid duplication\n"
        "6. Ensure ALL examples are included with their exact inputs, outputs, and explanations\n"
        "7. Include ALL constraints, even if they seem redundant or obvious\n"
        "8. Preserve ALL time and space complexity requirements exactly as stated\n"
        "9. If there are multiple approaches mentioned or hinted at, include ALL of them\n"
        "10. If edge cases are mentioned, make sure they are ALL included\n\n"
        "CRITICAL: Do not miss ANY information from the original problem. The extracted problem statement must be COMPLETE and EXACT, with no information lost or altered. This is extremely important for the user to solve the problem correctly.\n\n"
        "Return a valid, properly formatted JSON object with exactly two root keys: 'problemInfo' and 'language'. "
        "The 'problemInfo' key should contain a detailed string with the COMPLETE problem information EXACTLY as given in the images. "
        "The 'language' key should contain a string with the programming language. "
        "Do not include nested JSON objects or arrays. Ensure all quotes are properly escaped. "
        "Return only the JSON object with no additional text."
    )

    # Keep trying until we succeed or exhaust all retries
    while not chat_completed:
        try:
            # Since we're not using images directly anymore, we'll just use text
            chat_response = client.chat.complete(
                model="pixtral-12b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            # If we get here, the chat completed successfully
            chat_completed = True

            # Process the response
            try:
                response_content = chat_response.choices[0].message.content
                print(f"Raw response content: {response_content[:200]}...")

                # Try to parse the JSON response
                try:
                    response_dict = json.loads(response_content)

                    # Validate that the response has the required keys
                    if "problemInfo" not in response_dict or "language" not in response_dict:
                        print("Missing required keys in response")
                        # Try to extract problem info from the raw content if JSON is malformed
                        if "problemInfo" not in response_dict and len(response_content) > 20:
                            return {
                                "problemInfo": response_content,  # Use the raw content as problemInfo
                                "language": language
                            }
                        else:
                            return {
                                "problemInfo": f"Error: Response missing required keys. Using combined OCR text: {combined_ocr_text[:1000]}...",
                                "language": language
                            }

                    # Check if problemInfo is too short (likely incomplete)
                    if len(response_dict["problemInfo"]) < 100 and len(combined_ocr_text) > 200:
                        print("Problem info seems too short, using combined OCR text instead")
                        return {
                            "problemInfo": combined_ocr_text,
                            "language": response_dict.get("language", language)
                        }

                    return response_dict

                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error: {json_err}")
                    print(f"Raw content: {response_content[:200]}...")

                    # If JSON parsing fails but we have content, return it directly
                    if len(response_content) > 20:
                        return {
                            "problemInfo": response_content,  # Use the raw content as problemInfo
                            "language": language
                        }
                    else:
                        return {
                            "problemInfo": f"Error parsing response. Using combined OCR text: {combined_ocr_text[:1000]}...",
                            "language": language
                        }
            except Exception as e:
                print(f"Unexpected error processing chat response: {str(e)}")
                return {
                    "problemInfo": f"Error processing response: {str(e)}. Using combined OCR text: {combined_ocr_text[:1000]}...",
                    "language": language
                }

        except SDKError as chat_err:
            if "Requests rate limit exceeded" in str(chat_err):
                chat_rate_limit_attempt += 1
                wait_time = initial_chat_wait_time * (2 ** (chat_rate_limit_attempt - 1))  # Exponential backoff
                print(f"Chat rate limit exceeded, attempt {chat_rate_limit_attempt}/{max_chat_rate_limit_retries}. Waiting {wait_time} seconds...")

                if chat_rate_limit_attempt >= max_chat_rate_limit_retries:
                    print(f"Failed to complete chat after {max_chat_rate_limit_retries} rate limit retries.")
                    # After exhausting all retries, return the raw OCR text
                    return {
                        "problemInfo": f"API rate limited after {max_chat_rate_limit_retries} retries. Here's the raw OCR text:\n\n{combined_ocr_text}",
                        "language": language
                    }

                # Wait and retry
                time.sleep(wait_time)
                continue
            else:
                raise chat_err

        except Exception as e:
            chat_attempt += 1
            print(f"Error in chat completion: {str(e)}")

            if chat_attempt >= max_retries:
                print(f"Failed to complete chat after {max_retries} retries due to error: {str(e)}")
                # After exhausting all retries, return the raw OCR text
                return {
                    "problemInfo": f"Error processing chat: {str(e)}. Here's the raw OCR text:\n\n{combined_ocr_text}",
                    "language": language
                }

            # Wait and retry for general errors
            wait_time = backoff_time * (2 ** (chat_attempt - 1))
            time.sleep(wait_time)
            continue



        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
            chat_attempt += 1
            print(f"Network error in chat completion: {str(e)}")

            if chat_attempt >= max_retries:
                print(f"Failed to complete chat after {max_retries} network error retries")
                # After exhausting all retries, return the raw OCR text
                return {
                    "problemInfo": f"Network error: {str(e)}. Here's the raw OCR text:\n\n{combined_ocr_text}",
                    "language": language
                }

            # Wait and retry for network errors
            wait_time = backoff_time * (2 ** (chat_attempt - 1))
            time.sleep(wait_time)
            continue

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

    # Create a structured prompt that matches the expected JSON format for the frontend
    prompt_text = (
        f"You are an expert {language} developer tasked with solving a coding problem. \n\n"
        f"PROBLEM DESCRIPTION:\n{body['problemInfo']}\n\n"
        f"Please provide a comprehensive solution in {language} with the following structure:\n"
        f"1. Analyze the problem and explain different approaches (brute force, better, and optimal)\n"
        f"2. Implement the optimal solution with clean, well-commented code\n"
        f"3. Analyze the time and space complexity\n\n"
        f"IMPORTANT GUIDELINES:\n"
        f"- Pay careful attention to ALL constraints mentioned in the problem\n"
        f"- Ensure your solution handles ALL edge cases\n"
        f"- If the problem specifies time or space complexity requirements, your solution MUST meet these requirements\n"
        f"- The OPTIMAL solution should be the most efficient possible given the constraints\n"
        f"- If multiple optimal approaches exist, choose the one that is most readable and maintainable\n"
        f"- Ensure your code is correct and would pass all test cases\n\n"
        f"Format your response as a valid JSON object with EXACTLY these keys (case-sensitive):\n"
        f"- 'Problem Information': A summary of the problem\n"
        f"- 'Explanation': A detailed explanation with three sections clearly labeled and separated by newlines:\n"
        f"  BRUTE FORCE APPROACH:\n[detailed explanation]\n\nBETTER APPROACH:\n[detailed explanation]\n\nOPTIMAL APPROACH:\n[detailed explanation]\n"
        f"- 'Code': Your implementation in {language} wrapped in triple backticks\n"
        f"- 'Time Complexity': Big O analysis of your solution\n"
        f"- 'Space Complexity': Big O analysis of your solution\n"
        f"- 'complexity_explanation': Detailed explanation of both time and space complexity\n\n"
        f"IMPORTANT: Ensure your response is a valid JSON object with EXACTLY these keys. The frontend expects this specific format."
    )

    # Maximum retries for code generation
    max_gen_retries = 3
    gen_attempt = 0

    # Try to use Instructor for structured generation if available
    if INSTRUCTOR_AVAILABLE:
        try:
            print("Using Instructor for structured generation")
            # Create a patched client using Instructor with Groq
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            patched_client = instructor.patch(groq_client)

            # Generate structured output directly
            result = patched_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                response_model=CodeSolution,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            print("Instructor generation succeeded")

            # Convert to JSON string with aliases
            generated_content = json.dumps(result.model_dump(by_alias=True), indent=2)

            # Print the full generated content for debugging
            print("\n\nFULL GENERATED JSON OUTPUT:")
            print(generated_content)
            print("\n\n")

            # Also print the Explanation field specifically
            parsed = json.loads(generated_content)
            print("EXPLANATION FIELD:")
            print(repr(parsed.get("Explanation", "")))  # Use repr to show escape sequences
            print("\n\n")

            return {"code": generated_content}
        except Exception as instructor_err:
            print(f"Instructor generation failed: {str(instructor_err)}")
            print("Falling back to standard generation")
            # Fall back to standard generation

    # Standard generation approach
    while gen_attempt < max_gen_retries:
        try:
            # Call the ChatGroq LLM
            groq_response = groq_llm.invoke(prompt_text)
            print(f"Generation attempt {gen_attempt+1} succeeded")

            # Extract content
            generated_content = groq_response.content

            # Try to parse as JSON to validate
            try:
                # Just validate that it's valid JSON, we don't need to use the parsed content
                json.loads(generated_content)
                # If we get here, it's valid JSON
                return {"code": generated_content}

            except json.JSONDecodeError as json_err:
                print(f"JSON validation failed: {str(json_err)}")

                # If this is our last retry, try to fix the JSON or return what we have
                if gen_attempt == max_gen_retries - 1:
                    # Try to extract useful content even if JSON is invalid
                    if "Problem Information" in generated_content and "Code" in generated_content:
                        print("Returning non-JSON content as it contains useful information")
                        return {"code": generated_content}
                    else:
                        # Create a simple JSON structure with the error and content
                        fallback_json = {
                            "Problem Information": {"title": "Error in JSON generation", "description": body['problemInfo']},
                            "Explanation": "The model failed to generate valid JSON. Here is the raw output:",
                            "Code": generated_content,
                            "Time Complexity": "N/A",
                            "Space Complexity": "N/A",
                            "complexity_explanation": "N/A"
                        }
                        return {"code": json.dumps(fallback_json, indent=2)}

                # Try again with a more explicit prompt
                prompt_text += "\n\nYour previous response was not valid JSON. Please ensure you return ONLY a valid JSON object with no additional text."
                gen_attempt += 1
                continue

        except Exception as e:
            print(f"Error generating code (attempt {gen_attempt+1}/{max_gen_retries}): {str(e)}")
            gen_attempt += 1

            # If this is our last retry with the primary model, try a fallback model
            if gen_attempt == max_gen_retries - 1:
                print("Trying fallback model: deepseek-r1-distill-llama-70b")

                # First try with Instructor if available
                if INSTRUCTOR_AVAILABLE:
                    try:
                        print("Using Instructor with fallback model")
                        # Create a patched client using Instructor with Groq
                        from groq import Groq
                        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                        patched_client = instructor.patch(groq_client)

                        # Generate structured output directly with fallback model
                        result = patched_client.chat.completions.create(
                            model="deepseek-r1-distill-llama-70b",
                            response_model=CodeSolution,
                            messages=[
                                {"role": "user", "content": prompt_text}
                            ],
                            temperature=0.1,
                            max_tokens=4096
                        )
                        print("Instructor with fallback model succeeded")

                        # Convert to JSON string with aliases
                        fallback_content = json.dumps(result.model_dump(by_alias=True), indent=2)

                        # Print the full generated content for debugging
                        print("\n\nFULL GENERATED JSON OUTPUT (FALLBACK MODEL):")
                        print(fallback_content)
                        print("\n\n")

                        # Also print the Explanation field specifically
                        parsed = json.loads(fallback_content)
                        print("EXPLANATION FIELD (FALLBACK MODEL):")
                        print(repr(parsed.get("Explanation", "")))  # Use repr to show escape sequences
                        print("\n\n")

                        return {"code": fallback_content}
                    except Exception as instructor_fallback_err:
                        print(f"Instructor with fallback model failed: {str(instructor_fallback_err)}")
                        print("Falling back to standard fallback approach")

                # Standard fallback approach
                try:
                    # Create a fallback LLM with a different model
                    fallback_llm = ChatGroq(
                        model_name="deepseek-r1-distill-llama-70b",  # Fallback model
                        temperature=0.1,
                        max_tokens=4096,
                        model_kwargs={
                            "response_format": {"type": "json_object"},
                            "top_p": 0.95
                        },
                        groq_api_key=os.getenv("GROQ_API_KEY")
                    )

                    # Try with the fallback model
                    fallback_response = fallback_llm.invoke(prompt_text)
                    fallback_content = fallback_response.content

                    # Validate JSON
                    try:
                        json.loads(fallback_content)
                        print("Fallback model succeeded")
                        return {"code": fallback_content}
                    except json.JSONDecodeError:
                        print("Fallback model also produced invalid JSON")
                except Exception as fallback_err:
                    print(f"Fallback model error: {str(fallback_err)}")

            # If we've exhausted all options, return an error
            if gen_attempt == max_gen_retries:
                # Create a simple error JSON
                error_json = {
                    "Problem Information": {"title": "Error in code generation", "description": body['problemInfo']},
                    "Explanation": f"An error occurred: {str(e)}",
                    "Code": "// Error in code generation",
                    "Time Complexity": "N/A",
                    "Space Complexity": "N/A",
                    "complexity_explanation": "N/A"
                }
                return {"code": json.dumps(error_json, indent=2)}

            # Wait before retrying
            time.sleep(2)
            continue

    # This should never be reached due to the error handling above
    raise HTTPException(status_code=500, detail="Failed to generate code after maximum retries")


import socket

if __name__ == "__main__":
    import uvicorn

    # Get the local IP address of the device
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"Backend is running on: http://{local_ip}:3000")

    uvicorn.run("code-assist-backend:app", host="0.0.0.0", port=3000)