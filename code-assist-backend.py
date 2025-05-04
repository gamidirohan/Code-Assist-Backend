import time
import os
import dotenv
import json
import uvicorn
import httpx
import base64
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from mistralai import Mistral, TextChunk
from mistralai.models.sdkerror import SDKError
# Import DeepSeek AI client
import requests

# Define the expected response structure for code generation
"""
Expected JSON structure for code generation:
{
    "Problem Information": "A summary of the problem",
    "Explanation": "BRUTE FORCE APPROACH:\n[explanation]\n\nBETTER APPROACH:\n[explanation]\n\nOPTIMAL APPROACH:\n[explanation]",
    "Code": "```[language]\n[code implementation]\n```",
    "Time Complexity": "Big O notation",
    "Space Complexity": "Big O notation",
    "complexity_explanation": "Detailed explanation of both time and space complexity"
}
"""

# Define the expected response structure for debug analysis
"""
Expected JSON structure for debug analysis:
{
    "code": "The original code that was analyzed",
    "debug_analysis": "A detailed analysis of the issues found and how to fix them",
    "thoughts": ["Step 1", "Step 2", "Step 3"],
    "time_complexity": "Big O notation",
    "space_complexity": "Big O notation",
    "time_complexity_explanation": "Explanation of time complexity",
    "space_complexity_explanation": "Explanation of space complexity"
}
"""

dotenv.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)

# Configure DeepSeek API
DEEPSEEK_API_KEY = "sk-f4c9e265c5054bc2b382f9141710583e"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# DeepSeek API headers
DEEPSEEK_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

# Function to call DeepSeek API directly
def call_deepseek_api(prompt, model="deepseek-coder", temperature=0.1, max_tokens=4096, response_format=None):
    """
    Call the DeepSeek API directly with the given parameters.

    Args:
        prompt (str): The prompt to send to the model
        model (str): The model to use (default: "deepseek-coder")
        temperature (float): The temperature for generation (default: 0.1)
        max_tokens (int): The maximum number of tokens to generate (default: 4096)
        response_format (dict): The response format (default: None)

    Returns:
        dict: The response from the DeepSeek API
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format:
        payload["response_format"] = response_format

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=DEEPSEEK_HEADERS, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        raise

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
                # Use Mistral's vision model directly to extract text from the image
                try:
                    print(f"Processing image {i+1} with Mistral vision model directly")

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
                        raise Exception("Vision model returned insufficient text")

                    print(f"Successfully processed image {i+1}")
                    all_ocr_results.append({
                        "index": i,
                        "text": image_ocr_md
                    })
                    # Successfully processed this image
                    image_processed = True

                except SDKError as vision_err:
                    if "Requests rate limit exceeded" in str(vision_err):
                        rate_limit_attempt += 1
                        wait_time = initial_wait_time * (2 ** (rate_limit_attempt - 1))  # Exponential backoff
                        print(f"Vision model rate limit exceeded for image {i+1}, attempt {rate_limit_attempt}/{max_rate_limit_retries}. Waiting {wait_time} seconds...")

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
                        raise vision_err

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
        f"Provide a solution in {language} with the following structure:\n"
        f"1. Brief analysis of approaches (brute force, better, optimal)\n"
        f"2. Implement the optimal solution with clean code\n"
        f"3. Analyze time and space complexity\n\n"
        f"IMPORTANT: Be concise to avoid hitting token limits. Format your response as a valid JSON object with EXACTLY these keys (case-sensitive):\n"
        f"- 'Problem Information': Brief summary of the problem (1-2 sentences)\n"
        f"- 'Explanation': Three sections labeled and separated by newlines:\n"
        f"  BRUTE FORCE APPROACH:\n[brief explanation]\n\nBETTER APPROACH:\n[brief explanation]\n\nOPTIMAL APPROACH:\n[brief explanation]\n"
        f"- 'Code': Your implementation in {language} wrapped in triple backticks\n"
        f"- 'Time Complexity': Big O notation only (e.g., 'O(n log n)')\n"
        f"- 'Space Complexity': Big O notation only (e.g., 'O(n)')\n"
        f"- 'complexity_explanation': Brief explanation of complexity\n\n"
        f"Keep all explanations concise to ensure the response fits within token limits. The frontend expects this specific JSON format."
    )

    # Maximum retries for code generation
    max_gen_retries = 3
    gen_attempt = 0

    # Check if the input is a nested JSON string that needs to be parsed
    problem_info = body.get("problemInfo", "")
    if isinstance(problem_info, str) and problem_info.startswith("{") and problem_info.endswith("}"):
        try:
            # Try to parse the nested JSON
            parsed_problem = json.loads(problem_info)
            if isinstance(parsed_problem, dict) and "problemInfo" in parsed_problem:
                # Extract the actual problem info from the nested JSON
                problem_info = parsed_problem.get("problemInfo", "")
                print(f"Extracted nested problemInfo: {problem_info[:100]}...")
                # Update the body with the extracted problem info
                body["problemInfo"] = problem_info
                # Update the prompt with the extracted problem info
                prompt_text = (
                    f"You are an expert {language} developer tasked with solving a coding problem. \n\n"
                    f"PROBLEM DESCRIPTION:\n{problem_info}\n\n"
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
        except json.JSONDecodeError:
            # If parsing fails, use the original problem_info
            pass

    # Try to use DeepSeek API directly first
    try:
        print("Using DeepSeek API directly for code generation")

        # Call DeepSeek API
        response = call_deepseek_api(
            prompt=prompt_text,
            model="deepseek-coder",
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )

        # Extract the content from the response
        if "choices" in response and len(response["choices"]) > 0:
            generated_content = response["choices"][0]["message"]["content"]
            print("DeepSeek generation succeeded")

            # Try to parse as JSON to validate
            try:
                # Validate that it's valid JSON
                parsed = json.loads(generated_content)

                # Print the full generated content for debugging
                print("\n\nFULL GENERATED JSON OUTPUT:")
                print(generated_content[:500] + "..." if len(generated_content) > 500 else generated_content)
                print("\n\n")

                # Also print the Explanation field specifically if it exists
                if "Explanation" in parsed:
                    print("EXPLANATION FIELD:")
                    explanation = parsed.get("Explanation", "")
                    print(repr(explanation[:500] + "..." if len(explanation) > 500 else explanation))  # Use repr to show escape sequences
                    print("\n\n")

                return {"code": generated_content}
            except json.JSONDecodeError as json_err:
                print(f"JSON validation failed: {str(json_err)}")
                print("Falling back to standard generation")
                # Fall back to standard generation
        else:
            print("DeepSeek API returned an unexpected response format")
            print("Falling back to standard generation")
    except Exception as e:
        print(f"DeepSeek direct generation failed: {str(e)}")
        print("Falling back to standard generation")
        # Fall back to standard generation

    # Standard generation approach using DeepSeek API directly
    while gen_attempt < max_gen_retries:
        try:
            print(f"Calling DeepSeek API directly (attempt {gen_attempt+1}/{max_gen_retries})")

            # Call DeepSeek API directly
            response = call_deepseek_api(
                prompt=prompt_text,
                model="deepseek-coder",
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )

            # Extract the content from the response
            if "choices" in response and len(response["choices"]) > 0:
                generated_content = response["choices"][0]["message"]["content"]
                print(f"Generation attempt {gen_attempt+1} succeeded")
                print(f"Raw response content: {generated_content[:200]}...")

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
                                "Problem Information": "Error in JSON generation",
                                "Explanation": "BRUTE FORCE APPROACH:\nThe model failed to generate valid JSON.\n\nBETTER APPROACH:\nHere is the raw output.\n\nOPTIMAL APPROACH:\nPlease try again with a simpler problem.",
                                "Code": f"```{language}\n{generated_content}\n```",
                                "Time Complexity": "N/A",
                                "Space Complexity": "N/A",
                                "complexity_explanation": "N/A"
                            }
                            return {"code": json.dumps(fallback_json, indent=2)}

                    # Try again with a more explicit prompt
                    prompt_text += "\n\nYour previous response was not valid JSON. Please ensure you return ONLY a valid JSON object with no additional text."
                    gen_attempt += 1
                    continue
            else:
                print(f"DeepSeek API returned an unexpected response format: {response}")
                gen_attempt += 1
                continue

        except Exception as e:
            print(f"Error generating code (attempt {gen_attempt+1}/{max_gen_retries}): {str(e)}")
            gen_attempt += 1

            # If we've exhausted all options, return an error
            if gen_attempt == max_gen_retries:
                # Create a simple error JSON
                error_json = {
                    "Problem Information": "Error in code generation",
                    "Explanation": f"BRUTE FORCE APPROACH:\nAn error occurred: {str(e)}\n\nBETTER APPROACH:\nPlease try again.\n\nOPTIMAL APPROACH:\nConsider simplifying the problem.",
                    "Code": f"```{language}\n// Error in code generation\n```",
                    "Time Complexity": "N/A",
                    "Space Complexity": "N/A",
                    "complexity_explanation": "N/A"
                }
                return {"code": json.dumps(error_json, indent=2)}

            # Wait before retrying
            wait_time = 2 * (2 ** (gen_attempt - 1))  # Exponential backoff
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            continue

    # This should never be reached due to the error handling above
    raise HTTPException(status_code=500, detail="Failed to generate code after maximum retries")


@app.post("/api/debug")
async def debug_route(request: Request):
    """
    Expects a JSON body with:
      "imageDataList": Array of base64-encoded images (original + debugging screenshots)
      "problemInfo": The original problem information
      "language": The desired solution language (defaults to Python)
    Returns a detailed debug analysis of the code.
    """
    print("Incoming debug request:", request)  # Print the request object for debugging

    body = await request.json()
    print("Full debug request body:", body)  # Print the parsed JSON body

    # Extract request parameters
    imageDataList = body.get("imageDataList", [])
    problemInfo = body.get("problemInfo", "")
    language = body.get("language", "python")

    # Check if problemInfo is a nested JSON string that needs to be parsed
    if isinstance(problemInfo, str) and problemInfo.startswith("{") and problemInfo.endswith("}"):
        try:
            # Try to parse the nested JSON
            parsed_problem = json.loads(problemInfo)
            if isinstance(parsed_problem, dict) and "problemInfo" in parsed_problem:
                # Extract the actual problem info from the nested JSON
                problemInfo = parsed_problem.get("problemInfo", "")
                print(f"Extracted nested problemInfo: {problemInfo[:100]}...")
        except json.JSONDecodeError:
            # If parsing fails, use the original problemInfo
            pass

    # Validate request parameters
    if not imageDataList:
        raise HTTPException(status_code=400, detail="imageDataList cannot be empty.")

    if not problemInfo:
        print("Warning: problemInfo is empty")

    # Process parameters for OCR extraction
    max_retries = 5
    backoff_time = 2
    all_ocr_results = []

    # Process each image in the list to extract code and error messages
    for i, base64_data in enumerate(imageDataList):
        print(f"Processing debug image {i+1} of {len(imageDataList)}")

        # For rate limit errors, we'll be more patient and retry with longer waits
        max_rate_limit_retries = 10  # More retries for rate limits
        initial_wait_time = 1  # Start with 1 second wait

        # Process this image with persistent retries
        image_processed = False
        rate_limit_attempt = 0
        general_attempt = 0

        while not image_processed:
            try:
                # Use Mistral's vision model directly to extract text from the image
                try:
                    print(f"Processing debug image {i+1} with Mistral vision model directly")

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
                                        "text": "This is a screenshot of code with possible error messages or test cases. Extract ALL text from this image with perfect accuracy, preserving the exact formatting. Pay special attention to:\n\n1. The code implementation\n2. Any error messages or stack traces\n3. Test cases and their inputs/outputs\n4. Any failure messages or debugging information\n5. Any comments or annotations\n\nFormat your response as plain text without any additional commentary. Include EVERY number, symbol, and special character exactly as shown. Do not summarize or paraphrase anything - extract the EXACT text as it appears."
                                    }
                                ]
                            }
                        ],
                        temperature=0
                    )

                    # Extract the text from the vision model response
                    extracted_text = vision_response.choices[0].message.content

                    if extracted_text and len(extracted_text) > 20:
                        print(f"Successfully extracted text using vision model for debug image {i+1}")
                        image_ocr_md = extracted_text
                    else:
                        print(f"Vision model returned insufficient text for debug image {i+1}")
                        raise Exception("Vision model returned insufficient text")

                    print(f"Successfully processed debug image {i+1}")
                    all_ocr_results.append({
                        "index": i,
                        "text": image_ocr_md
                    })
                    # Successfully processed this image
                    image_processed = True

                except SDKError as vision_err:
                    if "Requests rate limit exceeded" in str(vision_err):
                        rate_limit_attempt += 1
                        wait_time = initial_wait_time * (2 ** (rate_limit_attempt - 1))  # Exponential backoff
                        print(f"Vision model rate limit exceeded for debug image {i+1}, attempt {rate_limit_attempt}/{max_rate_limit_retries}. Waiting {wait_time} seconds...")

                        if rate_limit_attempt >= max_rate_limit_retries:
                            print(f"Failed to process debug image {i+1} after {max_rate_limit_retries} rate limit retries.")
                            # Even after many retries, we'll still try to continue with what we have
                            if i > 0 and all_ocr_results:
                                print(f"Moving on with {len(all_ocr_results)} successfully processed debug images")
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
                        raise vision_err

            except HTTPException:
                # Re-raise HTTP exceptions
                raise

            except Exception as e:
                general_attempt += 1
                print(f"Error processing debug image {i+1}: {str(e)}")

                if general_attempt >= max_retries:
                    print(f"Failed to process debug image {i+1} after {max_retries} retries due to error: {str(e)}")
                    # After max general retries, continue with what we have if possible
                    if i > 0 and all_ocr_results:
                        print(f"Moving on with {len(all_ocr_results)} successfully processed debug images")
                        image_processed = True  # Mark as processed to exit the loop
                    else:
                        # If this is the first image, we need to succeed
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to process the first debug image after {max_retries} retries: {str(e)}"
                        )

                # Wait and retry for general errors
                wait_time = backoff_time * (2 ** (general_attempt - 1))
                time.sleep(wait_time)
                continue

    # If we couldn't process any images, return an error
    if not all_ocr_results:
        raise HTTPException(status_code=500, detail="Failed to process any debug images after maximum retries.")

    # Combine all OCR results, preserving order
    all_ocr_results.sort(key=lambda x: x["index"])
    combined_ocr_text = "\n\n---IMAGE BOUNDARY---\n\n".join([result["text"] for result in all_ocr_results])

    # Now process the combined OCR text for debugging
    print(f"Processing combined OCR text from {len(all_ocr_results)} debug images")

    # For chat completion, we'll also use a more patient retry approach
    max_chat_rate_limit_retries = 10
    initial_chat_wait_time = 1

    chat_attempt = 0
    chat_rate_limit_attempt = 0
    chat_completed = False

    # Prepare the prompt for debug analysis
    debug_prompt = (
        f"I need you to debug code from a coding problem. Here's the context:\n\n"
        f"ORIGINAL PROBLEM:\n{problemInfo}\n\n"
        f"CODE AND DEBUG INFORMATION:\n{combined_ocr_text}\n\n"
        f"Language: {language}\n\n"
        f"Your task is to analyze the code, identify issues, and provide a comprehensive debug analysis. "
        f"The code might have syntax errors, logical bugs, edge case failures, or performance issues. "
        f"There might be test cases showing expected vs. actual outputs, or error messages from failed executions.\n\n"
        f"IMPORTANT INSTRUCTIONS:\n"
        f"1. Carefully analyze the code and any error messages or test failures\n"
        f"2. Identify ALL bugs, issues, or inefficiencies in the code\n"
        f"3. Explain WHY each issue occurs and HOW to fix it\n"
        f"4. Consider edge cases that might not be handled correctly\n"
        f"5. Analyze time and space complexity of the current implementation\n"
        f"6. Suggest optimizations if the current solution doesn't meet the required complexity\n"
        f"7. If test cases are provided, explain why the code fails on specific inputs\n"
        f"8. Provide a step-by-step thought process for your debugging approach\n\n"
        f"Format your response as a valid JSON object with the following keys:\n"
        f"- 'code': The original code that was analyzed (extract it from the debug information)\n"
        f"- 'debug_analysis': A detailed analysis of the issues found and how to fix them\n"
        f"- 'thoughts': An array of strings representing your step-by-step thought process during debugging\n"
        f"- 'time_complexity': The Big O notation for the time complexity of the original code\n"
        f"- 'space_complexity': The Big O notation for the space complexity of the original code\n"
        f"- 'time_complexity_explanation': Detailed explanation of the time complexity analysis\n"
        f"- 'space_complexity_explanation': Detailed explanation of the space complexity analysis\n\n"
        f"Ensure your response is a valid JSON object with EXACTLY these keys. The frontend expects this specific format."
    )

    # Keep trying until we succeed or exhaust all retries
    while not chat_completed:
        try:
            # Try to use DeepSeek API directly for debug analysis
            try:
                print("Using DeepSeek API directly for debug analysis")

                # Call DeepSeek API
                response = call_deepseek_api(
                    prompt=debug_prompt,
                    model="deepseek-coder",
                    temperature=0.1,
                    max_tokens=4096,
                    response_format={"type": "json_object"}
                )

                # Extract the content from the response
                if "choices" in response and len(response["choices"]) > 0:
                    response_content = response["choices"][0]["message"]["content"]
                    print(f"DeepSeek debug analysis succeeded")
                    print(f"Raw debug response content: {response_content[:200]}...")

                    # Try to parse the JSON response
                    try:
                        response_dict = json.loads(response_content)

                        # Validate that the response has the required keys
                        required_keys = ["code", "debug_analysis", "thoughts", "time_complexity",
                                        "space_complexity", "time_complexity_explanation",
                                        "space_complexity_explanation"]

                        missing_keys = [key for key in required_keys if key not in response_dict]

                        if missing_keys:
                            print(f"Missing required keys in debug response: {missing_keys}")
                            # Try to create a valid response with default values for missing keys
                            for key in missing_keys:
                                if key == "thoughts":
                                    response_dict[key] = ["Could not generate detailed thoughts"]
                                elif key == "code":
                                    # Try to extract code from the OCR text
                                    code_blocks = []
                                    for result in all_ocr_results:
                                        text = result["text"]
                                        if "```" in text:
                                            # Extract code blocks
                                            code_start = text.find("```")
                                            code_end = text.rfind("```")
                                            if code_start != -1 and code_end != -1 and code_end > code_start + 3:
                                                code_blocks.append(text[code_start:code_end+3])

                                    if code_blocks:
                                        response_dict[key] = "\n".join(code_blocks)
                                    else:
                                        response_dict[key] = "Could not extract code from the debug information"
                                else:
                                    response_dict[key] = "Not available"

                        # If we get here, the chat completed successfully
                        chat_completed = True
                        return response_dict

                    except json.JSONDecodeError as json_err:
                        print(f"JSON decode error in DeepSeek debug response: {json_err}")
                        print(f"Raw debug content: {response_content[:200]}...")
                        # Fall back to Mistral
                else:
                    print(f"DeepSeek API returned an unexpected response format for debug analysis")
                    # Fall back to Mistral
            except Exception as deepseek_err:
                print(f"DeepSeek debug analysis failed: {str(deepseek_err)}")
                print("Falling back to Mistral for debug analysis")

            # Standard approach using Mistral as fallback
            chat_response = client.chat.complete(
                model="pixtral-12b-latest",
                messages=[
                    {
                        "role": "user",
                        "content": debug_prompt
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
                print(f"Raw debug response content: {response_content[:200]}...")

                # Try to parse the JSON response
                try:
                    response_dict = json.loads(response_content)

                    # Validate that the response has the required keys
                    required_keys = ["code", "debug_analysis", "thoughts", "time_complexity",
                                    "space_complexity", "time_complexity_explanation",
                                    "space_complexity_explanation"]

                    missing_keys = [key for key in required_keys if key not in response_dict]

                    if missing_keys:
                        print(f"Missing required keys in debug response: {missing_keys}")
                        # Try to create a valid response with default values for missing keys
                        for key in missing_keys:
                            if key == "thoughts":
                                response_dict[key] = ["Could not generate detailed thoughts"]
                            elif key == "code":
                                # Try to extract code from the OCR text
                                code_blocks = []
                                for result in all_ocr_results:
                                    text = result["text"]
                                    if "```" in text:
                                        # Extract code blocks
                                        code_start = text.find("```")
                                        code_end = text.rfind("```")
                                        if code_start != -1 and code_end != -1 and code_end > code_start + 3:
                                            code_blocks.append(text[code_start:code_end+3])

                                if code_blocks:
                                    response_dict[key] = "\n".join(code_blocks)
                                else:
                                    response_dict[key] = "Could not extract code from the debug information"
                            else:
                                response_dict[key] = "Not available"

                    return response_dict

                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error in debug response: {json_err}")
                    print(f"Raw debug content: {response_content[:200]}...")

                    # If JSON parsing fails but we have content, try to create a structured response
                    fallback_response = {
                        "code": "Could not extract code from the debug information",
                        "debug_analysis": response_content,
                        "thoughts": ["Could not generate structured thoughts due to JSON parsing error"],
                        "time_complexity": "Unknown",
                        "space_complexity": "Unknown",
                        "time_complexity_explanation": "Could not analyze time complexity",
                        "space_complexity_explanation": "Could not analyze space complexity"
                    }

                    return fallback_response

            except Exception as e:
                print(f"Unexpected error processing debug response: {str(e)}")
                fallback_response = {
                    "code": "Could not extract code from the debug information",
                    "debug_analysis": f"Error processing debug response: {str(e)}",
                    "thoughts": ["An error occurred during debug analysis"],
                    "time_complexity": "Unknown",
                    "space_complexity": "Unknown",
                    "time_complexity_explanation": "Could not analyze time complexity",
                    "space_complexity_explanation": "Could not analyze space complexity"
                }

                return fallback_response

        except SDKError as chat_err:
            if "Requests rate limit exceeded" in str(chat_err):
                chat_rate_limit_attempt += 1
                wait_time = initial_chat_wait_time * (2 ** (chat_rate_limit_attempt - 1))  # Exponential backoff
                print(f"Chat rate limit exceeded during debug, attempt {chat_rate_limit_attempt}/{max_chat_rate_limit_retries}. Waiting {wait_time} seconds...")

                if chat_rate_limit_attempt >= max_chat_rate_limit_retries:
                    print(f"Failed to complete debug chat after {max_chat_rate_limit_retries} rate limit retries.")
                    # After exhausting all retries, return a fallback response
                    fallback_response = {
                        "code": "Could not extract code from the debug information",
                        "debug_analysis": f"API rate limited after {max_chat_rate_limit_retries} retries. Here's the raw OCR text:\n\n{combined_ocr_text[:1000]}...",
                        "thoughts": ["Rate limit exceeded during debug analysis"],
                        "time_complexity": "Unknown",
                        "space_complexity": "Unknown",
                        "time_complexity_explanation": "Could not analyze time complexity due to rate limiting",
                        "space_complexity_explanation": "Could not analyze space complexity due to rate limiting"
                    }

                    return fallback_response

                # Wait and retry
                time.sleep(wait_time)
                continue
            else:
                raise chat_err

        except Exception as e:
            chat_attempt += 1
            print(f"Error in debug chat completion: {str(e)}")

            if chat_attempt >= max_retries:
                print(f"Failed to complete debug chat after {max_retries} retries due to error: {str(e)}")
                # After exhausting all retries, return a fallback response
                fallback_response = {
                    "code": "Could not extract code from the debug information",
                    "debug_analysis": f"Error during debug analysis: {str(e)}. Here's the raw OCR text:\n\n{combined_ocr_text[:1000]}...",
                    "thoughts": ["An error occurred during debug analysis"],
                    "time_complexity": "Unknown",
                    "space_complexity": "Unknown",
                    "time_complexity_explanation": "Could not analyze time complexity due to an error",
                    "space_complexity_explanation": "Could not analyze space complexity due to an error"
                }

                return fallback_response

            # Wait and retry for general errors
            wait_time = backoff_time * (2 ** (chat_attempt - 1))
            time.sleep(wait_time)
            continue


import socket

if __name__ == "__main__":
    import uvicorn

    # Get the local IP address of the device
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"Backend is running on: http://{local_ip}:3000")

    uvicorn.run("code-assist-backend:app", host="0.0.0.0", port=3000)