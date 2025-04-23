# Code Assist Backend

This is a Python-based backend that uses Mistral OCR to extract text from coding problem images and generates structured solutions using LLMs. It communicates with a front-end through POST requests over HTTP.

## Features

- **Advanced OCR**: Uses Mistral OCR to extract complete problem details from images
- **Structured Code Generation**: Leverages Instructor and Groq API for reliable JSON-structured outputs
- **Multiple Model Support**: Falls back to alternative models if primary generation fails
- **Rate Limit Handling**: Implements exponential backoff for API rate limits
- **Robust Error Handling**: Gracefully handles errors at every step of the process

## Project Structure

```
.env
.gitignore
leetcode_test.png
code-assist-backend.py
README.md
requirements.txt
myenv/
```

## Endpoints

**POST /api/extract**
- Input: JSON with "imageDataList" (array of base64-encoded images) and "language" (optional).
- Processing: Performs OCR on all provided images, merges content, and extracts a complete problem statement.
- Response: JSON with the complete problem information and language.

**POST /api/generate**
- Input: JSON with "problemInfo" (text of the coding problem) and "language" (optional).
- Processing: Generates a structured solution with multiple approaches, code implementation, and complexity analysis.
- Response: JSON with the following structure:
  ```json
  {
    "Problem Information": "Summary of the problem",
    "Explanation": "Detailed explanation of brute force, better, and optimal approaches",
    "Code": "Implementation in the specified language",
    "Time Complexity": "Big O analysis of time complexity",
    "Space Complexity": "Big O analysis of space complexity",
    "complexity_explanation": "Detailed explanation of both time and space complexity"
  }
  ```

## .env File
MISTRAL_API_KEY=<YOUR_MISTRAL_API_KEY>
GROQ_API_KEY=<YOUR_GROQ_API_KEY>

## Setup & Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/gamidirohan/Code-Assist-Backend.git
    cd interview-coder_backend
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv myenv
    ./myenv/Scripts/activate  # On Windows
    source myenv/bin/activate      # On Unix or MacOS
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

   Key dependencies include:
   - `fastapi` and `uvicorn` for the web server
   - `mistralai` for OCR processing
   - `langchain-groq` and `groq` for LLM integration
   - `instructor` for structured JSON generation

4. Provide valid environment variables in `.env`:
    ```sh
    MISTRAL_API_KEY=your_api_key_here
    GROQ_API_KEY=your_groq_api_key_here
    ```

5. Run `python code-assist-backend.py` to start the service.

## Cloning from GitHub
Clone the repo directly:
```sh
git clone https://github.com/gamidirohan/Code-Assist-Backend.git
cd Code-Assist-Backend
```

## Pushing Changes to GitHub

1. Add changes to the staging area:
    ```sh
    git add .
    ```

2. Commit the changes:
    ```sh
    git commit -m "Your commit message"
    ```

3. Push the changes to GitHub:
    ```sh
    git push origin main
    ```
## Hosting
Host it on Render:
For the absolute easiest deployment with the least configuration:

Create an account on Render
Connect your GitHub repository
Select "Web Service"
Set build command: pip install -r requirements.txt
Set start command: python code-assist-backend.py

## Models Used

- **OCR Processing**: Mistral OCR (pixtral-12b-latest)
- **Primary Code Generation**: Llama 3.1 8B Instant (llama-3.1-8b-instant)
- **Fallback Code Generation**: DeepSeek R1 (deepseek-r1-distill-llama-70b)

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

This means:
- ✅ You can use this for personal, non-commercial purposes
- ✅ You can modify and share this code
- ✅ You must give appropriate credit to Rohan Gamidi
- ❌ You cannot use this commercially
- ❌ You cannot claim this as your own work

For more details, see the [LICENSE](LICENSE) file or visit the [Creative Commons website](https://creativecommons.org/licenses/by-nc/4.0/).
