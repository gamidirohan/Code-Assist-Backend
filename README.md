# Code Assist Backend

This is a Python-based backend that uses Mistral OCR to extract text from images. It communicates with a front-end through POST requests over HTTP.

## Project Structure

```
.env
backend.js
backend.py
eng.traineddata
leetcode_test.png
mistral_ocr.py
mistral_temp.py
package.json
__pycache__/
myenv/
```

## Endpoints

**POST /api/extract**  
- Input: JSON with "imageDataList" (array of base64-encoded images) and "language" (optional).  
- Processing: Performs OCR on the first base64 image and returns extracted text.  
- Response: JSON with the recognized text.

**POST /api/generate**  
- Input: JSON with "problemInfo" (text of the coding problem) and "language" (optional).  
- Processing: Generates a code solution, explanation, and complexity details.  
- Response: JSON with "code" as the generated solution.

## .env File
MISTRAL_API_KEY=<YOUR_MISTRAL_API_KEY>  
GROQ_API_KEY=<YOUR_GROQ_API_KEY>

## Setup & Usage

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd interview-coder_backend
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv myenv
    source myenv/Scripts/activate  # On Windows
    source myenv/bin/activate      # On Unix or MacOS
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Provide valid environment variables in `.env`:
    ```sh
    MISTRAL_API_KEY=your_api_key_here
    GROQ_API_KEY=your_groq_api_key_here
    ```

5. Run `python mistral_ocr.py` to start the service.

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

Replace `<repository-url>` with the URL of your GitHub repository.

## License

This project is licensed under the MIT License.
