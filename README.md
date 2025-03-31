# Code Assist Backend

This is a Python-based backend that uses Mistral OCR to extract text from images. It communicates with a front-end through POST requests over HTTP.

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

## License

This project is licensed under the MIT License.



pyinstaller
auto-py-to-exe
