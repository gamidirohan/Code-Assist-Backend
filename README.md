# Code Assist Backend

This project is a backend service for extracting text from images using Mistral OCR and Flask.

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

## Setup

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

4. Create a `.env` file and add your Mistral API key:
    ```
    MISTRAL_API_KEY=your_api_key_here
    ```

## Running the Application

1. Start the Flask server:
    ```sh
    python mistral_ocr.py
    ```

2. The server will be running at `http://127.0.0.1:5000`.

## API Endpoints

### POST /api/extract

- **Description**: Extracts text from base64-encoded images.
- **Request Body**:
    ```json
    {
        "imageDataList": ["base64_image_data_1", "base64_image_data_2"],
        "language": "python"
    }
    ```
- **Response**:
    ```json
    [
        {
            "extracted_text": "..."
        }
    ]
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

Replace `<repository-url>` with the URL of your GitHub repository.

## License

This project is licensed under the MIT License.
