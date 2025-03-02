import os, base64, json
from io import BytesIO

import boto3
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from PIL import Image
import torch


# Import olmOCR utilities and model classes
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

app = FastAPI(title="olmOCR Service", version="1.0")

# Load model and processor at startup
# Specify local paths for model weights and processor
MODEL_DIR = "/home/ubuntu/models/olmOCR-7B-0225-preview"
PROCESSOR_NAME = "/home/ubuntu/models/Qwen2VL-7B-Instruct"  # or "Qwen/Qwen2-VL-7B-Instruct" if internet is available for cache

print("Loading model... This may take a few minutes.")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_DIR, torch.float16)  # use BF16/FP16 if available
model.eval()
processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
device = "cuda"
model.to(device)
print("Model loaded on device:", device)

# Initialize AWS S3 client (assumes IAM role or AWS creds configured)
s3_client = boto3.client("s3")
TARGET_IMAGE_DIM=1024
# --- Core function to prepare a page query ---
def build_page_query(local_pdf_path: str, pretty_pdf_path: str, page: int) -> dict:
    """
    Processes one PDF page:
      - Renders the page to a base64-encoded PNG image.
      - Extracts anchor text from the page.
      - Prepares a payload similar to an OpenAI Chat Completion query.
    """
    # Render the page as a base64 PNG image
    image_base64 = render_pdf_to_base64png(local_pdf_path, page, TARGET_IMAGE_DIM)
    # Extract anchor text from the page
    anchor_text = get_anchor_text(local_pdf_path, page, pdf_engine="pdfreport")
    
    # Build a unique identifier (e.g., PDF filename and page number)
    custom_id = f"{os.path.splitext(pretty_pdf_path)[0]}-page{page}"
    
    # Construct the payload that you could later send to your LLM
    # Note: We're not actually calling the LLM here.
    payload = {
        "custom_id": custom_id,
        "anchor_text": anchor_text,
        "image_data": f"data:image/png;base64,{image_base64}",
    }
    return payload

# --- FastAPI Endpoint ---
@app.post("/prepare")
async def prepare_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, processes each page to extract the base64 image and anchor text,
    and returns a JSON object with prepared query payloads for each page.
    """
    # Save the uploaded file to a temporary location
    contents = await file.read()
    local_pdf_path = f"/tmp/{file.filename}"
    with open(local_pdf_path, "wb") as f:
        f.write(contents)
    
    # Use PyPDF2 to determine the number of pages
    reader = PdfReader(local_pdf_path)
    num_pages = len(reader.pages)
    
    # Build the prepared query for each page
    queries = []
    for page in range(1, num_pages + 1):
        query = build_page_query(local_pdf_path, file.filename, page)
        queries.append(query)
    
    # Return the list of prepared payloads (one per page)
    return JSONResponse(content={"queries": queries})


@app.post("/ocr")
async def ocr_file(file: UploadFile = File(...)):
    """OCR a PDF from an uploaded file. Returns JSON with extracted text per page."""
    # Save uploaded file to a temporary path
    contents = await file.read()
    pdf_path = f"/tmp/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(contents)
    result = process_pdf(pdf_path)
    return JSONResponse(content=result)

@app.post("/ocr/s3")
def ocr_s3(input: dict):
    """
    OCR a PDF file from S3. Expects JSON input: {"bucket": ..., "key": ..., "output_bucket": ..., "output_key": ...}.
    The output_bucket/key are optional; if provided, results will be uploaded to S3 as JSON.
    """
    bucket = input.get("bucket")
    key = input.get("key")
    output_bucket = input.get("output_bucket")
    output_key = input.get("output_key")
    if not bucket or not key:
        return JSONResponse(content={"error": "bucket and key required"}, status_code=400)
    # Download the PDF from S3 to a temp file
    local_pdf = f"/tmp/{os.path.basename(key)}"
    s3_client.download_file(bucket, key, local_pdf)
    result = process_pdf(local_pdf)
    # Optionally upload results JSON to S3
    if output_bucket and output_key:
        tmp_out = "/tmp/ocr_result.json"
        with open(tmp_out, "w") as f:
            json.dump(result, f)
        s3_client.upload_file(tmp_out, output_bucket, output_key)
    return JSONResponse(content=result)

def process_pdf(pdf_path: str) -> dict:
    """Run olmOCR model on the given PDF file path. Returns a dict with pages and extracted text."""
    # Determine number of pages in PDF
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pages_output = []
    for page_num in range(1, num_pages+1):
        # Render page to image (base64 PNG) with target size ~1024px on the longest side
        image_b64 = render_pdf_to_base64png(pdf_path, page_num, target_longest_image_dim=1024)
        # Extract anchor text (metadata from PDF text layer for prompt context)
        anchor_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdfreport", target_length=4000)
        prompt_text = build_finetuning_prompt(anchor_text)
        # Prepare the multimodal input message
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }
        # Format input for the model
        # Use the processor to apply the chat template and create model inputs
        text_input = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        inputs = processor(text=[text_input], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k,v in inputs.items()}
        # Generate output from the model
        output = model.generate(**inputs, temperature=0.1, max_new_tokens=512, num_return_sequences=1)
        # Decode generated tokens to text
        prompt_len = inputs["input_ids"].shape[1]
        gen_tokens = output[:, prompt_len:]
        page_text = processor.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        # The model outputs a JSON string (or plain text) with fields including "natural_text"
        # Try to parse it as JSON
        try:
            page_data = json.loads(page_text)
        except json.JSONDecodeError:
            # If not a valid JSON, just return the raw text under a generic field
            page_data = {"text": page_text}
        page_data["page_number"] = page_num
        pages_output.append(page_data)
    # Combine results
    result = {"pages": pages_output}
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)