from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, ViTModel, ViTImageProcessor
import torch
from PIL import Image
import io
import base64
from typing import Optional
import time

app = FastAPI()

# Load models at startup
try:
    # GPT-2 model
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # ViT model
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt_model.to(device)
    vit_model.to(device)
    
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

class TextRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100

class ImageRequest(BaseModel):
    image: str  # base64 encoded image

@app.post("/generate_text")
async def generate_text(request: TextRequest):
    start_time = time.time()
    try:
        inputs = gpt_tokenizer(request.text, return_tensors="pt").to(device)
        outputs = gpt_model.generate(
            **inputs,
            max_length=request.max_length,
            num_return_sequences=1,
            pad_token_id=gpt_tokenizer.eos_token_id
        )
        generated_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        process_time = time.time() - start_time
        return {
            "generated_text": generated_text,
            "processing_time": process_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_image")
async def process_image(request: ImageRequest):
    start_time = time.time()
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process image
        inputs = vit_processor(images=image, return_tensors="pt").to(device)
        outputs = vit_model(**inputs)
        
        # Get the [CLS] token representation
        cls_token = outputs.last_hidden_state[:, 0].cpu().detach().numpy()
        
        process_time = time.time() - start_time
        return {
            "embedding_shape": cls_token.shape,
            "processing_time": process_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
