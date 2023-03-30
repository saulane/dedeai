from io import BytesIO
import os
import sys

from fastapi import FastAPI,File,HTTPException,UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.logger import logger

from PIL import Image
import torch
from module import ControlNetMLSD
import uvicorn
import logging

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the pytorch model
    model = ControlNetMLSD(device)

    # add model and other preprocess tools too app state
    app.package = {
        "model": model
    }

@app.get("/ping")
def ping():
    return Response("pong")

@app.post('/prediction')
async def prediction(room: str, style: str, file: UploadFile = File(...)):
     # Ensure that the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    content = await file.read()
    img = Image.open(BytesIO(content))
    # Read processed image from file
    controlnet_model = app.package["model"]
    processed_image = controlnet_model.generate_img(img, room, style)

    img_io = BytesIO()
    processed_image.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/jpeg")

@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }

# if __name__ == '__main__':
#     uvicorn.run("main:app", host="127.0.0.1", port=5000)