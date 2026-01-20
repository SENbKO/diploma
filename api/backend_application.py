from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import cv2
import uvicorn
from compression.CompressionService import DCTService
import numpy as np
import io
import os
from pathlib import Path
from contextlib import asynccontextmanager
from unet_impl.BackgroundBlurModel import BackgroundBlurModel
from upscaling_model.UpscalingModel import UpscalingModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
frontend_directory = str(Path(project_root) / "front_end")

@asynccontextmanager
async def lifespan(app: FastAPI):
    background_model = Path(project_root) / "unet_impl" / "unet_impl_iou_50_epochs.pth"
    upscaling_model = Path(project_root) / "upscaling_model" / "fsrcnn_x2_y.pth"
    # STARTUP
    app.state.bg_model = BackgroundBlurModel(
        weights_path=str(background_model)
    )
    app.state.upscaling_model = UpscalingModel(
        weights_path=str(upscaling_model)
    )
    print("Models loaded")
    print(project_root)

    yield

    print("App shutting down")

image_processing_api = FastAPI(lifespan=lifespan)

#Hosting for frontend
@image_processing_api.get("/", response_class=HTMLResponse)
def read_root():
    with open(Path(frontend_directory) / "main_page.html", "r", encoding="utf-8") as f:
        return f.read()

dct_service = DCTService(keep_ratio=0.5)


#Loading of static files
image_processing_api.mount(
    "/static",
    StaticFiles(directory=Path(frontend_directory) /"static"),
    name="static",
)

ALLOWED_CONTENT_TYPES = ["image/png", "image/jpeg", "image/tiff", "image/jpg"]

#Endpoint for segmentation and blurring
@image_processing_api.post("/api/v1/blur-background")
async def blur_background(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPEG, or TIFF are allowed.")
    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        image_np = np.array(image)

        result_np = image_processing_api.state.bg_model.predict(image_np)

        buffer = BytesIO()
        Image.fromarray(result_np).save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Endpoint to check if the backend works fine
@image_processing_api.get("/health")
def health():
    return {"status": "ok"}

@image_processing_api.post("/api/v1/compress")
async def dct_transform(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPEG, or TIFF are allowed.")
    #The value to control amount of kept frequency coeffs
    dct_service.keep_ratio = 0.5

    # Read image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        return {"error": "Invalid image file"}

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    processed_img = dct_service.process_image(img)


    _, buffer = cv2.imencode(".jpeg", processed_img)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/jpeg")

#Endpoint for upscaling
@image_processing_api.post("/api/v1/upscale")
async def upscale_image(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPEG, or TIFF are allowed.")
    try:
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")
        print("processing")
        output_image = image_processing_api.state.upscaling_model.transform(image)
        print("output image")
        buffer = BytesIO()
        output_image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(image_processing_api, host="0.0.0.0", port=8000)