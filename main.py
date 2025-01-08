from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import uvicorn  # Importing uvicorn
import os
from io import BytesIO

# Initialize the FastAPI app
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
MODEL_PATH = r"C:\Users\Nikshith\Documents\CVPRO\best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
MODEL = YOLO(MODEL_PATH)

# Class names from the model
CLASS_NAMES = ['Bacterial', 'Downy mildew', 'Healthy', 'Powdery mildew', 'Septoria Blight', 'Virus', 'Wilt - Leaf Blight']

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data: bytes) -> Image.Image:
    """Reads the uploaded file as an image."""
    return Image.open(BytesIO(data))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Log the file details
    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    # Validate file MIME type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, JPEG, and PNG are allowed.")
    
    # Save the input image temporarily
    input_image_path = "temp_image.jpg"
    output_image_path = "temp_output.jpg"

    try:
        # Save the input image
        image = read_file_as_image(await file.read())
        image.save(input_image_path)

        # Run inference with YOLOv8
        results = MODEL.predict(source=input_image_path, conf=0.25)

        # Generate the output image with bounding boxes
        output_image = results[0].plot()  # Draws bounding boxes on the image
        output_image_pil = Image.fromarray(output_image)
        output_image_pil.save(output_image_path)

        # Extract detections
        detections = []
        for result in results:
            for box in result.boxes:
                label = CLASS_NAMES[int(box.cls.item())]
                confidence = float(box.conf.item())
                detections.append({"label": label, "confidence": confidence})

        # Return the processed image and detections
        return {
            "detections": detections,
            "image_url": "/output-image"
        }
    finally:
        # Clean up input image
        if os.path.exists(input_image_path):
            os.remove(input_image_path)

@app.get("/output-image")
async def get_output_image():
    """Serve the output image with bounding boxes."""
    output_image_path = "temp_output.jpg"
    if not os.path.exists(output_image_path):
        raise HTTPException(status_code=404, detail="Output image not found")
    return FileResponse(output_image_path, media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
