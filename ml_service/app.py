"""
PlantVerse — Plant Disease Detection ML Microservice
===================================================
Production-ready ASGI / REST API for plant leaf disease inference.

Endpoints:
- GET  /health               : Service health check & model metadata
- POST /api/disease/predict  : Run neural disease prediction on uploaded image
- POST /api/disease/diagnosis: Save or validate diagnosis record
"""

import os
import sys
import time
from typing import Dict, Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# Ensure parent directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_service.services.disease_predictor import predictor
from ml_service.utils.preprocessing import validate_and_load_image

START_TIME = time.time()


async def health(request: Request) -> JSONResponse:
    """Service health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "PlantVerse Plant Disease Prediction ML API",
        "version": "1.0.0",
        "uptimeSeconds": round(time.time() - START_TIME, 1),
        "modelBackend": predictor.backend,
        "modelLoaded": predictor.is_loaded,
        "threshold": predictor.threshold,
    })


async def predict(request: Request) -> JSONResponse:
    """
    Run plant disease prediction on an uploaded image.
    Expects multipart/form-data with 'image' field.
    """
    try:
        form = await request.form()
        image_field = form.get("image") or form.get("file")

        if not image_field:
            return JSONResponse(
                {
                    "success": False,
                    "error": "No image uploaded. Please provide an image file under the 'image' field."
                },
                status_code=400
            )

        # Read file contents
        if hasattr(image_field, "read"):
            if hasattr(image_field, "seek"):
                res = image_field.seek(0)
                if hasattr(res, "__await__"):
                    await res
            content = await image_field.read()
        else:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Invalid upload format."
                },
                status_code=400
            )

        if not content or len(content) == 0:
            return JSONResponse(
                {
                    "success": False,
                    "error": "Uploaded image file is empty."
                },
                status_code=400
            )

        # Preprocess and validate
        try:
            image = validate_and_load_image(content)
        except ValueError as val_err:
            return JSONResponse(
                {
                    "success": False,
                    "error": str(val_err)
                },
                status_code=400
            )

        # Execute prediction
        result = predictor.predict(image)
        return JSONResponse(result)

    except Exception as exc:
        return JSONResponse(
            {
                "success": False,
                "error": f"Internal prediction engine error: {exc}"
            },
            status_code=500
        )


async def save_diagnosis(request: Request) -> JSONResponse:
    """
    Save or validate a plant disease diagnosis record.
    """
    try:
        body = await request.json()
        required = ["plant", "disease", "confidence"]
        for field in required:
            if field not in body:
                return JSONResponse(
                    {
                        "success": False,
                        "error": f"Missing required diagnosis field: {field}"
                    },
                    status_code=400
                )

        diagnosis_record = {
            "success": True,
            "message": "Diagnosis validated and received.",
            "record": {
                "plant": body.get("plant"),
                "disease": body.get("disease"),
                "healthStatus": body.get("healthStatus", "Diseased"),
                "confidence": body.get("confidence"),
                "location": body.get("location"),
                "timestamp": body.get("timestamp", time.time()),
            }
        }
        return JSONResponse(diagnosis_record)

    except Exception as err:
        return JSONResponse(
            {
                "success": False,
                "error": f"Unable to parse diagnosis payload: {err}"
            },
            status_code=400
        )


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    print("[ML Service] Initializing model inference engine at startup...")
    predictor.load_model()
    print("[ML Service] Ready to receive prediction requests.")
    yield


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/api/disease/predict", predict, methods=["POST"]),
    Route("/api/disease/diagnosis", save_diagnosis, methods=["POST"]),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"]
    )
]

app = Starlette(
    debug=False,
    routes=routes,
    middleware=middleware,
    lifespan=lifespan
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("ml_service.app:app", host="0.0.0.0", port=port, reload=False)
