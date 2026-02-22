import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import os

from models.model3 import Model3Drowsiness

app = FastAPI()

# Load model once at startup
model_path = os.environ.get("MODEL3_PATH", "models/face_landmarker.task")
model3 = Model3Drowsiness(model_path=model_path)


@app.get("/")
def home():
    return {"status": "Frame Relay Server Running"}


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()

            # Decode frame
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            try:
                # Resize for stability
                frame = cv2.resize(frame, (480, 360))

                # Process frame
                frame = model3.process(frame)

                # Encode processed frame
                success, buffer = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 60],
                )

                if not success:
                    continue

                # Send back to client
                await websocket.send_bytes(buffer.tobytes())

            except Exception as e:
                print("Frame processing error:", e)
                continue

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print("Unexpected server error:", e)


if __name__ == "__main__":
    # IMPORTANT: filename must match this string
    uvicorn.run("stream_server3:app", host="0.0.0.0", port=5000)