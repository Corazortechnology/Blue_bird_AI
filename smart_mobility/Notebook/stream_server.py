import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from models.model1 import Model1RetinaFace
#from models.model2 import Model2ArcFace
#from models.model3 import Model3Drowsiness

import uvicorn

app = FastAPI()

# Load model once
model1 = Model1RetinaFace()
#model3 = Model3Drowsiness()

@app.get("/")
def home():
    return {"status": "Frame Relay Server Running"}

@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            data = await websocket.receive_bytes()

            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Optional resize for stability
            frame = cv2.resize(frame, (480, 360))

            # ✅ Correct call
            detections = model1.detect(frame)
            frame = model1.draw(frame, detections)
#           frame = model3.process(frame)


            _, buffer = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            )

            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("stream_server:app", host="0.0.0.0", port=5000)
