import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from models.model4 import ModelHeadPose

app = FastAPI()

# Correct initialization (NO argument confusion)
model = ModelHeadPose()

@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_bytes()

            frame = cv2.imdecode(
                np.frombuffer(data, np.uint8),
                cv2.IMREAD_COLOR
            )

            frame = cv2.resize(frame, (320, 240))

            frame = await loop.run_in_executor(
                None, model.process, frame
            )

            _, buffer = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 40]
            )

            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    uvicorn.run(
        "stream_server4:app",
        host="0.0.0.0",
        port=5000
    )