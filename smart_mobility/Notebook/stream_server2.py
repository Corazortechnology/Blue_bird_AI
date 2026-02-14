import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from models.model2 import Model2ArcFace
import uvicorn

app = FastAPI()

# -----------------------------------------
# Load ArcFace Model (Training happens here)
# -----------------------------------------
print("🚀 Starting ArcFace WebSocket Server...")
model2 = Model2ArcFace()
print("✅ Server Ready\n")


@app.get("/")
def home():
    return {"status": "ArcFace WebSocket Server Running"}


# -----------------------------------------
# WebSocket for ArcFace Recognition
# -----------------------------------------
@app.websocket("/arcface")
async def arcface_stream(websocket: WebSocket):
    await websocket.accept()
    print("🔵 ArcFace client connected")

    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_bytes()

            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Optional resize for stability
            frame = cv2.resize(frame, (480, 360))

            # 🔥 Face Recognition (Testing)
            frame = model2.process(frame)

            # Encode processed frame
            _, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            )

            # Send back to client
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("🔴 ArcFace client disconnected")

    except Exception as e:
        print("❌ Error:", e)


# -----------------------------------------
# Run Server
# -----------------------------------------
if __name__ == "__main__":
    uvicorn.run("stream_server:app",
                host="0.0.0.0",
                port=5000,
                reload=False)
