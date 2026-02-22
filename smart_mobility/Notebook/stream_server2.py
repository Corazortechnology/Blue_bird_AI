import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from models.model2 import ArcFaceModel

app = FastAPI()

DB_PATH = "models/db_embeddings.npz"

print("🚀 Starting ArcFace prediction WebSocket server (no training on connect)...")
model = ArcFaceModel(load_db=True, db_path=DB_PATH)  # does not build DB; just loads if exists
print("✅ Model initialized. DB size:", len(model.db))

@app.get("/")
def home():
    return {"status": "ArcFace prediction server running", "db_size": len(model.db)}

@app.websocket("/arcface")
async def arcface_stream(websocket: WebSocket):
    await websocket.accept()
    print("🔵 Client connected")

    try:
        while True:
            data = await websocket.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # optional resize to keep CPU stable
            frame = cv2.resize(frame, (480, 360))

            frame = model.process(frame)

            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("🔴 Client disconnected")
    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    uvicorn.run("predict_server:app", host="0.0.0.0", port=5000, reload=False)
