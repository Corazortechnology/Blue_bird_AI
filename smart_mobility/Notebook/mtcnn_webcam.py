import cv2
from mtcnn import MTCNN

cap = cv2.VideoCapture(0)
detector = MTCNN()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        output = detector.detect_faces(frame)
    except:
        output = []

    for face in output:
        if 'box' not in face or 'keypoints' not in face:
            continue

        x, y, w, h = face['box']
        if w <= 0 or h <= 0:
            continue

        keypoints = face['keypoints']

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for point in keypoints.values():
            cv2.circle(frame, point, 4, (0, 255, 0), -1)

    cv2.imshow("MTCNN Safe Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
