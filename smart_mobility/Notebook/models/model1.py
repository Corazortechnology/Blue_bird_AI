import cv2
from retinaface import RetinaFace

def main():
    cap = cv2.VideoCapture(0)

    print("Starting Live Face Detection...")
    print("Press 'x' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        try:
            detections = RetinaFace.detect_faces(frame)
        except:
            detections = {}

        if detections:
            for key in detections:
                face = detections[key]

                if "facial_area" not in face:
                    continue

                x1, y1, x2, y2 = face["facial_area"]

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0, 255, 0), 2)

            cv2.putText(frame, "FACE DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "NO FACE",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("x"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
