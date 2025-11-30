from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import threading
import simpleaudio as sa  # for alarm sound

app = Flask(__name__)

# Face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (60, 68)  # inner mouth

# Global alarm state
ALARM_ON = False
alarm_obj = None
lock = threading.Lock()

# ---------------- Helper Functions ----------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[6])
    B = distance.euclidean(mouth[3], mouth[5])
    C = distance.euclidean(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)

def play_alarm():
    global ALARM_ON, alarm_obj
    with lock:
        if not ALARM_ON:
            ALARM_ON = True
            try:
                wave_obj = sa.WaveObject.from_wave_file("drowsiness_alarm.wav")  # Make sure this file exists
                alarm_obj = wave_obj.play()
                print("🔊 Alarm started")
            except Exception as e:
                print("Error playing alarm:", e)

def stop_alarm_func():
    global ALARM_ON, alarm_obj
    with lock:
        if ALARM_ON and alarm_obj is not None:
            alarm_obj.stop()
            ALARM_ON = False
            print("🔇 Alarm stopped")

# ---------------- Frame Generator ----------------
def generate_frames():
    global ALARM_ON
    cap = cv2.VideoCapture(0)
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 15
    MOUTH_AR_THRESH = 0.7
    counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Drowsiness detection
            if ear < EYE_AR_THRESH:
                counter += 1
                if counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    threading.Thread(target=play_alarm).start()
            else:
                counter = 0

            # Yawning detection
            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "YAWNING ALERT!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                threading.Thread(target=play_alarm).start()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ---------------- Flask Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm_route():
    stop_alarm_func()
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    app.run(debug=True)
