"""
Live hand-detection system.
Reads webcam frames, detects hands with the fine-tuned YOLOv8n model (Model 3),
and sends a note to Sonic Pi over OSC when a hand enters one of four screen zones.
Prints the mean detection time per frame on exit.
"""

from ultralytics import YOLO
import cv2 as cv
import time
import statistics
from pythonosc import udp_client

CAMERA_INDEX = 0  # 0 = built-in webcam, change if using an external camera
model = YOLO("face_model_seed0.pt")  # Trained model
model.to("mps")          # Apple Silicon GPU

osc_client = udp_client.SimpleUDPClient("127.0.0.1", 4560)

def run_camera():
    stream = cv.VideoCapture(CAMERA_INDEX, cv.CAP_AVFOUNDATION)
    window_name = "Hand Detection"
    cv.namedWindow(window_name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    time.sleep(1)

    if not stream.isOpened():
        print("Could not open camera")
        exit()

    ret, frame = stream.read()
    height, width = frame.shape[:2]

    rec_piano_width = int(width/4)
    rec_piano_height = int(height)
    colour = (0, 255, 0)


    piano_coordinates = [] 
    for i in range(4):
        start_point = (i * rec_piano_width, 0)
        piano_coordinates.append(start_point)
    
    def play_sound(key_index):
        num = 60
        note = key_index + num
        osc_client.send_message("/note/play", [note])

    last_zone = None
    latencies = []
    while(True):
        ret, frame = stream.read()
        
        if not ret:
            print("Camera stream ended")
            break
        
        frame = cv.flip(frame, 1)  # flip horizontally
        t0 = time.perf_counter()
        results = model(frame, imgsz=320, conf=0.6, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)
        annotated = results[0].plot(img=frame)
        current_zone = None

        # Bounding box coordinates
        for box in results[0].boxes:
           
            x1, y1, x2, y2 = map(int, box.xyxy[0])
                
            # centre of detected hand
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # check which piano zone the hand is in
            for i, start_point in enumerate(piano_coordinates):
                if start_point[0] < center_x < start_point[0] + rec_piano_width and center_y < rec_piano_height:
                    current_zone = i
                    break
    
        # Only play sound if hand moved into a new zone
        if current_zone is not None and current_zone != last_zone:
            play_sound(current_zone)
    
        last_zone = current_zone  # update last zone

        # Draw piano keys
        for i, start_point in enumerate(piano_coordinates):
            # start at origin (0,0), end of start point + width  of rectangle 
            end_point = ((i + 1) * rec_piano_width, rec_piano_height)
            cv.rectangle(annotated, start_point, end_point, colour, 2)
         # quit textbox 
        cv.rectangle(annotated, (0, height - 50), (280, height), 0, -1)
        cv.putText(annotated, "Press 'q' to quit", (10, height - 20), cv.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

        cv.imshow(window_name, annotated)
    
        if cv.waitKey(1) == ord('q'):
            break

    stream.release()
    cv.destroyAllWindows()

    # skip first 10 frames
    timed = latencies[10:]  
    print(f"Frames: {len(timed)}")
    print(f"Detection time per frame: {statistics.mean(timed):.1f} ± {statistics.stdev(timed):.1f} ms")

if __name__ == "__main__":
    run_camera()