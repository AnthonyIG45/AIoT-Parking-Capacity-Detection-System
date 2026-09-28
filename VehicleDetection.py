import time
import cv2
import psutil
from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

# Force DirectShow backend to grab the Surface's built-in camera reliably
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Target specs
TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Setup Video Writer (saves as an AVI file using the XVID codec at 15 FPS, 640x640)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('vehicle_detection_output.avi', fourcc, TARGET_FPS, (640, 640))

# Initialize process handle for resource tracking
process = psutil.Process()

print("Starting camera feed, live window, and recording...")

try:
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        # Resize input frame to 640x640 for lightweight processing
        frame = cv2.resize(frame, (640, 640))

        # Run YOLO26 inference (classes 2, 3, 5, 7 correspond to vehicles)
        results = model(frame, verbose=False, classes=[2, 3, 5, 7])
        annotated_frame = results[0].plot()

        # Write the annotated frame to the recording file
        out.write(annotated_frame)

        # Get real-time resource metrics
        cpu_usage = process.cpu_percent(interval=None) / psutil.cpu_count()
        ram_usage_mb = process.memory_info().rss / (1024 * 1024)

        # Display resource stats on the video window
        cv2.putText(annotated_frame, f"CPU: {cpu_usage:.1f}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"RAM: {ram_usage_mb:.1f} MB", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the live display window
        cv2.imshow("Vehicle Detection & Recording", annotated_frame)

        # Frame-rate limiting logic (15 FPS cap)
        elapsed_time = time.time() - start_time
        if elapsed_time < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed_time)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording saved successfully.")