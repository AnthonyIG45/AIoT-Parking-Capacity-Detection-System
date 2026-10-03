import time
import cv2
import psutil
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('vehicle_detection_output.avi', fourcc, TARGET_FPS, (640, 640))

process = psutil.Process()

# --- Tracking & Counting Setup for Vertical Line ---
counted_ids_left = set()   # IDs that crossed into the left side
counted_ids_right = set()  # IDs that crossed into the right side
previous_x_positions = {}  # Stores the last known X position for each track ID

entrance_count = 0  # e.g., moving right -> left
exit_count = 0      # e.g., moving left -> right
LINE_X = 320        # Vertical dividing line right down the middle

print("Starting camera feed with vertical split counting...")

try:
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        frame = cv2.resize(frame, (640, 640))

        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
        annotated_frame = results[0].plot()

        # Draw the vertical dividing tripwire line (Blue)
        cv2.line(annotated_frame, (LINE_X, 0), (LINE_X, 640), (255, 0, 0), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)

                if track_id in previous_x_positions:
                    prev_x = previous_x_positions[track_id]

                    # Crossing Left to Right (e.g., Exit)
                    if prev_x <= LINE_X and center_x > LINE_X and track_id not in counted_ids_right:
                        exit_count += 1
                        counted_ids_right.add(track_id)

                    # Crossing Right to Left (e.g., Entrance)
                    elif prev_x >= LINE_X and center_x < LINE_X and track_id not in counted_ids_left:
                        entrance_count += 1
                        counted_ids_left.add(track_id)

                # Update previous X position
                previous_x_positions[track_id] = center_x

        # Display resource stats and separate counts
        cpu_usage = process.cpu_percent(interval=None) / psutil.cpu_count()
        ram_usage_mb = process.memory_info().rss / (1024 * 1024)

        cv2.putText(annotated_frame, f"CPU: {cpu_usage:.1f}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"RAM: {ram_usage_mb:.1f} MB", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Entrances (R->L): {entrance_count}", (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Exits (L->R): {exit_count}", (20, 145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        out.write(annotated_frame)
        cv2.imshow("Parking Lot Tracking", annotated_frame)

        elapsed_time = time.time() - start_time
        if elapsed_time < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Total Entrances: {entrance_count} | Total Exits: {exit_count}")