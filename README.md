# Smart Parking Occupancy Tracking System (Edge YOLO & IoT Architecture)

An edge-computing, computer vision-based smart parking solution designed to run on low-cost hardware (such as a Raspberry Pi) and sync real-time occupancy data over Wi-Fi to a central server.

---

## 1. System Architecture
* **Edge Nodes (Raspberry Pi + Camera):** Captures video feeds, runs lightweight object detection (`YOLO26` Nano/Small) restricted to vehicle classes, performs line-crossing tracking (entry/exit detection), and throttles processing to a sustainable 15 FPS.
* **Communication Layer:** Uses lightweight HTTP REST requests (or MQTT) over local Wi-Fi to push incremental state changes (`+1` or `-1`) to the central system.
* **Central Server:** Maintains the live occupancy database, aggregates data across multiple gates, and provides a dashboard interface.

---

## 2. Prerequisites & Installation

### Dependencies
Install the required Python libraries on your testing machine (or Raspberry Pi):
```bash
pip install ultralytics opencv-python psutil requests
```

### Script Setup
Ensure you have your model weights downloaded (`yolo26s.pt` or `yolo26n.pt`) in the root directory alongside your main Python script.

---

## 3. Evaluation Framework: Accuracy & Cost-Effectiveness

To validate this implementation against existing commercial or traditional alternatives (such as ultrasonic loop detectors, magnetic ground sensors, or expensive proprietary ANPR camera systems), use the following comparison criteria:

### A. Accuracy Metrics
* **Counting Accuracy ($A_c$):** 
  $$\text{Accuracy} = \left(1 - \frac{|\text{Actual Count} - \text{System Count}|}{\text{Actual Count}}\right) \times 100$$
  * *Test Method:* Record a 30-minute test video or live session with a known number of entering/exiting vehicles and compare system logs against ground truth.
* **False Positive/Negative Rate:** Track instances where shadows, pedestrians, or weather events trigger false vehicle entries/exits.

### B. Cost-Effectiveness Breakdown
| Implementation Type | Hardware Cost per Gate | Installation Complexity | Maintenance / Scaling Cost |
| :--- | :--- | :--- | :--- |
| **This Edge YOLO Pi System** | **Low** (~$50–$80 per Pi + Camera) | **Medium** (Mounting & power setup) | **Low** (Over-the-air software updates) |
| Ultrasonic/Magnetic Ground Sensors | High (Civil engineering required) | Very High (Trenching/paving) | High (Battery replacement & road wear) |
| Proprietary Enterprise ANPR Cameras | Very High ($500+ per unit) | Low-Medium | High (Proprietary licensing fees) |

---

## 4. Next Steps for Testing
1. **Local Benchmark:** Run the script on your Microsoft Surface to verify model confidence and bounding box stability.
2. **Field Simulation:** Test with recorded vehicle traffic videos to measure line-crossing accuracy before deploying to physical hardware.
3. **Network Resiliency:** Implement local buffering on the Raspberry Pi to prevent lost occupancy counts if the Wi-Fi connection drops temporarily.