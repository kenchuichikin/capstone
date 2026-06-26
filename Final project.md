import cv2
import os
import random
import time
import threading
import queue
import dill
import urllib.request
from firenet import construct_firenet
try:
  import paho.mqtt.client as mqtt
  HAS_MQTT = True
except ImportError:
  HAS_MQTT = False

keep_processing = True;

def download_video():
  #url = "https://"
  vid_path = "Firewood.mp4"
  if not os.path.exists(vid_path):
    print("[*] Downloading sample video...")
    urllib.request.urlretrieve(url, vid_path)
  return vid_path

def publish_data(payload):
  frame = payload
  byte_array = bytearray(dill.dumps(frame))
  MQTT_BROKER = "localhost" # Broker's IP address
  MQTT_PORT = 1883
  """Simulate MQTT Publishing"""
  # Simulate a connection success rate of 80%
  success = random.random() > 0.2 
  if success and HAS_MQTT:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    #client.connect(MQTT_BROKER, port=MQTT_PORT)
    time.sleep(4) # Wait for connection setup to complete
    #client.loop_start()
    client.publish("homie/mac_webcam/capture", byte_array, qos=1)
    print(f"[CLOUD] Successfully published via MQTT QoS 1.")
  else:
    print(f"[CACHE] Connection failed or missing package.")

# Initialize a thread-safe queue
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

def producer_thread(video_path, case):
  video = cv2.VideoCapture(video_path)
  frame = None
  # Get video properties
  w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print("[Producer] Started loading video...")
  while not stop_event.is_set():
    # Read video frames from file
    ret, frame = video.read()
    if not ret:
      if case == 1:
        print("... end of video file reached")
        break
      else:
        # Loop the video if it reaches the end
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    print("[Producer] Sending video frames...")
    # Put frames into a queue without blocking
    try: frame_queue.put_nowait(frame)
    except queue.Full:
      try: frame_queue.get_nowait()
      except queue.Empty: pass
      frame_queue.put_nowait(frame)
      # Simulate camera hardware delay of approx. 30FPS
    time.sleep(0.03)
    publish_data(frame_queue)
    video.release()
  return w,h

def consumer_thread(width, height):
  # Network input sizes
  rows = 224
  cols = 224
  frame = None
  resized_frame = None
  # Display settings
  window_name = "Live Fire Detection - FireNet"
  # Create window
  #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
  print("[Consumer] Loading inference model...")
  # Use FireNet CNN model
  model = construct_firenet (rows, cols, training=False)
  model.load(os.path.join("models/FireNet", "firenet"), weights_only=True)
  while not stop_event.is_set():
    # Retrieve frame and run inference
    try: frame = frame_queue.get(timeout=1.0)
    except queue.Empty: continue
    # Resize image to network input size
    resized_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)
    converted_frame = resized_frame[:, :, ::-1] # Convert to RGB
    # Perform prediction
    output = model.predict([converted_frame])
    # Label image based on prediction
    if round(output[0][0]) == 1: # Equivalent to 0.5 threshold
      cv2.rectangle(frame, (0,0), (width, height), (0,0,255), 50)
      cv2.putText(frame, 'Fire', (int(width/16), int(height/4)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 10, cv2.LINE_AA)
    else:
      cv2.rectangle(frame, (0,0), (width, height), (0,255,0), 50)
      cv2.putText(frame, 'Clear', (int(width/16), int(height/4)), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 10, cv2.LINE_AA)
    # Image display
    #cv2.imshow(window_name, frame)
    cv2.imwrite("output.png", frame)
    #cv2.waitKey(0)
    continue

if __name__ == '__main__':
  while (keep_processing):
    vid_path = download_video()
    w, h = producer_thread(vid_path, 1)
    # Start threads
    t_prod = threading.Thread(target=producer_thread, args=(vid_path, 0))
    t_cons = threading.Thread(target=consumer_thread, args=(w,h))
    t_prod.start()
    t_cons.start()
    try:
      print("\n[Main] System running. Let it run for 60 seconds...")
      time.sleep(60)
      print("\n[Main] Time's up. Initiating shutdown...")
    except KeyboardInterrupt:
      print("\n[Main] Interrupted by user. Shutting down...")
    finally:
      # Signal all threads to stop
      stop_event.set()
      t_prod.join()
      t_cons.join()
      print("[Main] All threads closed cleanly. System exit.")
      keep_processing = False
