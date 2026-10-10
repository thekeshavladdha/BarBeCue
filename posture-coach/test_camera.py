"""BarBeCue test_camera.py — verify order step 1 (Appx D.1.4).

Run: python test_camera.py
Tries index 0, then 1, 2. 'q' quits. If black: close Zoom/Meet holding camera.
"""
import cv2

for idx in (0, 1, 2):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"[camera] index {idx} failed, trying next...")
        continue
    print(f"[camera] opened index {idx}. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[camera] read failed")
            break
        cv2.imshow(f"BarBeCue camera test (index {idx}) - q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    break
else:
    print("[camera] no device 0/1/2. Close video-call apps, check USB, or use data/*.mp4 offline.")
