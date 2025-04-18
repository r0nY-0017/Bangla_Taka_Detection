import os
import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import time

# Save images to directories
base_directory = 'Taka/'
for i in range(10):
    os.makedirs(os.path.join(base_directory, str(i)), exist_ok=True)

# Start scrcpy and wait for it to open
print("🔄 Launching scrcpy...")
os.system(r'start /B C:\Users\rony\scrcpy\scrcpy-win64-v3.1\scrcpy.exe --max-fps=30 --stay-awake --turn-screen-off')
time.sleep(10)  # Wait a few seconds for scrcpy to open

# Ensure scrcpy window is detected
scrcpy_window = None
while not scrcpy_window:
    windows = gw.getWindowsWithTitle("Redmi Note 7")
    if windows:
        scrcpy_window = windows[0]
        print("✅ scrcpy window detected!....")
    else:
        print("🔄 Waiting for scrcpy window.......")
        time.sleep(1)

# Start capturing frames
while True:
    # Capture scrcpy screen
    screenshot = pyautogui.screenshot(region=(scrcpy_window.left, scrcpy_window.top, scrcpy_window.width, scrcpy_window.height))
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize for better display
    frame = cv2.resize(frame, (540, 1170))
    
    # Rotate the frame to landscape mode
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Draw a region for hand capture
    cv2.rectangle(frame, (170, 70), (830, 450), (255, 255, 255), 2)

    # Show mobile screen
    cv2.imshow("Mobile Camera", frame)
    
    cropped_frame = frame[70:450, 170:830]

    # Get counts for naming
    Signs = {str(i): len(os.listdir(os.path.join(base_directory, str(i)))) for i in range(10)}

    # Capture image when key is pressed
    key = cv2.waitKey(1) & 0xFF
    

    if key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
        label = chr(key)  # Convert key to string
        save_path = os.path.join(base_directory, label, f"{Signs[label]}.png")
        cv2.imwrite(save_path, cropped_frame)
        print(f"✅ Saved: {label} Gesture at {save_path}")
    elif key == 27:  # Press 'Esc' to exit
        print("🛑 Exiting...")
        break

cv2.destroyAllWindows()