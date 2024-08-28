
import json
import os
import pickle
import sys
import time
import tkinter as tk
from tkinter import HORIZONTAL, Button, Canvas, Scale

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from handTracker import MediapipeHands, findDistances, findError

f=open('settings.json')
settings=json.load(f)
# del f#Deletes the file object (good practice to free resources).

drawState='Standby'
color='white'
brush_size=20
# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=settings['model_complexity'],
    min_detection_confidence=settings['min_detection_confidence'],
    min_tracking_confidence=settings['min_tracking_confidence']
)
mp_drawing = mp.solutions.drawing_utils

camera=cv2.VideoCapture(settings['camera_port'],cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,settings['window_height'])
camera.set(cv2.CAP_PROP_FRAME_WIDTH,settings['window_width'])
camera.set(cv2.CAP_PROP_FPS,settings['fps'])
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

cv2.namedWindow('OpenCV Paint',cv2.WINDOW_NORMAL)
if settings['fullscreen']:
    cv2.setWindowProperty('OpenCV Paint',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

gesturenames=[]
knowngestures=[]
prevcanvas=np.zeros([settings['window_height'],settings['window_width'],3],dtype=np.uint8)
fps=0
fpsfilter=settings['fpsfilter']
starttime=time.time()# Time for FPS calculation.
savetime=-1# Time to track when an image was last saved.
run=True#to control main loop

if os.path.exists('gesture_data.pkl'):
    with open('gesture_data.pkl','rb') as f:
        gesturenames=pickle.load(f)
        knowngestures=pickle.load(f)
else:
    print('No gesture data found')
    sys.exit()

findhands=MediapipeHands(
    model_complexity=settings['model_complexity'],
    min_detection_confidence=settings['min_detection_confidence'],
    min_tracking_confidence=settings['min_tracking_confidence']
)
threshold=settings['confidence']
keypoints=settings['keypoints']
color_idx=['red','orange','yellow','green','cyan','blue','purple','pink','white','black']
brush_y_positions = []

def is_thumb_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_cmc = landmarks[2]
    
    return (thumb_tip[1] < thumb_ip[1] and thumb_ip[1] < thumb_cmc[1])

def count_fingers(hand_landmarks):
    fingers = []
    
    # Thumb
    if is_thumb_up(hand_landmarks):
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    for tip in [8, 12, 16, 20]:
        if hand_landmarks[tip][1] < hand_landmarks[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return sum(fingers)

def convert_toBNW(frame):#to convert a color image into a black-and-white
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    objectFrame=np.zeros([settings['window_height'],settings['window_width'],3],dtype=np.uint8)
    frame=cv2.addWeighted(frame,.8,objectFrame,.9,0)#BGR , weight of frame , blank(black)image , weight of objframe
    return frame

def clearcanvas():
    global prevcanvas
    #new image
    prevcanvas=np.zeros([settings['window_height'],settings['window_width'],3],dtype=np.uint8)

def preprocess(frame, drawState, fps):
    frameleft = frame[60:settings['window_height']-60, :80]
    objectframeleft = np.zeros([settings['window_height']-120, 80, 3], dtype=np.uint8)
    frameleft = cv2.addWeighted(frameleft, .6, objectframeleft, .9, 0)
    frame[60:settings['window_height']-60, :80] = frameleft
    framebottom = frame[settings['window_height']-60:, :]
    objectframebottom = np.zeros([60, settings['window_width'], 3], dtype=np.uint8)
    framebottom = cv2.addWeighted(framebottom, .6, objectframebottom, .9, 0)
    frame[settings['window_height']-60:, :] = framebottom
    cv2.line(frame, (0, 60), (settings['window_width'], 60), (10, 10, 10), 2)

    cntr = 0
    for x in range(0, settings['window_width'], settings['window_width']//10):
        pt1 = (x, 0)
        pt2 = (x + settings['window_width'], 0)
        pt4 = (x, 60)
        pt3 = (x + settings['window_width'], 60)
        cv2.fillPoly(frame, [np.array([pt1, pt2, pt3, pt4])], settings['color_swatches'][color_idx[cntr]])
        cntr += 1

    cntr = 0
    brush_y_positions.clear()  # Clear previous positions
    for x in range((settings['window_height']-120)//6, settings['window_height']-60, (settings['window_height']-120)//6):
        cv2.circle(frame, (40, x), brush_size, (255, 255, 255), -1)
        brush_y_positions.append(x)
        cntr += 1

    cv2.line(frame, (80, 60), (80, settings['window_height']-60), (10, 10, 10), 1)
    cv2.line(frame, (0, settings['window_height']-60), (int(3.4*settings['window_width']//5), settings['window_height']-60), (10, 10, 10), 1)
    cv2.putText(frame, f'{drawState}', (20, settings['window_height']-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # print(f'Detected Finger Count: {finger_count}')  # This should print the actual finger count detected

    # Place finger count text near the brush width icons
    for i, y in enumerate(brush_y_positions):
        x_pos = int(80 + 50)  # Adjust X position here
        y_pos = y - 10        # Adjust Y position here
        font_size = 0.6       # Adjust font size here
    cv2.putText(frame, f'Fingers: {finger_count-1}', (x_pos, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)  
    cv2.line(frame, (settings['window_width']//8, settings['window_height']-60), (settings['window_width']//8, settings['window_height']), (10, 10, 10), 1)
    pt1 = (settings['window_width']//7, settings['window_height']-50)
    pt2 = (2 * settings['window_width']//7, settings['window_height']-50)
    pt3 = (2 * settings['window_width']//7, settings['window_height']-10)
    pt4 = (settings['window_width']//7, settings['window_height']-10)
    cv2.fillPoly(frame, [np.array([pt1, pt2, pt3, pt4])], settings['color_swatches'][color])
    if color == 'black':
        cv2.putText(frame, f'Eraser', (int(1.22 * settings['window_width'] // 7), settings['window_height']-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.line(frame, (int(1.8 * settings['window_width'] // 6), settings['window_height']-60), (int(1.8 * settings['window_width'] // 6), settings['window_height']), (10, 10, 10), 1)
    if brush_size == 30:
        cv2.circle(frame, (int(2 * settings['window_width'] // 6), settings['window_height']-30), brush_size-4, (255, 255, 255), -1)
    else:
        cv2.circle(frame, (int(2 * settings['window_width'] // 6), settings['window_height']-30), brush_size, (255, 255, 255), -1)

    cv2.line(frame, (int(2.2 * settings['window_width'] // 6), settings['window_height']-60), (int(2.2 * settings['window_width'] // 6), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'C to clear', (int(2.3 * settings['window_width'] // 6), settings['window_height']-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.line(frame, (int(3.15 * settings['window_width'] // 6), settings['window_height']-60), (int(3.15 * settings['window_width'] // 6), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'S to save', (int(3.25 * settings['window_width'] // 6), settings['window_height']-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.line(frame, (int(3.4 * settings['window_width'] // 5), settings['window_height']-60), (int(3.4 * settings['window_width'] // 5), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'Q to quit', (int(3.48 * settings['window_width'] // 5), settings['window_height']-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.line(frame, (int(4.8 * settings['window_width'] // 6), settings['window_height']-60), (int(4.8 * settings['window_width'] // 6), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'G to gesture ', (int(4.8 * settings['window_width'] // 6), settings['window_height']-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.line(frame, (int(3.3 * settings['window_width'] // 4), settings['window_height']-60), (int(3.3 * settings['window_width'] // 4), settings['window_height']), (10, 10, 10), 1)
    cv2.putText(frame, f'Eraser', (int(2.74 * settings['window_width'] // 3), 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'{int(fps)} FPS', (int(3.45 * settings['window_width'] // 4), settings['window_height']-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def drawcircle(circ, color):
    cv2.circle(prevcanvas, circ, brush_size, settings['color_swatches'][color], -1)

def getgesture(gesturedata, fingers):
    global drawState
    if len(gesturedata) == 0:
        drawState = 'No Gesture'
        return
    found = False
    for gesture in gesturedata:
        gesture_fingers = gesture['fingers']
        if len(gesture_fingers) == len(fingers):
            if all([gesture_fingers[i] == fingers[i] for i in range(len(fingers))]):
                drawState = gesture['name']
                found = True
                break
    if not found:
        drawState = 'No Gesture'

def saveimage():
    global savetime
    filename=''
    for i in range(6):
        filename+=f'{time.localtime()[i]}'
    cv2.imwrite(os.path.join('pictures',filename+f'.jpeg'),frame)
    savetime=time.time()

def mouseclick(event,xpos,ypos,*args,**kwargs):
    global color,brush_size,run,prevcanvas
    if event==cv2.EVENT_LBUTTONDOWN:
        if ypos>0 and ypos<60:
            if xpos>0 and xpos<settings['window_width']//10:
                color='red'
            elif xpos>settings['window_width']//10 and xpos<2*settings['window_width']//10:
                color='orange'
            elif xpos>2*settings['window_width']//10 and xpos<3*settings['window_width']//10:
                color='yellow'
            elif xpos>3*settings['window_width']//10 and xpos<4*settings['window_width']//10:
                color='green'
            elif xpos>4*settings['window_width']//10 and xpos<5*settings['window_width']//10:
                color='cyan'
            elif xpos>5*settings['window_width']//10 and xpos<6*settings['window_width']//10:
                color='blue'
            elif xpos>6*settings['window_width']//10 and xpos<7*settings['window_width']//10:
                color='purple'
            elif xpos>7*settings['window_width']//10 and xpos<8*settings['window_width']//10:
                color='pink'
            elif xpos>8*settings['window_width']//10 and xpos<9*settings['window_width']//10:
                color='white'
            else:
                color='black'
        if xpos>0 and xpos<60 and ypos>60 and ypos<settings['window_height']-60:
            diff=(settings['window_height']-120)//6
            if ypos>60 and ypos<60+diff:
                brush_size=5
            elif ypos>60+diff and ypos<60+2*diff:
                brush_size=10
            elif ypos>60+2*diff and ypos<60+3*diff:
                brush_size=15
            elif ypos>60+3*diff and ypos<60+4*diff:
                brush_size=20
            elif ypos>60+4*diff and ypos<60+5*diff:
                brush_size=25
            else:
                brush_size=30
        if xpos>0 and xpos<int(3.3*settings['window_width']//4) and ypos>settings['window_height']-60:
            if xpos>0 and xpos<int(3.15*settings['window_width']//6):
                clearcanvas()
            elif xpos>int(3.15*settings['window_width']//6) and xpos<int(3.4*settings['window_width']//5):
                saveimage()
            else:
                run=False

cv2.setMouseCallback('OpenCV Paint',mouseclick)

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint Application with Hand Gestures")

        self.canvas = Canvas(root, bg='white', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.brush_size = 5
        self.brush_color = 'black'

        self.create_controls()

        self.cap = cv2.VideoCapture(0)
        self.process_video()

    def create_controls(self):
        controls_frame = tk.Frame(self.root)
        controls_frame.pack(fill=tk.X)

        self.brush_size_scale = Scale(controls_frame, from_=1, to_=20, orient=HORIZONTAL, label="Brush Size")
        self.brush_size_scale.pack(side=tk.LEFT)

        self.clear_button = Button(controls_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.save_button = Button(controls_frame, text="Save", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)

    def clear_canvas(self):
        self.canvas.delete("all")

    def save_image(self):
        self.canvas.postscript(file="drawing.eps")
        img = Image.open("drawing.eps")
        img.save("drawing.png", "png")

    def process_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_np = np.array(frame_rgb)
            self.detect_gestures(frame_np)
            cv2.imshow("Hand Gesture Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
        self.root.after(10, self.process_video)

    def detect_gestures(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        finger_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2:  # Approximate circle shape
                    finger_count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    radius = max(w, h) // 2

                    # Draw circle on canvas if exactly one finger is detected
                    if finger_count == 1:
                        self.draw_on_canvas(center_x, center_y, radius)

    def draw_on_canvas(self, x, y, radius):
        # Draw circle on canvas
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, 
                                outline=self.brush_color, width=self.brush_size)
while run:
    dt = time.time() - starttime
    starttime = time.time()
    currentfps = 1 / dt
    fps = fps * fpsfilter + (1 - fpsfilter) * currentfps
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    if settings['coloured_background'] == False:
        frame = convert_toBNW(frame)
    canvas = prevcanvas
    handlandmarks, handstype = findhands.handsdata(frame)

    finger_count = 0  # Initialize finger count

    for idx, handtype in enumerate(handstype):
        if handtype == settings['command_hand']:
            distMatrix = findDistances(handlandmarks[idx])
            error, idx2 = findError(knowngestures, distMatrix, keypoints)
            if error < threshold and idx != -1:
                drawState = gesturenames[idx2]
            else:
                drawState = 'Standby'
            frame = findhands.drawLandmarks(frame, [handlandmarks[idx]], False)
            break

    if settings['command_hand'] not in handstype:
        drawState = 'Standby'
    for idx, handtype in enumerate(handstype):
        if handtype == settings['brush_hand']:
            cv2.circle(frame, (handlandmarks[idx][8][0], handlandmarks[idx][8][1]), brush_size, settings['color_swatches'][color], -1)
            finger_count = count_fingers(handlandmarks[idx])  # Count fingers for the brush hand

            # Color swatches logic
            if handlandmarks[idx][8][1] < 60:
                if handlandmarks[idx][8][0] > 0 and handlandmarks[idx][8][0] < settings['window_width'] // 10:
                    color = 'red'
                elif handlandmarks[idx][8][0] > settings['window_width'] // 10 and handlandmarks[idx][8][0] < 2 * settings['window_width'] // 10:
                    color = 'orange'
                elif handlandmarks[idx][8][0] > 2 * settings['window_width'] // 10 and handlandmarks[idx][8][0] < 3 * settings['window_width'] // 10:
                    color = 'yellow'
                elif handlandmarks[idx][8][0] > 3 * settings['window_width'] // 10 and handlandmarks[idx][8][0] < 4 * settings['window_width'] // 10:
                    color = 'green'
                elif handlandmarks[idx][8][0] > 4 * settings['window_width'] // 10 and handlandmarks[idx][8][0] < 5 * settings['window_width'] // 10:
                    color = 'cyan'
                elif handlandmarks[idx][8][0] > 5 * settings['window_width'] // 10 and handlandmarks[idx][8][0] < 6 * settings['window_width'] // 10:
                    color = 'blue'
                elif handlandmarks[idx][8][0] > 6 * settings['window_width'] // 10 and handlandmarks[idx][8][0] < 7 * settings['window_width'] // 10:
                    color = 'purple'
                elif handlandmarks[idx][8][0] > 7 * settings['window_width'] // 10 and handlandmarks[idx][8][0] < 8 * settings['window_width'] // 10:
                    color = 'pink'
                elif handlandmarks[idx][8][0] > 8 * settings['window_width'] // 10 and handlandmarks[idx][8][0] < 9 * settings['window_width'] // 10:
                    color = 'white'
                else:
                    color = 'black'
            # Brush size logic
            if handlandmarks[idx][8][0]>0 and handlandmarks[idx][8][0]<60 and handlandmarks[idx][8][1]>60 and handlandmarks[idx][8][1]<settings['window_height']-60:
                diff=(settings['window_height']-120)//6
                if handlandmarks[idx][8][1]>60 and handlandmarks[idx][8][1]<60+diff:
                    brush_size=5
                elif handlandmarks[idx][8][1]>60+diff and handlandmarks[idx][8][1]<60+2*diff:
                    brush_size=10
                elif handlandmarks[idx][8][1]>60+2*diff and handlandmarks[idx][8][1]<60+3*diff:
                    brush_size=15
                elif handlandmarks[idx][8][1]>60+3*diff and handlandmarks[idx][8][1]<60+4*diff:
                    brush_size=20
                elif handlandmarks[idx][8][1]>60+4*diff and handlandmarks[idx][8][1]<60+5*diff:
                    brush_size=25
                else:
                    brush_size=30
            # Paint logic
            if drawState=='Draw':
                cv2.circle(canvas, (handlandmarks[idx][8][0], handlandmarks[idx][8][1]), brush_size, settings['color_swatches'][color], -1)
    
    # Display the finger count on the top-right corner of the screen
    # print(f'Finger Count: {finger_count-1}')

    # Ensure this is done before the final frame rendering
    cv2.putText(frame, f'Fingers: {finger_count-1}', (settings['window_width'] - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Change color to red for visibility

    frame = cv2.addWeighted(frame, .6, canvas, 1, 1)
    frame = preprocess(frame, drawState, fps)
    prevcanvas = canvas
    if time.time() - savetime <= 1:
        cv2.putText(frame, 'Image saved successfully', (settings['window_width']//2 - 380, settings['window_height']//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (10, 250, 10), 2)
    cv2.imshow('OpenCV Paint', frame)
    key = cv2.waitKey(1)
    if key & 0xff == ord('q'):
        run = False
    if key & 0xff == ord('s'):
        saveimage()
    if key & 0xff == ord('c'):
        clearcanvas()
    if key & 0xff == ord('g'):
        root = tk.Tk()
        app = PaintApp(root)
        root.mainloop()

camera.release()
cv2.destroyAllWindows()

