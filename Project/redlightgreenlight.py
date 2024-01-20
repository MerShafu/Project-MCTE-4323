import cv2
import numpy as np
import time
import random
from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')

vid = cv2.VideoCapture(0)

folderPath = 'Pictures'
mylist = os.listdir(folderPath)
graphic = [cv2.imread(f'{folderPath}/{imPath}') for imPath in mylist]
intro = graphic[0];
kill = graphic[1];
winner = graphic[2];
green = graphic[3];
red = graphic[4];
testing = graphic[5];

test = True

current_img = intro.copy()

while test:
    ret, frame = vid.read()
    result = model(frame, stream = True)
    for r in result:
        boxes = r.boxes
        for bbox in boxes:
            x1,y1,x2,y2 = bbox.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            cls_idx = int(bbox.cls[0])
            cls_name = model.names[cls_idx]
            conf = round(float(bbox.conf[0]),2)
            
            if cls_name == "person":
                perPos = x1,y1,x2,y2
                testroi = frame[y1:y2,x1:x2]

    cv2.imshow('Main', cv2.hconcat([current_img, frame]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        current_img = green.copy()
        cv2.imshow('Main', cv2.hconcat([current_img, frame]))
        test = False
        intro = False


people = {}  # Dictionary to store information about each person
person_id_counter = 0  # Counter for assigning unique IDs to people


prev = time.time()
prevDoll = prev

win = False
player_detected = False
checker = False

people = {}  # Dictionary to store information about each person
# person_id_counter = 0  # Counter for assigning unique IDs to people


while not test and not checker:
    ret, frame = vid.read()

    if not player_detected and (cv2.waitKey(10) & 0xFF == ord(' ')):
        cv2.destroyAllWindows()
        win = True
        break

    cur = time.time()
    no = random.randint(4, 5)
    if cur - prev >= no:
        prev = cur
        #print("RED!")

        if cv2.waitKey(10) & 0xFF == ord('w'):
            cv2.destroyAllWindows()
            win = True
            break

        if not player_detected:
            person_id_counter = 0
            #console = cv2.imshow('RED/GREEN',red)
            current_img = red.copy()
            cv2.imshow('Main', cv2.hconcat([current_img, frame]))
            results = model(frame, stream=True)
            new_people = {}
            print("Detecting..")
            for idx, r in enumerate(results):
                boxes = r.boxes
                for bbox in boxes:
                    x1, y1, x2, y2 = bbox.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls_idx = int(bbox.cls[0])
                    cls_name = model.names[cls_idx]
                    conf = round(float(bbox.conf[0]), 2)

                    if cls_name == "person":
                        person_id = None
                        for pid, pinfo in people.items():
                            px1, py1, px2, py2 = pinfo['position']
                            # Check if the detected person overlaps with an existing person
                            if (x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                person_id = pid
                                break

                        if person_id is None:
                            person_id_counter += 1
                            person_id = person_id_counter

                        roi = frame[y1:y2, x1:x2]
                        ref = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        new_people[person_id] = {
                            'position': (x1, y1, x2, y2),
                            'roi': roi,
                            'last_frame': ref,
                            'moving': False
                        }   
                
            player_detected = True
            people = new_people

            for person_id, person_info in people.items():
                if person_id not in new_people:
                # Close the window for people who are not detected in the current frame
                    cv2.destroyWindow(f'Person {person_id}')

        else:
            #cconsole = cv2.imshow('RED/GREEN',green)
            current_img = green.copy()
            cv2.imshow('Main', cv2.hconcat([current_img, frame]))
            player_detected = False


    if player_detected:

        if person_id == 0: #If there is no player detected
            break

        for person_id, person_info in people.items():
            x1, y1, x2, y2 = person_info['position']
            roi_moving = frame[y1:y2,x1:x2]
            last_frame = person_info['last_frame']
            moving = person_info['moving']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,f'Player:{person_id}',(x1,y1-5),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

            gray = cv2.cvtColor(roi_moving, cv2.COLOR_BGR2GRAY)
            frame_delta = cv2.absdiff(last_frame, gray)
            thresh = cv2.threshold(frame_delta, 20, 255, cv2.THRESH_BINARY)[1] 
            change = np.sum(thresh)
            #print(change)

            if change > 6500000:  # Adjust this threshold as needed (change sensitivity)
                moving = True
                print(f'Person {person_id} is moving!')
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) #Highlighting the player moved 
                cv2.putText(frame,f'Player:{person_id}',(x1,y1-5),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2) 
                cv2.putText(frame,f'Player {person_id} is moving!',(10,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                checker = True
                #person_id_counter -= 1

            else:
                moving = False

    else:
        if cv2.waitKey(10) & 0xFF == ord(' '):
            cv2.destroyAllWindows()
            win = True
            break

    while checker:
        if person_id_counter >= 2:
            ret, frame = vid.read()
            cv2.putText(frame,f'PLayer {person_id} please leave!',(50,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            cv2.putText(frame,f'Press a to continue',(70,120),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            current_img = testing.copy()
            cv2.imshow('Main', cv2.hconcat([current_img, frame]))

            if cv2.waitKey(1) & 0xFF == ord('a'):
                current_img = green.copy()
                cv2.imshow('Main', cv2.hconcat([current_img, frame]))
                checker = False
        else:
            cv2.destroyAllWindows()
            break


    cv2.imshow('Main', cv2.hconcat([current_img, frame]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


vid.release()
cv2.destroyAllWindows()

if not win:
    for i in range(10):
        cv2.imshow('Squid Game',kill)
   
    while True:
        cv2.imshow('Squid Game',kill)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
else:

    cv2.imshow('Squid Game', winner)
    cv2.waitKey(125)
    

    while True:
        cv2.imshow('Squid Game', winner)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()