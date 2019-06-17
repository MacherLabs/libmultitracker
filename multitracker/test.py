import dlib
import cv2
import argparse as ap
import numpy as np
from __init__ import *
import cProfile as profile
import logging
logging.basicConfig(level=logging.DEBUG)

def hogpoints(img):
    new_points=[]
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    points,w=hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05) 
    for x,y,w,h in points:
        new_points.append((int(x),int(y),int(w),int(h)))
    return new_points

def run(source=0, dispLoc=False):
    # Create the VideoCapture object
    cam = cv2.VideoCapture("../../a.avi")
    # mtracker = MultiTracker(SingleTrackerType = cv2.TrackerKCF_create)
    mtracker = MultiTracker(SingleTrackerType = CorrelationTracker)

    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print "Video device or file couldn't be opened"
        exit()

    while True:
        retval,img = cam.read()
        points = hogpoints(img)
        print points
        if len(points) != 0:
            break

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)


    mtracker.add_trackers(img,points)


    frame_number =0
    while frame_number<1000:
        # Read frame from device or file
        retval, img = cam.read()
        if not retval:
            print "Cannot capture frame device | CODE TERMINATION :( "
            exit()

        # points = get_points.run(img, multi=True)
        
        # Update tracker

        # profile.run('temp = mtracker.update(img,points)')
        # import pdb;pdb.set_trace()
        if frame_number%10==0:
            points = hogpoints(img)
            # Start timer
            timer = cv2.getTickCount()
            temp = mtracker.update(img,points)
        else:
            # Start timer
            timer = cv2.getTickCount()
            temp = mtracker.update(img)
        # ok, bbox = tracker.update(frame)
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        logging.info("fps: {}".format(fps))

        # Update the tracker  
        for tid,tracker in mtracker.trackers.items():
            # Get the position of th object, draw a 
            # bounding box around it and display it.
            rect = tracker.bbox
            pt1 = (int(rect[0]),int(rect[1]))
            pt2 = (int(rect[0]+rect[2]),int(rect[1]+rect[3]))
            color =(255, 255, 255)
            if tracker.consecutive_invisible_count> 10: color = (255, 0, 0)
            cv2.rectangle(img, pt1, pt2, color , 3)
            # print "Object {} tracked at [{}, {}] \r".format(i, pt1, pt2),
            if True:
                loc = (pt2[0]-20, pt1[1]+20)
            txt = str(tid)
            cv2.putText(img, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break
        frame_number +=1

    # Relase the VideoCapture object
    cam.release()

if __name__ == "__main__":
    run(0,False)
    profile.run('run(0,False)')