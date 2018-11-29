'''
  File name: run.py
  Author:
  Date created:
'''
import cv2

from objectTracking import objectTracking

if __name__ == "__main__":
    filename = './Easy.mp4'
    vid = cv2.VideoCapture(filename)
    objectTracking(vid)
    vid.release()