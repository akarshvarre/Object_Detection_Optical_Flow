
import numpy as np
import cv2
from scipy import signal

from estimateFeatureTranslation import estimateFeatureTranslation

def estimateAllTranslation(startXs,startYs,img1,img2):
    newXs = np.zeros(startXs.shape)
    newYs = np.zeros(startYs.shape)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    Iy = signal.convolve(img2,np.array([[1], [-1]]).T)
    Ix = signal.convolve(img2,np.array([[1], [-1]]))
    num_features = 0
    for startX, startY in zip(startXs.T, startYs.T):
        newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)
        newXs[:, num_features], newYs[:, num_features] = newX, newY
        num_features += 1
    return newXs, newYs