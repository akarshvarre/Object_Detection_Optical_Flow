import numpy as np
import cv2
import matplotlib.pyplot as plt

from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation
from getFeatures import getFeatures

def objectTracking(rawVideo):
    frames = rawVideo.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    top_left_pts = []
    bottom_right_pts = []
    startXs = []
    startYs = []
    objects = 0
    bbox = []
    prevFrame = []
    out = cv2.VideoWriter('Easy_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), rawVideo.get(cv2.CAP_PROP_FPS), (int(rawVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(rawVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while rawVideo.isOpened():
        ret, frame = rawVideo.read()
        if not ret:
            break
        frame = np.array(frame)
        count += 1
        print('\rProcessing frame ' + str(count) + ' of ' + str(frames), end = '')
        F = np.copy(frame)
        if count == 1:
            fig = plt.figure()
            plt.imshow(frame)
            im_pts = fig.ginput(n = -1, timeout = 0, show_clicks = True)
            im_pts = np.dstack((np.array(im_pts)[:, 1], np.array(im_pts)[:, 0]))[0]
            top_left_pts = im_pts[np.arange(0, im_pts.shape[0], 2), :].round().astype(np.int32)
            bottom_right_pts = im_pts[np.arange(1, im_pts.shape[0], 2), :].round().astype(np.int32)
            assert top_left_pts.shape == bottom_right_pts.shape
            objects = int(len(im_pts) / 2)
            if objects < 1:
                print('No object selected')
                raise SystemExit
            for pts in range(objects):
                cv2.rectangle(F, (top_left_pts[pts][1], top_left_pts[pts][0]), (bottom_right_pts[pts][1], bottom_right_pts[pts][0]), (0,255,0), 2)

            bbox = np.hstack([top_left_pts, bottom_right_pts]).reshape([objects, 2, 2]) #Just sending the top_left and bottom_right points for now
            startXs, startYs = getFeatures(frame, bbox)
            for pts in range(objects):
                for x, y in zip(startXs[:, pts], startYs[:, pts]):
                    cv2.circle(F, (int(y), int(x)), 3, 255, -1)            
        else:
            if(bbox.shape[0]):
                newXs, newYs = estimateAllTranslation(startXs, startYs, prevFrame, frame)
                [Xs, Ys, newbox] = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)
                delete_objs = []
                for objs in range(bbox.shape[0]):
                    if (np.any(np.logical_or(newXs[:, objs] > frame.shape[0], newXs[:, objs] < 0)) or np.any(np.logical_or(newYs[:, objs] > frame.shape[1], newYs[:, objs] < 0))):
                        delete_objs.append(objs)
                if(len(delete_objs)):
                    newXs = np.delete(newXs, delete_objs, 1)
                    newYs = np.delete(newYs, delete_objs, 1)
                    newbox = np.delete(newbox, delete_objs, 0)
                for pts in range(newbox.shape[0]):
                    for x, y in zip(Xs[:, pts], Ys[:, pts]):
                        cv2.circle(F, (int(y), int(x)), 3, 255, -1)  
                    b = np.int0(newbox[pts,:,:])
                    cv2.rectangle(F, (b[0][1], b[0][0]), (b[1][1], b[1][0]), (0,255,0), 2)

                startXs, startYs =  np.copy(newXs), np.copy(newYs)
                bbox = np.copy(newbox)

        out.write(F)
        cv2.imwrite('Easy_frame-' + str(count) + '.png', F)
        prevFrame = np.copy(frame)

    out.release()