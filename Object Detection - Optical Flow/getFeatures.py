
import numpy as np
import cv2

def getFeatures(img,bbox):
    num_objects = bbox.shape[0]
    features = []
    max_num_features = 0
    max_points = 25
    for objects in range(num_objects):
        im_obj = img[bbox[objects, 0, 0] : bbox[objects, 1, 0],  bbox[objects, 0, 1] : bbox[objects, 1, 1], :]

        img_feature = cv2.goodFeaturesToTrack(cv2.cvtColor(im_obj, cv2.COLOR_BGR2GRAY), max_points, 0.01, 1)
        img_feature = np.int0(img_feature)

        features.append(img_feature.reshape([img_feature.shape[0], 2]))

        if features[objects].shape[0] > max_num_features:
            max_num_features = features[objects].shape[0]
    x = np.zeros([max_num_features, num_objects], dtype='int32')
    y = np.zeros([max_num_features, num_objects], dtype='int32')
    for objects in range(num_objects):
        x[0 : features[objects].shape[0], objects] = features[objects][:, 1] + bbox[objects, 0, 0]
        y[0 : features[objects].shape[0], objects] = features[objects][:, 0] + bbox[objects, 0, 1]
        img[x[:, objects], y[:, objects], :] = 255

    return x, y