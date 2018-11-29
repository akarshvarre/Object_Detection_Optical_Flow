def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    import numpy as np
    import skimage.transform as tf
    import random 

    [rows,nobj] = np.asarray(startXs.shape)
    temp_Xs = []
    temp_Ys = []
    r = []

    homography = []
    second_box = np.copy(bbox.astype(np.double))
    count = 0
    for i in range(nobj):
        inliers_count = []
        inliers_firstX = []
        inliers_firstY = []
        inliers_secondX = []
        inliers_secondY = []
        box_corners = np.matrix.transpose(bbox[i, :, :])
        firstX = startXs[:, i]
        firstY = startYs[:, i]
        firstX = firstX[firstX!=-1]
        firstY = firstY[firstY!=-1]
        secondX = newXs[:, i]
        secondY = newYs[:, i]
        secondX = secondX[secondX!=-1]
        secondY = secondY[secondY!=-1]
        for k in range(500):
            for j in range(4):
                r.append(random.randint(0,np.shape(firstX)[0])-1)
            four_firstX = [firstX[x] for x in r]
            four_firstY = [firstY[x] for x in r]
            four_secondX = [secondX[x] for x in r]
            four_secondY = [secondY[x] for x in r]
            H_matrix = tf.SimilarityTransform()
            four_first = np.matrix.transpose(np.vstack((four_firstX,four_firstY)))
            four_second = np.matrix.transpose(np.vstack((four_secondX,four_secondY)))
            H_matrix.estimate(four_first, four_second)
            H = H_matrix.params
            homog_first = np.vstack((firstX, firstY, np.ones(rows)))
            new_second = np.dot(H, homog_first)
            diff_vector = np.vstack((secondX, secondY)) - new_second[0:2, :]

            diff_vector=diff_vector*diff_vector#find the square of the difference
            squared_dist=diff_vector[0,:]+diff_vector[1,:]
    
            temp_firstX = firstX[squared_dist <= 16]
            temp_firstY = firstY[squared_dist <= 16]
            temp_secondX = secondX[squared_dist <= 16]
            temp_secondY = secondY[squared_dist <= 16]
            inliers_count.append(np.shape(firstX)[0])
            homography.append(H)
            inliers_firstX.append(temp_firstX)
            inliers_firstY.append(temp_firstY)
            inliers_secondX.append(temp_secondX)
            inliers_secondY.append(temp_secondY)
        max_count_index = inliers_count.index(max(inliers_count))
        firstX = inliers_firstX[max_count_index]
        firstY = inliers_firstY[max_count_index]
        secondX = inliers_secondX[max_count_index]
        secondY = inliers_secondY[max_count_index]

        H_matrix.estimate(np.matrix.transpose(np.vstack((firstX, firstY))), np.matrix.transpose(np.vstack((secondX, secondY))))
        count = max(count, len(secondX))
        temp_Xs.append(secondX)
        temp_Ys.append(secondY)
        homo_box_corners = np.vstack((box_corners, np.ones(2)))
        H = H_matrix.params
        second_homo_box_corners = np.dot(H, homo_box_corners)
        second_box[i, :, :] = np.matrix.transpose(second_homo_box_corners[0:2, :])

    Xs = np.ones([count, nobj])*-1
    Ys = np.ones([count, nobj])*-1
    for m in range(nobj):
        Xs[0:len(temp_Xs[m]), m] = temp_Xs[m]
        Ys[0:len(temp_Ys[m]), m] = temp_Ys[m]
    return Xs, Ys, second_box