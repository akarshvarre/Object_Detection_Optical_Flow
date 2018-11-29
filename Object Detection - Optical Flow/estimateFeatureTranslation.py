
import numpy as np
from scipy import interpolate, linalg

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    iterations = 8
    window_size = 9
    newX = np.copy(startX).astype(np.double)
    newY = np.copy(startY).astype(np.double)
    num_feature = 0
    for x, y in zip(startX.astype(np.double), startY.astype(np.double)):
        img1_interp = interpolate.RectBivariateSpline(np.arange(img1.shape[0]), np.arange(img1.shape[1]), img1)
        img2_interp = interpolate.RectBivariateSpline(np.arange(img2.shape[0]), np.arange(img2.shape[1]), img2)
        Ix_interp = interpolate.RectBivariateSpline(np.arange(Ix.shape[0]), np.arange(Ix.shape[1]), Ix)
        Iy_interp = interpolate.RectBivariateSpline(np.arange(Iy.shape[0]), np.arange(Iy.shape[1]), Iy)
        
        window_coords_x = np.arange(x - int(window_size / 2), x - int(window_size / 2) + window_size)
        window_coords_y = np.arange(y - int(window_size / 2), y - int(window_size / 2) + window_size)
        window_y, window_x = np.meshgrid(window_coords_y, window_coords_x)
        Wx = Ix_interp.ev(window_x, window_y)
        Wy = Iy_interp.ev(window_x, window_y)
        I0 = img1_interp.ev(window_x, window_y)
        A = np.array([Wx.ravel(), Wy.ravel()]).T
        u, v = 0, 0
        q_x, q_y = window_x.astype(np.double), window_y.astype(np.double)
        for t in range(iterations):
            q_x = q_x + u
            q_y = q_y + v
            I1 = img2_interp.ev(q_x, q_y)
            It = (I0 - I1).ravel().T
            A = np.array([Ix_interp.ev(q_x, q_y).ravel(), Iy_interp.ev(q_x, q_y).ravel()]).T
            det = linalg.det(np.dot(A.T, A))
            if det == 0:
                print('Found singular matrix !')
            solution = linalg.solve(np.dot(A.T, A), np.dot(A.T, It))
            u, v = solution[0], solution[1]
            newX[num_feature] = newX[num_feature] + u
            newY[num_feature] = newY[num_feature] + v
        num_feature += 1
    return newX, newY