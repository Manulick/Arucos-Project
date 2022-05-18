import cv2
import numpy as np

parameter = cv2.aruco.DetectorParameters_create()

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    esquinas, ids, candidatos_malos = cv2.aruco.detectMarkers(gray, dictionary, parameters = parameter)

    if np.all(ids != None):
        aruco = cv2.aruco.drawDetectedMarkers(frame, esquinas)

        c1 = (esquinas[0][0][0][0], esquinas[0][0][0][1])
        c2 = (esquinas[0][0][1][0], esquinas[0][0][1][1])
        c3 = (esquinas[0][0][2][0], esquinas[0][0][2][1])
        c4 = (esquinas[0][0][3][0], esquinas[0][0][3][1])

        copy = frame

        image = cv2.imread("imagen1.jpg")

        tamaño = image.shape

        aruco_points = np.array([c1,c2,c3,c4])

        image_points = np.array([
            [0,0],
            [tamaño[1] - 1, 0],
            [tamaño[1] - 1,tamaño[1] - 1],
            [0,tamaño[1] - 1]
        ],dtype = float) 

        h, state = cv2.findHomography(image_points,aruco_points)

        perspective = cv2.warpPerspective(image, h, (copy.shape[1], copy.shape [0]))

        cv2.fillConvexPoly(copy, aruco_points.astype(int), 0, 16)
        copy = copy + perspective
        cv2.imshow("realidad aumentada", copy)
    else:
        cv2.imshow("realidad aumentada", frame)

    k = cv2.waitKey(1)

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


