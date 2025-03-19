import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
maker = cv2.aruco.generateImageMarker(aruco_dict, 29, 400)

cv2.imwrite("test_7x7_1000_29.jpg",maker)