import cv2
import os
new_width = 224
new_height = 224
path1 = "../Images/Flower/"
for j in os.listdir(os.path.expanduser(path1)):
    path2 = path1+j+"/"
    for i in os.listdir(os.path.expanduser(path2)):
        path3 = path2+i
        img = cv2.imread(path3)
        img_resized = cv2.resize(src=img, dsize=(new_width, new_height))
        cv2.imwrite(path3, img_resized)

