from glob import glob                                                           
import cv2 
import os
#change to *.jpeg to conver jpeg to jpg
pngs = glob('darknet/cov_data/cov_images/*.png')

for j in pngs:
    print(j)
    img = cv2.imread(j)
    #change -3 to -4 to covert extension of size 4
    cv2.imwrite(j[:-3] + 'jpg', img)
    os.remove(j)