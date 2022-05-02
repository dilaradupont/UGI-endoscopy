import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
import glob

CLASSES = ['other1', 'pylorus', 'z-line', 'retroflex-stomach']
PATH = '/home/dilara/UGI-endoscopy/testing-data'

all_paths = []

def main():
    for landmark in CLASSES:
        other_path = os.path.join(PATH, landmark, '*')
        all_paths.append(sorted(glob.glob(other_path)))

    for path in all_paths:
        file_name = path[:len(path)-4]
        file_name = file_name[:len(file_name)-4]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rows, cols, dim = image.shape

        # flipping images upside down and leftright
        image_fliplr = np.fliplr(image)
        image_flipud = np.flipud(image)
        plt.imsave(f'{path}_image_flipud.jpg', image_flipud)
        plt.imsave(f'{path}_image_fliplr.jpg', image_fliplr)

        # rotating image
        rotated_img = skimage.transform.rotate(image, 90)
        plt.imsave(f'{path}_rotated_img.jpg', rotated_img)

        # shear transform on x-axis
        M = np.float32([[1, 0.7, 0],
                        [0, 1  , 0],
                        [0, 0  , 1]])             
        sheared_img = cv2.warpPerspective(image,M,(int(cols*1.5),int(rows*1.5)))
        plt.imsave(f'{path}_sheared_img.jpg', sheared_img)

        ## zoom out
        M = np.float32([[1.5, 0  , 0],
                        [0,   1.5, 0],
                        [0,   0,   1]])
        scaledin_img = cv2.warpPerspective(image,M,(cols,rows))
        plt.imsave(f'{path}_scaledin_img.jpg', scaledin_img)

        ##  zoom in
        M = np.float32([[0.8, 0  , 0],
                        [0,   0.8, 0],
                        [0,   0,   1]])
        scaledout_img = cv2.warpPerspective(image,M,(cols,rows))
        plt.imsave(f'{path}_scaledout_img.jpg', scaledout_img)

if __name__ == '__main__':
    main()