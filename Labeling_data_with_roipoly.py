import logging
import os, cv2, numpy
import numpy as np
from matplotlib import pyplot as plt
from roipoly import RoiPoly

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)


def rgb2gray(rgb):
    '''
    :param rgb:
    :return: gray image
    '''
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

label_matrix = np.load("label_matrix.npy")
folder = 'trainset'
count = 2

#Loop though image and draw ROI
for i in range(1, 47):
    # Create image
    masking_matrix = np.zeros([800, 1200])
    rgb_img = cv2.imread(os.path.join(folder, str(i) + ".png"))
    img = rgb2gray(rgb_img)
    # Show the image
    fig = plt.figure()
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    plt.title("left click: line segment         right click: close region")
    plt.show(block=False)

    # Let user draw first ROI
    roi1 = RoiPoly(color='r', fig=fig)

    # Show the image with the first ROI
    fig = plt.figure()
    #plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    roi1.display_roi()
    plt.title('draw second ROI')
    plt.show(block=False)

    # Let user draw second ROI
    roi2 = RoiPoly(color='b', fig=fig)

    # Show the image with both ROIs and their mean values
    plt.imshow(img, interpolation='nearest', cmap="Greys")
    plt.colorbar()
    [x.display_roi() for x in [roi1, roi2]]
    [x.display_mean(img) for x in [roi1, roi2]]
    plt.title('The two ROIs')
    plt.show()

    # Show ROI masks
    plt.imshow(roi1.get_mask(img) + roi2.get_mask(img),
               interpolation='nearest', cmap="Greys")
    plt.title('ROI masks of the two ROIs')
    plt.show()
    masking_matrix[roi1.get_mask(img)] = 1
    masking_matrix[roi2.get_mask(img)] = 2
    label_matrix[:, :, i-1] = masking_matrix
    np.save('label_matrix.npy', label_matrix)
    #count = count + 1
