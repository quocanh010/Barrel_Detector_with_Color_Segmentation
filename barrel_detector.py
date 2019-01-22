'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
import numpy as np
from skimage.measure import label, regionprops


class BarrelDetector():
    def __init__(self):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
        '''
        self.x_train  = None
        self.x_label  = None
        self.w        = None
        self.b        = None
        self.feature_vector("label_matrix.npy")


    def segment_image(self, img):
        '''
            Calculate the segmented image using a classifier
            eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        '''
        # YOUR CODE HERE
        return mask_img

    def get_bounding_box(self, img):
        '''
            Find the bounding box of the blue barrel
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        # YOUR CODE HERE
        return boxes

    def softmax(self):
        '''
            Softmax function that will return the probability of the input in
            range of [0, 1]
        '''
        return np.exp(a - a.max(axis=0)) / np.exp(a - a.max(axis=0)).sum(axis=0)

    def label2onehot(lbl):
        '''
            Convert lbl vector to a one hot code
            Input: Label vector with first axis is number of samples
            Output: One-hot code vector associate with the each label
        '''
        d = np.zeros((lbl.size, lbl.max() + 1))
        d[np.arange(0, lbl.size), lbl] = 1
        return d

    def onehot2label(d):
        '''
            Convert onehot vector to an associate label
        '''
        lbl = d.argmax(axis=1)
        return lbl

    def feature_vector(self, label_m_name):
        '''
            Convert training image to feature vector with label
        '''
        z = np.zeros([46 * 800 * 1200, 3])
        z[:, 0] = np.load(label_m_name).reshape([46 * 800 * 1200])
        z[:, 1] = np.load(label_m_name).reshape([46 * 800 * 1200])
        z[:, 2] = np.load(label_m_name).reshape([46 * 800 * 1200])
        self.x_label = z.flatten() / 255.
        folder = "trainset"
        sample = np.zeros([46 * 800 * 1200, 3])
        for i in range(1, 47):

            sample[(i-1) *  800 * 1200 : i * 800 * 1200, :] = np.reshape(cv2.imread(os.path.join(folder, str(i) + ".png")), (800 * 1200, 3))

        self.x_train = sample.flatten() / 255.
        print (str(self.x_train.shape) + str (self.x_label.shape))

    def initilize_parameters(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0

    def propagate(self):

        m = self.x_train.shape[0]
        A = softmax(np.dot(self.w.T,self.x_train) + b)
        cost =


        



if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    a = 1
    # for filename in os.listdir(folder):
    #   # read one test image
    #   img = cv2.imread(os.path.join(folder, filename))
    #   cv2.imshow('image', img)
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()


    # Display results:
    # (1) Segmented images
    #    mask_img = my_detector.segment_image(img)
    # (2) Barrel bounding box
    #    boxes = my_detector.get_bounding_box(img)
    # The autograder checks your answers to the functions segment_image() and get_bounding_box()
    # Make sure your code runs as expected on the testset before submitting to Gradescope

