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
        self.W        = None
        self.b        = None

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
        r_img = np.zeros([3, img.shape[0] * img.shape[1]])
        r_img[0, :] = img[:, :, 0].flatten()
        r_img[1, :] = img[:, :, 1].flatten()
        r_img[2, :] = img[:, :, 1].flatten()
        r_img = r_img / 255.
        scores = np.dot(self.W, r_img) + self.b
        predicted_class = np.argmax(scores, axis=0)
        mask_img = predicted_class.reshape(800, 1200)

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

    def softmax(self, a):
        '''
            Softmax function that will return the probability of the input in
            range of [0, 1]
        '''
        return np.exp(a - a.max(axis=0)) / np.exp(a - a.max(axis=0)).sum(axis=0)

    def label2onehot(self, lbl):
        '''
            Convert lbl vector to a one hot code
            Input: Label vector with first axis is number of samples
            Output: One-hot code vector associate with the each label
        '''
        d = np.zeros((lbl.astype(np.int).max() + 1, lbl.astype(np.int).size))
        d[lbl.astype(np.int), np.arange(0, lbl.astype(np.int).size)] = 1
        return d.astype(int)

    def onehot2label(self, d):
        '''
            Convert onehot vector to an associate label
        '''
        lbl = d.argmax(axis=0)
        return lbl

    def feature_vector(self, label_m_name):
        '''
            Convert training image to feature vector with label
        '''
        k = np.load(label_m_name)
        #x_label = k[:, :, 0:33].reshape(1, 33* 800 * 1200)
        # x_test_label =  k[:, :, 33:46].reshape(1, (46 - 33) * 800 * 1200)
        x_label = np.zeros([1, 33 * 800 * 1200])

        folder = "trainset"
        sample_train = np.zeros([3, 33 * 800 * 1200])
        #sample_test  = np.zeros([3, 13 * 800 * 1200])
        for i in range(1, 46):
            if (i <= 33):
                x_label[0, (i-1) * 800 * 1200 : i * 800 * 1200] = k[:, :, i-1].flatten()
                img = cv2.imread(os.path.join(folder, str(i) + ".png"))
                sample_train[0:, (i - 1) * 800 * 1200: i * 800 * 1200] = img[:, :, 0].flatten()
                sample_train[1:, (i - 1) * 800 * 1200: i * 800 * 1200] = img[:, :, 1].flatten()
                sample_train[2:, (i - 1) * 800 * 1200: i * 800 * 1200] = img[:, :, 2].flatten()




                #sample_train[:, (i-1) *  800 * 1200 : i * 800 * 1200] = np.reshape(cv2.imread(os.path.join(folder, str(i) + ".png")), (3, 800 * 1200))
            # else:
            #     sample_test[:, (i - 1) * 800 * 1200: i * 800 * 1200] = np.reshape(cv2.imread(os.path.join(folder, str(i) + ".png")), (3, 800 * 1200))

        x_train = sample_train / 255.
        # x_test  = sample_test / 255.
        # print (str(x_train.shape) + str(x_label.shape))
        return x_train, x_label

    def initilize_parameters(self, dim):
        # Xavier initialization
        self.W = np.random.randn(3, 3) / np.sqrt(3. + 1)
        self.b = np.zeros([3, 1])

    def propagate(self, x_train, x_label):


        m = x_train.shape[1]
        # Forward
        reg = 1e-3
        A = self.softmax(np.dot(self.W, x_train) + self.b)
        loss = np.sum(-np.log(A[x_label, range(m)])) / m
        reg_loss = 0.5 * reg * np.sum(self.W * self.W)
        total_loss = reg_loss + loss

        # Backward
        dscores = A
        dscores[x_label, range(m)] -= 1
        dscores /= m

        #Compute gradient
        dW = np.dot(x_train, dscores.T)
        db = np.sum(dscores, axis=1, keepdims=True)

        dW += reg * self.W
        grads = {"dW": dW, "db": db}
        return grads, total_loss

    def optimize(self, x_train, x_label, n_iter = 100, alpha = 0.2):
        '''
        :param n_iter: number of iteration
        :param alpha: learning rate
        :param print_cost: weather to print cost or not
        :return:
        :grads: dW, db
        :costs: total cost
        '''
        costs = []
        for i in range(n_iter):
            grads, cost = self.propagate(x_train = x_train, x_label = x_label)
            dW = grads['dW']
            db = grads['db']

            #Update
            self.W = self.W - alpha * dW
            self.b = self.b - alpha * db
            # Record the costs
            if i % 10 == 0:
                costs.append(cost)
                # Print the cost every 100 training iterations
            if i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))


        return  costs


if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    x_train, x_label = my_detector.feature_vector('label_matrix.npy')
    one_hot_lable = my_detector.label2onehot(x_label)
    my_detector.initilize_parameters(x_train.shape[0])
    my_detector.optimize(x_train = x_train, x_label = one_hot_lable)
    a = 1
    folder = "trainset"
    # for filename in os.listdir(folder):
    #   # read one test image
    #   img = cv2.imread(os.path.join(folder, filename))
    #   cv2.imshow('image', img)
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()

    img = cv2.imread(os.path.join(folder, '1.png'))
    # Display results:
    # (1) Segmented images
    mask_img = my_detector.segment_image(img)
    mask_img[mask_img == 1] = 255
    cv2.imshow('image', mask_img.astype(np.uint8))
    # (2) Barrel bounding box
    #    boxes = my_detector.get_bounding_box(img)
    # The autograder checks your answers to the functions segment_image() and get_bounding_box()
    # Make sure your code runs as expected on the testset before submitting to Gradescope

