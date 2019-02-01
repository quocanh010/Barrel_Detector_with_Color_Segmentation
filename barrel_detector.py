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
        self.W        = np.load('W.npy')
        self.b        = np.load('b.npy')

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
        r_img[2, :] = img[:, :, 2].flatten()

        predicted_class = self.predict(r_img)
        mask_img = predicted_class.reshape(800, 1200)

        #processing
        contours, hierarchy = cv2.findContours(mask_img.astype(np.uint8), cv2.RETR_EXTERNAL, 2)
        # props = skimage.measure.regionprops(contours_mask)

        # Connect disconnected region
        connected_contours = self.connected_region(contours)

        # computes the bounding box for the contour, and draws it on the frame,
        for contour in connected_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            rect_area = w * h
            extent = float(area) / rect_area
            if aspect_ratio < 0.8:
                mask_img[y:y+h, x:x+w] = 1
        # mask_img[mask_img == 1] = 255
        # cv2.imshow('contour', mask_img.astype(np.uint8))

        return mask_img.astype(np.uint8)

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
        r_img = np.zeros([3, img.shape[0] * img.shape[1]])
        r_img[0, :] = img[:, :, 0].flatten()
        r_img[1, :] = img[:, :, 1].flatten()
        r_img[2, :] = img[:, :, 2].flatten()

        predicted_class = self.predict(r_img)
        binary_mask = predicted_class.reshape(800, 1200)
        contours, hierarchy = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, 2)
        # props = skimage.measure.regionprops(contours_mask)

        #Connect disconnected region
        connected_contours = self.connected_region(contours)
        cv2.drawContours(img, connected_contours, -1, (0, 255, 0), 3)

        # computes the bounding box for the contour, and draws it on the frame,
        boxes = []
        for contour in connected_contours:
            area = cv2.contourArea(contour)
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            rect_area = w * h
            extent = float(area) / rect_area
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255,0, 0), 3)
            if aspect_ratio < 0.9:
                boxes.append([x,y,x+w,y+h])




        return boxes

    def sigmoid(self, z):
        """
            Compute the sigmoid of z

            Arguments:
            z -- A scalar or numpy array of any size.

            Return:
            s -- sigmoid(z)
            """
        s = 1 / (1 + np.exp(-z))
        return s

    def softmax(self, a):
        '''
            Softmax function that will return the probability of the input in
            range of [0, 1]
        '''
        return np.exp(a - a.max(axis=0)) / np.exp(a - a.max(axis=0)).sum(axis=0)

    def softmaxp(self, b, d):
        '''
        :param b:
        :param d:
        :return:
        '''
        return  (d - np.sum(d * self.softmax(b), axis=0)) * self.softmax(b)
    def label2onehot(self, lbl):
        '''
            Convert lbl vector to a one hot code
            Input: Label vector with first axis is number of samples
            Output: One-hot code vector associate with the each label
        '''

        d = np.zeros((lbl.astype(np.int).max() + 1, lbl.size))
        d[lbl.astype(np.int), np.arange(0, lbl.astype(np.int).size)] = 1
        return d.astype(np.int)

    def onehot2label(self, d):
        '''
            Convert onehot vector to an associate label
        '''
        lbl = d.argmax(axis=0)
        return lbl

    def eval_loss(self, y, d):
        '''
        :param d: true abel
        :param y: prediction
        :return:
        '''
        return -(d * np.log(y) + (1-d) * np.log(1-y)).mean()

    def feature_vector(self, label_m_name):
        '''
            Convert training image to feature vector with label
        '''
        k = np.load(label_m_name)
        #x_label = k[:, :, 0:33].reshape(1, 33* 800 * 1200)
        # x_test_label =  k[:, :, 33:46].reshape(1, (46 - 33) * 800 * 1200)
        x_label = np.zeros([1, 46 * 800 * 1200])
        folder = "trainset"
        sample_train = np.zeros([3, 46 * 800 * 1200])
        #sample_test  = np.zeros([3, 13 * 800 * 1200])
        for i in range(1, 46):
            if (i <= 46):
                x_label[0, (i - 1) * 800 * 1200: i * 800 * 1200] = k[:, :, i - 1].flatten()
                img = cv2.imread(os.path.join(folder, str(i) + ".png"))
                sample_train[0, (i - 1) * 800 * 1200: i * 800 * 1200] = img[:, :, 0].flatten()
                sample_train[1, (i - 1) * 800 * 1200: i * 800 * 1200] = img[:, :, 1].flatten()
                sample_train[2, (i - 1) * 800 * 1200: i * 800 * 1200] = img[:, :, 2].flatten()




                #sample_train[:, (i-1) *  800 * 1200 : i * 800 * 1200] = np.reshape(cv2.imread(os.path.join(folder, str(i) + ".png")), (3, 800 * 1200))
            # else:
            #     sample_test[:, (i - 1) * 800 * 1200: i * 800 * 1200] = np.reshape(cv2.imread(os.path.join(folder, str(i) + ".png")), (3, 800 * 1200))
        x_label[x_label == 2] = 1
        x_train = sample_train / 255.
        # x_test  = sample_test / 255.
        # print (str(x_train.shape) + str(x_label.shape))
        return x_train, x_label

    def initilize_parameters(self, dim):
        # Xavier initialization
        self.W = np.random.randn(dim, 1) / np.sqrt((3 + 1.))
        self.b = 0

    def propagate(self, x_train, x_label):


        m = x_train.shape[1]
        # Forward
        A = self.sigmoid(np.dot(self.W.T, x_train) + self.b)  # compute activation
        cost = (-1 / m) * np.sum(x_label * np.log(A) + (1 - x_label) * np.log(1 - A))  # compute cost


        #Compute gradient

        dW = (1 / m) * np.dot(x_train, (A - x_label).T)
        db = (1 / m) * np.sum(A - x_label)
        grads = {"dW": dW, "db": db}

        return grads, cost

    def optimize(self, x_train, x_label, n_iter = 1000, alpha = 0.01):
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
            grads, cost= self.propagate(x_train = x_train, x_label = x_label)
            dW = grads['dW']
            db = grads['db']

            #Update
            self.W = self.W - alpha * dW
            self.b = self.b - alpha * db

            # Record the cos

            if i % 10 == 0:
                costs.append(cost)
                print("Cost after iteration %i: %f" % (i, cost))

        return cost

    def predict(self, x_train):

        m = x_train.shape[1]
        Y_prediction = np.zeros((1, m))
        self.W = self.W.reshape(x_train.shape[0], 1)

        A = self.sigmoid(np.dot(self.W.T, x_train) + self.b)

        for i in range(A.shape[1]):
            if A[0, i] < 0.5:
                Y_prediction[0, i] = 0
            else:
                Y_prediction[0, i] = 1
            pass
        assert (Y_prediction.shape == (1, m))
        return Y_prediction

    def model(self, x_train, x_label,   n_iter = 2000, alpha = 0.01, print_cost = False):
        self.W, self.b = self.initilize_parameters(x_train.shape[0])
        costs = self.optimize( x_train, x_label, n_iter, alpha)
        #Y_prediction_train = predict(x_train)
        #print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - x_label)) * 100))

    def connected_region(self, contours ):
        l = len(contours)
        close_region = np.zeros((l, 1))
        for i, region1, in enumerate(contours):
            count = i
            if i != l - 1:
                for j, region2 in enumerate(contours[i + 1:]):
                    count = count + 1
                    is_close = self.close_region_test(region1, region2)
                    if is_close:
                        close_region[i] = min(close_region[i], close_region[count])
                        close_region[count] = close_region[i]
                    elif close_region[count] == close_region[i]:
                        close_region[count] = i + 1

        connected_contours = []
        for i in range(int(close_region.max()) + 1):
            b = np.where(close_region == i) [0]
            if b.size != 0:
                contour = np.vstack((contours[i] for i in b))
                hull = cv2.convexHull(contour)
                connected_contours.append(hull)

        return connected_contours

    def close_region_test(self, region1, region2):
        r_1, r_2 = region1.shape[0], region2.shape[0]
        for i in range(r_1):
            for j in range(r_2):
                dist = np.linalg.norm(region1[i] - region2[j])
                if abs(dist) < 25:
                    return True
                elif i == (r_1 - 1) and (j == r_2 -1):
                    return False




if __name__ == '__main__':
    #folder = "trainset"
    my_detector = BarrelDetector()
    #img = cv2.imread(os.path.join(folder, '5.png'))
    # mask_img = my_detector.segment_image(img)
    # mask_img[mask_img == 1] = 255
    # cv2.imshow('image', mask_img.astype(np.uint8))
    #
    # x_train, x_label = my_detector.feature_vector('label_matrix.npy')
    # my_detector.initilize_parameters(x_train.shape[0])
    # my_detector.optimize(x_train = x_train, x_label = x_label)
    # a = 1
    folder = "trainset"
    # for filename in os.listdir(folder):
    #   # read one test image
    #   img = cv2.imread(os.path.join(folder, filename))
    #   cv2.imshow('image', img)
    #   cv2.waitKey(0)
    #   cv2.destroyAllWindows()

    img = cv2.imread(os.path.join(folder, '5.png'))
    # Display results:
    # (1) Segmented images
    # mask_img = my_detector.segment_image(img)
    # cv2.imshow('image', mask_img.astype(np.uint8))
    # (2) Barrel bounding box
    boxes = my_detector.get_bounding_box(img)
    # The autograder checks your answers to the functions segment_image() and get_bounding_box()
    # Make sure your code runs as expected on the testset before submitting to Gradescope

