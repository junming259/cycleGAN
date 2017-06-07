'''
This file defines Dataset class for read and process train and test dataset. If
you want to use this code, you need to specify the path of training and test
dataset.

'''


import os
import numpy as np

from random import shuffle






BATCH_SIZE = 1
SCALE_SIZE = 256
TRAIN_DATA_A_PATH = 'datasets/apple2orange/trainA/'
TRAIN_DATA_B_PATH = 'datasets/apple2orange/trainB/'

TEST_DATA_A_PATH = 'datasets/apple2orange/testA/'
TEST_DATA_B_PATH = 'datasets/apple2orange/testB/'







class Dataset:
    """
    Dataset is a class, which seperate dataset into training set and test set.
    Beside, it provides function to get batch of data
    """
    # the position for next batch
    batch_offset_A = 0
    batch_offset_B = 0
    # the number of epochs
    epochs_completed_A = 0
    epochs_completed_B = 0


    def __init__(self, train_A_path=TRAIN_DATA_A_PATH,
                    train_B_path=TRAIN_DATA_B_PATH,
                    test_A_path=TEST_DATA_A_PATH,
                    test_B_path=TEST_DATA_B_PATH,
                    batch_size=1):
        """
        Initial function to initialize Dataset class. It seperates dataset into
        training set and test set.

        : param images: entire set of images
        : param labels: entire set of labels
        : param batch_size: the size of batch data
        """

        self.train_A, self.train_B = self.read_data(train_A_path, train_B_path)
        self.test_A, self.test_B = self.read_data(test_A_path, test_B_path)

        print('finish reading all data...')

        self.batch_size = batch_size



    def read_resize_image(self, image_path):
        '''
        This function is used to read and process image and label. First resize
        image and label into SCALE_SIZE, and then crop the resized image into
        CROP_SIZE.
        : param image_path: the path of image
        : return: two ndarray with dtype uint8
        '''

        # resize image
        image = Image.open(image_path)
        if SCALE_SIZE == 128:
            image = image.resize((SCALE_SIZE, SCALE_SIZE), Image.BILINEAR)
        image = np.array(image)

        return image.astype(np.uint8)



    def read_data(self, A_path, B_path):
        '''
        This function reads images and labels from target path. Because the dtype
        of original images and labels is uint8, I need to convert them into float32
        and scale the value within range -1 to 1.
        '''
        data_A = []
        data_B = []

        A_name = os.listdir(A_path)
        A_name = [item for item in A_name if item.endswith('jpg')]
        B_name = os.listdir(B_path)
        B_name = [item for item in B_name if item.endswith('jpg')]

        shuffle(A_name)
        shuffle(B_name)
        # A_name.sort()
        # B_name.sort()

        for item in A_name:
            path = A_path + item
            A = self.read_resize_image(path)
            if A.shape[-1] == 3:
                data_A.append(A)

        for item in B_name:
            path = B_path + item
            B = self.read_resize_image(path)
            if B.shape[-1] == 3:
                data_B.append(B)

        # process images from uint8 to float32 within range (-1, 1)
        data_A = np.array(data_A)
        data_B = np.array(data_B)

        data_A = data_A / 255.0
        data_A = 2 * data_A - 1.0

        data_B = data_B / 255.0
        data_B = 2 * data_B - 1.0

        return data_A.astype(np.float32), data_B.astype(np.float32)



    def next_train_A_batch(self):
        """
        Helper function to retrive next batch of images from self.train_A, which
        is keep in the memory. If entire images set is train out, then shuffle
        dataset. This code is adopted from internet.
        : param batch_size: int, the size of batch
        : return: next batch of images and labels
        """

        start = self.batch_offset_A
        self.batch_offset_A += self.batch_size
        if self.batch_offset_A > self.train_A.shape[0]:
            # Finished epoch
            self.epochs_completed_A += 1
            print("****************** TrainA Epochs completed: " + str(self.epochs_completed_A) + "******************")

            # Shuffle the data
            index = list(range(self.train_A.shape[0]))
            shuffle(index)
            self.train_A = self.train_A[index]

            # Start next epoch
            start = 0
            self.batch_offset_A = self.batch_size

        end = self.batch_offset_A
        images = self.train_A[start:end]

        return images



    def next_train_B_batch(self):
        """
        Helper function to retrive next batch of images from self.train_A, which
        is keep in the memory. If entire images set is train out, then shuffle
        dataset. This code is adopted from internet.
        : param batch_size: int, the size of batch
        : return: next batch of images and labels
        """

        start = self.batch_offset_B
        self.batch_offset_B += self.batch_size
        if self.batch_offset_B > self.train_B.shape[0]:
            # Finished epoch
            self.epochs_completed_B += 1
            print("****************** TrainB Epochs completed: " + str(self.epochs_completed_B) + "******************")

            # Shuffle the data
            index = list(range(self.train_B.shape[0]))
            shuffle(index)
            self.train_B = self.train_B[index]

            # Start next epoch
            start = 0
            self.batch_offset_B = self.batch_size

        end = self.batch_offset_B
        images = self.train_B[start:end]

        return images



    def next_batch(self):
        A = self.next_train_A_batch()
        B = self.next_train_B_batch()

        return A, B



    def get_random_train_batch(self, batch_size=1):
        """
        Helper function to retrive random size of batch images and labels.
        : param batch_size: the size of batch
        """

        index_A = np.random.randint(0, self.train_A.shape[0], size=batch_size).tolist()
        index_B = np.random.randint(0, self.train_B.shape[0], size=batch_size).tolist()

        A = self.train_A[index_A]
        B = self.train_B[index_B]

        return A, B



    def get_random_test_batch(self, batch_size=1):
        """
        Helper function to retrive random size of batch images and labels from
        test dataset.
        : param batch_size: the size of batch
        """
        index_A = np.random.randint(0, self.test_A.shape[0], size=batch_size).tolist()
        index_B = np.random.randint(0, self.test_B.shape[0], size=batch_size).tolist()

        A = self.test_A[index_A]
        B = self.test_B[index_B]

        return A, B
