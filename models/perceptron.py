"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # Randomly initialize W matrix (self.n_class x # X_train.shape[1]) with rows w_c
        self.w = np.random.rand(self.n_class, X_train.shape[1])

        # print("OG Self.w: ", self.w)
        # print("X_train[0]: ", X_train[0])

        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]): #this iterates through every sample; may want to do mini-batch for efficiency
                # Matrix Multiplication of W and X_train[i] to obtain the response vector
                W_x = np.dot(self.w, X_train[i].T) # (n_class x 1)
                # print("W_x: ", np.amax(W_x))
                # print("W_x: ", W_x)


                # Subtracting label from response vector
                subLoss = W_x - W_x[y_train[i]] # n_class x 1
                # print("subLoss: ", subLoss)

                lossMax = np.maximum(subLoss, 0) # returns a vector of 0s and w_c.x_i > w_yi.x_i
                # print("lossMax: ", lossMax)
                # Summing the hinge losses for all incorrect classes
                # When c = y_i, the maximum will be 0, leaving our result unaffected
                # loss = np.sum(lossMax)

                # Update self.w
                # y_train[i] corresponds to the row of self.w

                # print("lossMax[0]: ", lossMax.shape[0])
                for L in range(lossMax.shape[0]):
                  if lossMax[L] > 0:
                    # print("Label: ", y_train[i])
                    # print("Prediction: ", np.amax(W_x))


                    self.w[y_train[i]] += self.lr * X_train[i]
                    self.w[L] -= self.lr * X_train[i]
                    # print("self.w: ", self.w)

            self.lr = self.lr/self.epochs
                # Vectorize Above For Loop
                # lossMax[lossMax > 0]


        # print("This is our weights: ", self.w)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        pred = np.zeros(X_test.shape[0])
        # print("self.w: ", self.w[0])
        # print("X_test[0]: ", X_test[0])
        # print("w dot X_test: ", np.dot(self.w, X_test[0].T))

        for i in range(X_test.shape[0]):
          predicted = np.argmax(np.dot(self.w, X_test[i].T))
          # print("predicted: ", predicted)
          pred[i] = int(predicted)
        print("Our Pred: ", pred)

        return pred.astype(int)
