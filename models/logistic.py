"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me

        return np.exp(-np.logaddexp(0, -z))
    
    def convertLabels(self, z: np.ndarray):
      """Convert y Labels from 0 to -1.

      Parameters:
          z: a numpy array of shape (N,)

      """
      for i in range(z.shape[0]):
        if z[i] == 0:
          z[i] = -1

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = np.random.rand(X_train.shape[1])
        for i in range(y_train.shape[0]):
          if y_train[i] == 0:
            y_train[i] = -1

        # Loop through each sample
        for epoch in range(self.epochs):
          for i in range(X_train.shape[0]):
            # Initializing ndarray to be passed through sigmoid()
            # np.dot(self.w, X_train[i]): self.w, X_train[i]
            sig = -y_train[i] * np.dot(self.w, X_train[i])
            sigm = self.sigmoid(sig)

            # Update for every X_train[i] no matter what (property of logistic regression)
            self.w += self.lr * np.dot(sigm, X_train[i]) * y_train[i]
            
          # Learning rate decay (manually tweaked)
          self.lr += self.lr/self.epochs
        
        # print("our self.w: ", self.w)

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

        # Initialize predicted labels vector
        y_test = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
          # Assign sign(wT.x_i) to y_test 
          y = np.sign(np.dot(self.w, X_test[i]))
          y_test[i] = y

        return y_test.astype(int)
