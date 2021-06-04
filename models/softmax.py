"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        grad_Delta = np.zeros(self.w.shape)
        for i in range(X_train.shape[1]):
            score_per_class_vec = np.zeros(self.n_class)
            for c in range(self.n_class):
                score_per_class_vec[c] = np.dot(self.w[c, :], X_train[:, i])

            score_per_class_vec -= np.max(score_per_class_vec)
            temp = np.exp(score_per_class_vec) / np.sum(np.exp(score_per_class_vec))
            for c in range(self.n_class):
                grad_Delta[c, :] += temp[c] * X_train[:, i]
                
            grad_Delta[y_train[i], :] -= X_train[:, i] 
        return (grad_Delta /X_train.shape[1]) + (self.reg_const * self.w)





    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # initialize weight matrix
        if self.w is None: 
          self.w = 0.00001 * np.random.randn(self.n_class, X_train.shape[1])

        # Solve for W using SGD
        for epoch in range(self.epochs):          
          grad_Delta = self.calc_gradient(X_train.T, y_train)         
          self.w -= self.lr*grad_Delta
          
        


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
        return np.argmax(np.dot(self.w,X_test.T),axis=0)
