"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me

        dW_matrix = np.zeros(self.w.shape)
        X_train_Size = X_train.shape[0]   
        calc_Scores = np.dot(X_train,self.w) 
        score_index = np.arange(calc_Scores.shape[0]), y_train

        true_Class = calc_Scores[score_index]
        loss_arg = calc_Scores.T - true_Class + 1
        loss_M = np.maximum(0, loss_arg.T)

        temp_index = np.arange(loss_M.shape[0]), y_train
        
        loss_M[temp_index] = 0        
        binarized_loss_M = np.where(loss_M == 0, 0, 1)

        temp_index = np.arange(X_train_Size), y_train
        binarized_loss_M[temp_index] -= np.sum(binarized_loss_M, axis=1)
        dW_matrix = np.dot(X_train.T,binarized_loss_M)
        loss_term = dW_matrix/X_train_Size
        reg_term = self.reg_const*self.w
        dW_matrix = loss_term + reg_term 

        return dW_matrix


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
          self.w = 0.000001 * np.random.randn(X_train.shape[1], self.n_class)

        # Solve for W using SGD
        for epoch in range(self.epochs):          
          grad_Delta = self.calc_gradient(X_train, y_train)         
          self.w -= self.alpha*grad_Delta

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
        return np.argmax(np.dot(X_test,self.w),axis=1)