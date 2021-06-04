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
        # TODO: implement me
        # Gradient with respect to weights: n_class x Dimensions matrix
        grad_w = np.zeros(self.w.shape)

        # # calc_Scores: N x self.n_class matrix containing  
        # calc_Scores = np.dot(X_train,self.w) 

        # # Tuple of arrays:
        # score_index = np.arange(calc_Scores.shape[0]), y_train

        # # N x 1 vector containing each samples true class
        # true_Class = calc_Scores[score_index]


        # Loop through all N samples in batch
        for i in range(X_train.shape[0]):
          # Feature Vector: n_classxD x Dx1 -> n_class x 1
          scores = self.w @ X_train[i].T
          # print("scores: ", scores)
          max_score = np.amax(scores)
          # print("max score: ", max_score)
          scores = scores - max_score
          # print("scores-max: ", scores)

          expo = np.exp(scores)
          # print("expo: ", expo)
          # Takes the exponentiated sum
          sum_exp_scores = np.sum(expo)
          # print("sum_exp_scores", sum_exp_scores)


          # print("scores[y_train[i]]: ", scores[y_train[i]])
          # Gets the exponentiated score of the correct class: a scalar 
          get_Yscore = np.exp(scores[y_train[i]])
          # print("get_Yscore: ", get_Yscore)

          # Probability of correct class
          Py = get_Yscore/sum_exp_scores
          # print("Py: ", Py)

          # Gradient w.r.t w_yi: 1xD vector 
          y_grad = -(X_train[i]) + Py*X_train[i]
          grad_w[y_train[i]] = y_grad.T
          
          # print("y_grad: ", y_grad)
          # print("grad_w[y_train[i]]: ", grad_w[y_train[i]])

          

          for c in range(self.n_class):
            if c != y_train[i]:
              # Gets the exponentiated score of the incorrect class: a scalar 
              get_Cscore = np.exp(scores[c])
              # print("scores[c]: ", scores[c])
              # print("get_Cscore: ", get_Cscore)
              # Probability of incorrect class
              Pc = get_Cscore/sum_exp_scores
              # Gradient w.r.t w_c: 1xD vector
              c_grad = Pc * X_train[i]
              grad_w[c] = c_grad.T


        return grad_w

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # Weights: n_class x Dimensions matrix
        self.w = np.random.rand(self.n_class,X_train.shape[1])
        
        for epoch in range(self.epochs):
          grad_Delta = self.calc_gradient(X_train, y_train) 
          # print("grad_Delta: ", grad_Delta)
          reg = self.reg_const/X_train.shape[0]
          t = reg * self.w
          self.w -= self.lr* (t + grad_Delta)
          # print("w.shape: ", self.w.shape)
          # print("grad.shape: ", grad_Delta.shape)
          # self.w[y_train] += self.lr * (1-grad_Delta[y_train]) * X_train[y_train]
          # self.w[]

          # for i in range(X_train.shape[0]):
          # # Feature Vector: n_classxD x Dx1 -> n_class x 1
          #   scores = self.w @ X_train[i].T
          #   max_score = np.amax(scores)
          #   scores = scores - max_score

          #   expo = np.exp(scores)
          #   # Takes the exponentiated sum
          #   sum_exp_scores = np.sum(expo)

          #   # Gets the exponentiated score of the correct class: a scalar 
          #   get_Yscore = np.exp(scores[y_train[i]])

          #   # Probability of correct class
          #   Py = get_Yscore/sum_exp_scores

          #   # Gradient w.r.t w_yi: 1xD vector 
          #   y_grad = -(X_train[i]) + Py*X_train[i]
            
          #   self.w[y_train[i]] += self.lr * np.dot((1-y_grad), X_train[i])
            
          #   for c in range(self.n_class):
          #     if c != y_train[i]:
          #       # Gets the exponentiated score of the incorrect class: a scalar 
          #       get_Cscore = np.exp(scores[c])
          #       # Probability of incorrect class
          #       Pc = get_Cscore/sum_exp_scores
          #       # Gradient w.r.t w_c: 1xD vector
          #       c_grad = Pc * X_train[i]
          #       self.w[c] -= self.lr * np.dot(c_grad, X_train[i])
            
          # self.lr = self.lr / self.epochs
        return

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
        return np.argmax(np.dot(X_test,self.w.T),axis=1)
