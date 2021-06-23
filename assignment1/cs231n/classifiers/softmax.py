from builtins import range
import numpy as np
from random import shuffle


# from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X @ W
    scores -= scores.max()

    for i in range(num_train):
        y_score = scores[i][y[i]]
        loss_term = np.exp(y_score) / np.sum(np.exp(scores[i]))
        loss -= np.log(loss_term)

        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (loss_term - 1) * X[i]
                continue

            dW[:, j] += (np.exp(scores[i][j]) / np.sum(np.exp(scores[i]))) * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X @ W
    scores -= scores.max()  # normalization

    y_score = np.sum(scores[np.arange(scores.shape[0]), y])
    scores_sum = np.sum(np.exp(scores), axis=1)
    scores_log_sum = np.sum(np.log(scores_sum))

    loss -= y_score - scores_log_sum
    loss /= num_train
    loss += reg * np.sum(W * W)

    w_j = np.exp(scores) / scores_sum.reshape(-1, 1)
    w_y = np.zeros_like(w_j)
    w_y[np.arange(w_y.shape[0]), y] = 1

    dW = X.T @ (w_j - w_y)
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
