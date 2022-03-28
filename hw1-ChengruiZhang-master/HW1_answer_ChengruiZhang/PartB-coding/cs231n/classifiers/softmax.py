import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_class = W.shape[1]

  output = np.exp(np.dot(X, W)) # softmax
  output_sum = np.sum(output, axis=1)
  regular = np.sum(np.sum(np.square(W)))
  probability = np.zeros([num_train, num_class])
  for i in range(num_train): # 计算softmax，计算概率值的矩阵，计算softmax loss，之后根据标签的矩阵计算权重矩阵，并带入至当前的dW中。
    truth_num = y[i]
    p = output[i, :]/output_sum[i]
    loss = loss - np.log(p[truth_num])
    probability[i, :] = p[:]
    probability[i, truth_num] = probability[i, truth_num] - 1
  loss = loss / num_train + 0.5 * reg * regular
  dW = X.T.dot(probability)/num_train + reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.    #
  # Store the loss in loss and the gradient in dW. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                          #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  output = np.exp(np.dot(X, W)) # 计算softmax，计算概率值的矩阵，计算softmax loss，之后根据标签的矩阵计算权重矩阵，并带入至当前的dW中。
  output_sum = np.sum(output, axis=1)
  regular = np.sum(np.sum(np.square(W)))
  output_sum = output_sum.reshape((num_train,1))
  probability = output / output_sum
  loss = -np.log(probability[np.arange(num_train), y]).sum()
  loss = loss / num_train + 0.5 * reg * regular
  probability[np.arange(num_train), y] = probability[np.arange(num_train), y] - 1
  dW = X.T.dot(probability)
  dW = dW/num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

