import numpy as np

# Least squares regression

def compute_mse(y, tx, w):
    """ Compute the loss by mse. """
    e = y - np.dot(tx, w)
    return np.dot(e.T, e) / (2*len(y))

def compute_gradient_mse(y, tx, w):
    """ Compute the gradient of mse. """
    e = y - np.dot(tx, w)
    return -np.dot(tx.T, e) / len(y)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Least squares regression using gradient descent. """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_mse(y, tx, w)
        w -= gamma * grad
        loss = compute_mse(y, tx, w)
        # log info
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
    return w, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset. """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Least squares regression using stochastic gradient descent. """
    batch_size = 256
    w = initial_w
    for n_iter, (minibatch_y, minibatch_tx) in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        grad = compute_gradient_mse(minibatch_y, minibatch_tx, w)
        w -= gamma * grad
        loss = compute_mse(y, tx, w)
        # log info
        #print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
         #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def least_squares(y, tx):
    """ Least squares regression using direct approach. """
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    return w, loss

# Ridge regression

def compute_loss_ridge_regression(y, tx, w, lambda_):
    """ Compute the loss for ridge regression. """
    e = y - np.dot(tx, w)
    return np.dot(e.T, e) / (2*len(y)) + lambda_*np.dot(w.T, w)

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using direct approach. """
    N, D = tx.shape
    w = np.linalg.solve(np.dot(tx.T, tx) + 2*N*lambda_*np.eye(D), np.dot(tx.T, y))
    loss = compute_loss_ridge_regression(y, tx, w, lambda_)
    return w, loss

# Logistic regression

def sigmoid(t):
    """ Apply sigmoid function on t. """
    return 1 / (1 + np.exp(-t))

def compute_loss_lr(y, tx, w):
    """ Compute the cost by negative log likelihood for logistic regression. """
    p = sigmoid(np.dot(tx, w))
    return -(np.dot(y.T, np.log(p)) + np.dot((1-y).T, np.log(1-p)))

def compute_gradient_lr(y, tx, w):
    """ Compute the gradient of logistic regression loss. """
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent. """
    w = initial_w
    for iter in range(max_iters):
        grad = compute_gradient_lr(y, tx, w)
        loss = compute_loss_lr(y, tx, w)
        w -= gamma * grad
        # log
        #print("Current iteration={i}, loss={l}".format(i=iter, l=loss/len(y)))
    return w, loss

def compute_loss_reg_lr(y, tx, w, lambda_):
    """ Compute the cost by negative log likelihood for logistic regression. """
    p = sigmoid(np.dot(tx, w))
    return -(np.dot(y.T, np.log(p)) + np.dot((1-y).T, np.log(1-p))) + lambda_*np.dot(w.T, w)

def compute_gradient_reg_lr(y, tx, w, lambda_):
    """ Compute the gradient of logistic regression loss. """
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y) + 2*lambda_*w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent. """
    w = initial_w
    for iter in range(max_iters):
        grad = compute_gradient_reg_lr(y, tx, w, lambda_)
        w -= gamma * grad
        loss = compute_loss_reg_lr(y, tx, w, lambda_)
        # log
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss/len(y)))
    return w, loss

def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using stochastic gradient descent. """
    w = initial_w
    batch_size = 128
    n_iter = 0
    nb_batches = int(len(tx) / batch_size)
    nb_passes = int(np.ceil(max_iters / nb_batches))
    for _ in range(nb_passes):
        for _, (minibatch_y, minibatch_tx) in enumerate(batch_iter(y, tx, batch_size, min(nb_batches, max_iters - n_iter))):
            grad = compute_gradient_reg_lr(minibatch_y, minibatch_tx, w, lambda_)
            w -= gamma * grad
            loss = compute_loss_reg_lr(minibatch_y, minibatch_tx, w, lambda_)
            # log
            #print("Current iteration={}/{}, loss={}".format(n_iter+1, max_iters, loss/batch_size))
            n_iter += 1
    return w, loss