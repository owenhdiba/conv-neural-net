'''
This module defines classes for constructing a neural network. A base
`Layer` class is defined which has the child classes `ConvLayer`, 
`VeccingLayer`, `Dense Layer` and `SoftmaxLayer`. Create instances of
these layers to match the required architecture of your neural network
model and then add them to an instance of `Neural` with the `add_layer`
method. Finally, train the neural network with `Neural`'s `fit` method.
The history of the loss and error during the training process are
stored as attriubutes in the `Neural` object.
'''
import numpy as np
import random


class Layer():
    '''
    Base class for Layer objects
    '''
    def __init__(self, W_shape, b_shape, std_dev, eta=0.001):
        '''
        Parameters
        ----------

        W_shape : tuple,
            Dimensions of the weight array

        b_shape : tuple,
            Dimensions of the biases.

        std_dev : float,
           Standard deviation of the normal distribution
           from which weights and biases are initialised.
        eta : float,
            Learning rate.
        '''
        # Initialises weights W and biases according to Xavier
        # initialisation
        self.W = np.random.normal(scale=std_dev, size=W_shape)
        self.b = np.random.normal(scale=std_dev, size=b_shape)
        # initialise parameters of ADAM optimisation
        self.mW = np.zeros(W_shape)
        self.vW = np.zeros(W_shape)
        self.mb = np.zeros(b_shape)
        self.vb = np.zeros(b_shape)
        # constant coefficients of ADAM
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 10**(-8)
        self.eta = eta

    def forward(self):
        pass

    def backward(self):
        pass

    def update(self, dW, db, t):
        '''
        Updates weights and Adam parameters of layer

        Parameters
        ----------

        dW : numpy array
            derivative of weights

        db : numpy array
            derivative of biases

        t : int
            iteration of optimisation
        '''
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (dW**2)
        mW = self.mW / (1 - self.beta1**t)
        vW = self.vW / (1 - self.beta2**t)
        self.W -= self.eta * mW / (np.sqrt(vW) + self.epsilon)

        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (db**2)
        mb = self.mb / (1 - self.beta1**t)
        vb = self.vb / (1 - self.beta2**t)
        self.b -= self.eta * mb / (np.sqrt(vb) + self.epsilon)


class VeccingLayer(Layer):
    '''
    intermediate layer between convolutional and
    dense layer
    '''

    def __init__(self, channels_in, dim_in):
        '''
        Parameters
        ----------

        channels_in : int
            Number of input channels to this layer.
        dim_in : int
            The height and width in pixels of the input image to
            this layer.
        '''
        self.channels_in = channels_in
        self.dim_in = dim_in
        self.flattened_dim = channels_in * (dim_in**2)
        return

    def forward(self, Q, batch_size, _):
        return Q.reshape(batch_size, self.flattened_dim)

    def backward(self, dQ, batch_size, _):
        dQ = dQ.reshape(batch_size, self.channels_in,
                        self.dim_in, self.dim_in)
        return dQ


class DenseLayer(Layer):
    '''
    densely connected neural layer
    with ReLU activation
    '''

    def __init__(self, dim_in, dim_out, eta=0.001):
        '''
        Parameters
        ----------

        dim_in : int
            The height and width in pixels of the input image to this 
            layer.

        dim_out : int
            The height and width in pixels of the desired output image 
            fo this layer. 

        eta : float
            Learning rate for Adam optimization.

        '''
        W_shape = (dim_out, dim_in)
        b_shape = (dim_out, 1)
        std_dev = np.sqrt(1 / dim_in)  # for Xavier dist
        super().__init__(W_shape, b_shape, std_dev, eta)

    def forward(self, Q, _, test=False):
        Z = np.matmul(Q, self.W.T) + self.b.T
        # if training (test == False), save Q and Z for
        # use in backprop
        if test is False:
            self.Q = Q
            self.Z = Z
        else:
            pass
        Q = np.maximum(0, Z)  # ReLU
        return Q

    def backward(self, dQ, batch_size, t):
        dZ = dQ * (self.Z > 0)
        dQ = np.matmul(dZ, self.W)
        dW = np.matmul(dZ.T, self.Q) / batch_size
        db = np.mean([dZ], axis=1).T
        self.update(dW, db, t)
        return dQ


class SoftmaxLayer(Layer):
    '''
    densely connected neural layer
    with softmax activation
    '''

    def __init__(self, dim_in, dim_out, eta=0.001):
        '''
        Parameters
        ----------

        dim_in : int
            The height and width in pixels of the input image to this
            layer.

        dim_out : int
            The height and width in pixels of the desired output image 
            fo this layer. 

        eta : float
            Learning rate for Adam optimization.
        '''
        W_shape = (dim_out, dim_in)
        b_shape = (dim_out, 1)
        std_dev = np.sqrt(1 / dim_in)
        super().__init__(W_shape, b_shape, std_dev, eta)

    def forward(self, Q, _, test=False):
        Z = np.matmul(Q, self.W.T) + self.b.T
        if test is False:
            self.Q = Q
            self.Z = Z
        else:
            pass
        P = self.softmax(Z)
        return P

    def forward_fast(self, Q, _):
        Z = np.matmul(Q, self.W.T) + self.b.T
        P = self.softmax(Z)
        return P

    def backward(self, dZ, batch_size, t):
        dQ = np.matmul(dZ, self.W)
        dW = np.matmul(dZ.T, self.Q) / batch_size
        db = np.mean([dZ], axis=1).T
        self.update(dW, db, t)
        return dQ

    @staticmethod
    def softmax(Z):
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        return P


class ConvLayer(Layer):
    '''
    Convolutional layer with ReLU activation
    '''

    def __init__(self, channels_in, channels_out, dim_in, dim_out, dim_W,
                 stride, eta=0.001):
        '''
        Parameters
        ----------
        channels_in : int
            Number of input channels to this layer.

        channels_out : int
            Number of output channels from this layer. 

        dim_in : int
            The height and width in pixels of the input image to this 
            layer.

        dim_out : int
            The height and width in pixels of the desired output image 
            fo this layer. 

        dim_W : int
            The height and width in pixels of the kernel (filter) of 
            the convolution.

        stride : int
            The stride of the convolution operation.

        eta : float
            Learning rate for Adam optimization.

        '''
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_W = dim_W
        self.stride = stride

        # padding parameters of Q_in for forwards prop
        # total padding on width and height of image:
        self.pad_forward = stride * (dim_out - 1) + dim_W - dim_in
        # dimension of image after padding
        self.dim_pad_forward = self.pad_forward + dim_in
        # padding to be added to height of image on top
        # and width of image on the left
        self.top_pad_forward = np.ceil(self.pad_forward / 2).astype(int)
        # padding to be added to height of image on bottom
        # and width of image on the right
        self.bottom_pad_forward = self.pad_forward - self.top_pad_forward

        # padding parameters of dZ for back prop
        self.pad_back = dim_in + dim_W - stride * (dim_out - 1) - 2
        self.dim_pad_back = self.pad_back + stride * (dim_out - 1) + 1
        self.bottom_pad_back = np.ceil(self.pad_back / 2).astype(int)
        self.top_pad_back = self.pad_back - self.bottom_pad_back

        # filters and rotated filters
        self.W_shape = (channels_in, channels_out, dim_W, dim_W)
        self.W_shape_flat = (channels_in, channels_out, dim_W**2)
        std_dev = np.sqrt(1. / (channels_in * dim_W**2))
        super().__init__(self.W_shape_flat, channels_out, std_dev, eta)

        # indices in padded and dilated dZ that are set to
        # original dZ values
        self.idxs_dZ = self.top_pad_back + \
            self.stride * np.arange(0, self.dim_out)

        # these indices are used by calculate_X functions
        # they define the indexes of the windows of the image that are
        # multiplied by the kernel in the cross correlation operation
        self.idxs_calc_Z = (np.arange(0,
                                      self.dim_pad_forward - self.dim_W + 1,
                                      self.stride)[:, None]
                            + np.arange(self.dim_W))

        self.idxs_calc_dQ = (np.arange(0,
                                       self.dim_pad_back - self.dim_W + 1
                                       )[:, None]
                             + np.arange(self.dim_W))

        self.idxs_calc_dW = np.arange(0, self.dim_W)[
            :, None] + self.stride * np.arange(0, self.dim_out)

    def forward(self, Q_in, batch_size, test=False):
        Q_in = np.pad(
            Q_in,
            (
                (0, 0), (0, 0),
                (self.top_pad_forward, self.bottom_pad_forward),
                (self.top_pad_forward, self.bottom_pad_forward)
            )
        )
        Z = self.calculate_Z(Q_in, batch_size)
        if test is False:
            self.Q_in = Q_in
            self.Z = Z
        else:
            pass
        Q_out = np.maximum(0, Z)
        return Q_out

    def backward(self, dQ_in, batch_size, t):
        dZ = dQ_in * (self.Z > 0)
        dZ_dilated = np.zeros((batch_size, self.channels_out,
                               self.dim_pad_back,
                               self.dim_pad_back))
        dZ_dilated[np.ix_(np.arange(batch_size),
                          np.arange(self.channels_out),
                          self.idxs_dZ, self.idxs_dZ)] = dZ

        dQ_out = self.calculate_dQ(dZ_dilated, batch_size)
        dW = self.calculate_dW(dZ, batch_size)
        db = dZ.sum((0, 2, 3)) / batch_size
        self.update(dW, db, t)
        return dQ_out

    def calculate_Z(self, Q_in, batch_size):
        sub_arrays = self.partition(Q_in, self.idxs_calc_Z)
        sub_arrays = sub_arrays.reshape(batch_size, self.channels_in,
                                        self.dim_out**2,
                                        self.dim_W**2)
        QccW = np.tensordot(sub_arrays, self.W, axes=([1, 3], [0, 2]))
        Z = QccW + self.b
        return Z.swapaxes(1, 2).reshape(batch_size, self.channels_out,
                                        self.dim_out, self.dim_out)

    def calculate_dQ(self, dZ_dilated, batch_size):
        sub_arrays = self.partition(dZ_dilated, self.idxs_calc_dQ)
        sub_arrays = sub_arrays.reshape(batch_size, self.channels_out,
                                        self.dim_in**2,
                                        self.dim_W**2)

        rot_W = np.rot90(self.W.reshape(*self.W_shape), k=2,
                         axes=(-2, -1)).reshape(*self.W_shape_flat)

        dQ = np.tensordot(sub_arrays, rot_W, axes=(
            [1, 3], [1, 2])).swapaxes(1, 2)
        return dQ.reshape(batch_size, self.channels_in,
                          self.dim_in, self.dim_in)

    def calculate_dW(self, dZ, batch_size):
        sub_arrays = self.partition(self.Q_in, self.idxs_calc_dW)
        sub_arrays = sub_arrays.reshape(batch_size, self.channels_in,
                                        self.dim_W**2,
                                        self.dim_out**2)
        dZ = dZ.reshape(batch_size, self.channels_out, self.dim_out**2)
        dW = np.einsum('ijkl, ivl->jvk', sub_arrays, dZ) / batch_size
        return dW.reshape(self.channels_in, self.channels_out,
                          self.dim_W**2)

    @staticmethod
    def partition(X, idxs):
        '''
        Partitions an array into subarrays which are acted on by the
        filter.

        Parameters
        ----------

        X : array of dimension
            (batch size)*(channels)*(image pixels)*(image pixels)
        idxs : list of length "d"
            each element is a list of indices of pixels to be
            included in the subarray
        Returns
        -------

        array of dimension
        (batch size)*(channels)*("d")*(filter pixels)*(filter pixels)
        '''
        return (X[:, :, idxs]
                .transpose((0, 1, 2, 4, 3))[:, :, :, idxs]
                .transpose((0, 1, 2, 3, 5, 4)))


class Neural():
    '''
    Neural network class

    Attributes
    ----------
    costs : list
        The history of the cost at every iteration of optimization.

    test_costs : list
        The history of the cost of the test dataset calculated every
        `test_period` iterations.

    errors : list
        The history of the error at every iteration of optimization.
    
    test_errors : list
        The history of the error of the test dataset calculated every
        `test_period` iterations.

    Methods
    -------

    add_layer
        add layer to the network.

    fit
        Train the network on testdata.

    '''

    def __init__(self, epochs, num_classes):
        '''
        Parameters
        ----------

        epochs : int
            Number of complete cycles through the training dataset.

        num_classes : int
            Total number of unique classes in the dataset e.g. for 
            MNIST this is 10.
        '''
        self.layers = []
        self.epochs = epochs

        self.costs = []
        self.test_costs = []

        self.errors = []
        self.test_errors = []

        self.c = num_classes
        self.classes = np.arange(self.c)

    def process_y(self, y):
        '''
        one-hot encoding of labels

        Parameters
        ----------

        y : numpy array
            array of categorical labels

        Returns
        -------

        array of dimension (dataset size)*(number of classes)
        '''
        Y = np.array([y == a for a in self.classes]).astype(int).T
        return Y

    def add_layer(self, layer):
        '''
        Add layer to the network. The layers should be added
        from the bottom up i.e. the first layer which is added is the 
        first layer to process the data. The last layer added should 
        be a `SoftmaxLayer`.

        Parameters
        ----------

        layer : Layer
            Layer object.
        '''
        self.layers.append(layer)

    def forward(self, X, batch_size, test=False):
        Q = X
        for layer in self.layers:
            Q = layer.forward(Q, batch_size, test)
        return Q

    def backward(self, dQ, batch_size, t):
        for layer in reversed(self.layers):
            dQ = layer.backward(dQ, batch_size, t)
            '''
            if isinstance(layer, veccingLayer):
                dQ = layer.backward(dQ)
            else:
                dQ = layer.backward(dQ, t)
            '''
        return dQ

    def fit(self, X, y, batch_size=1, test_X=None, test_y=None,
            test_period=None):
        '''
        Train the neural network

        Parameters
        ----------

        X : nd numpy array, 
            The training data. This should have dimensions which
            match the input dimensions of the first layer of the
            network.

        y : 1d numpy array or list, 
            The targets of the training data. This should be an array
            of integers from 0 to `self.num_classes`-1  that index the
            different categories of the data.

        batch_size : int, 
            The maximum number of samples in a batch. If the first
            dimension of X (the number of samples) is divisible by
            batch_size each batch will be of size batch_size, if not,
            the final batch will be the size of the remainder of
            division.

        test_ X : nd numpy array,
            The test (or validation) data.

        test_y : 1d numpy array or list, 
            The targets of the test data.

        test_period : int,
            The number of iterations between calculating the loss and
            error on the test_datset.
        '''
        n_samples = X.shape[0]

        Y = self.process_y(y)

        if test_X is not None:
            n_test_samples = test_X.shape[0]
            test_Y = self.process_y(test_y)

        t = 1  # iteration of optimisation
        for epoch in range(self.epochs):
            # batch cost (itself an average) averaged over all batches
            # in an epoch.
            epoch_cost = 0
            gen = self.yield_idxs(n_samples, batch_size)
            for idxs in gen:
                current_batch_size = len(idxs)
                P = self.forward(X[idxs], current_batch_size)
                batch_cost = self._mean_cost(P, Y[idxs])
                self.costs.append(batch_cost)
                epoch_cost += batch_cost * current_batch_size
                self.errors.append(self._mean_error(P, y[idxs]))
                dZ = P - Y[idxs]
                self.backward(dZ, current_batch_size, t)
                if test_X is not None and t % test_period == 0:
                    test_gen = self.yield_idxs(n_test_samples, batch_size,
                                               test=True)
                    # initialise avg cost and error of whole testset
                    test_cost = 0
                    test_error = 0
                    # use batches to speed up forward prop over testset
                    for test_idxs in test_gen:
                        test_batch_size = len(test_idxs)
                        test_P = self.forward(test_X[test_idxs],
                                              test_batch_size,
                                              test=True)
                        test_cost += self._sum_cost(test_P,
                                                    test_Y[test_idxs])
                        test_error += self._sum_error(test_P,
                                                      test_y[test_idxs])
                    self.test_costs.append(test_cost / n_test_samples)
                    self.test_errors.append(test_error / n_test_samples)
                t += 1
            epoch_cost /= n_samples
            fstring = f"Epoch {epoch} of {self.epochs}: loss = {epoch_cost:.3f}"
            print(fstring)

    def yield_idxs(self, n_samples, batch_size, test=False):
        '''
        Parameters
        ----------

        n_samples : int
            number of images in training or testing dataset

        batch_size : int
            max number of images in a batch

        test : bool
            Is it a training or testing dataset?

        Returns
        -------

        A generator of indices of the dataset to be included in
        each batch. If n_samples is divisible by batch_size each
        batch will be of size batch_size, if not, the final batch will
        be the size of the remainder of division.
        If we have a training dataset (i.e. test is False) then indices
        in each batch are drawn at random from whole dataset.
        '''
        p = list(range(n_samples))
        if test is False:
            random.shuffle(p)
        q = list(np.arange(0, n_samples, batch_size, dtype=int))
        q.append(n_samples)
        for i in range(len(q) - 1):
            yield [p[j] for j in range(q[i], q[i + 1])]

    @ staticmethod
    def _mean_cost(P, Y):
        # average cost over images in batch
        return -np.mean(np.log(P[Y.astype(bool)]), axis=0)

    def _mean_error(self, P, y):
        # average error over images in batch
        predictions = self.classes[np.argmax(P, axis=1)]
        return np.mean(predictions != y, axis=0)

    @ staticmethod
    def _sum_cost(P, Y):
        # sum cost over images in batch
        return -np.sum(np.log(P[Y.astype(bool)]), axis=0)

    def _sum_error(self, P, y):
        # sum error over images in batch
        predictions = self.classes[np.argmax(P, axis=1)]
        return np.sum(predictions != y, axis=0)
