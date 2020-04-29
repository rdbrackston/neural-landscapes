import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt


class FunctionApproximator(Model):
    def __init__(self, function, n_dims=1, loss_object=None, optimizer=None):
        super(FunctionApproximator, self).__init__()
        self.d1 = Dense(128, activation='tanh', input_shape=(n_dims,))
        self.d2 = Dense(128, activation='tanh')
        self.d3 = Dense(1)

        if loss_object is None:
            self.loss_object = tf.keras.losses.MeanSquaredError()
        else:
            self.loss_object = loss_object

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = optimizer

        self.func = function

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

    @tf.function
    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            predictions = self(x_train)
            loss = self.loss_object(y_train, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def train(self, x_ranges, n_x=1000, n_epochs=50):
        x_train = np.linspace(x_ranges[0], x_ranges[1], n_x)
        y_train = [self.func(x) for x in x_train]

        x_train = tf.reshape(x_train, [n_x, 1])
        x_train = tf.cast(x_train, tf.float32)
        y_train = tf.reshape(y_train, [n_x, 1])
        y_train = tf.cast(y_train, tf.float32)

        for epoch in range(n_epochs):
            self.train_step(x_train, y_train)

        return x_train, self(x_train)


class Landscape(Model):
    def __init__(self, dyn_func, n_dims=1, optimizer=None):
        super(Landscape, self).__init__()
        self.d1 = Dense(128, activation='tanh', input_shape=(n_dims,))
        self.d2 = Dense(128, activation='tanh')
        self.d3 = Dense(1, use_bias=False)  # No bias since offset of the landscape is irrelevant

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = optimizer

        self.dyn_func = dyn_func

        # Initialise the Lagrange multiplier as a trainable variable
        self.lamda = tf.Variable(1., trainable=True, name='Lambda')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

    # Grad U obtained via automatic differentiation
    def grad_U(self, x):
        with tf.GradientTape() as tU:
            tU.watch(x)
            y = self(x)
        return tU.gradient(y, x)

    # Returns the negative of the Lagrangian which aims to maximise gU s.t. dot(gU,fU)=0
    def loss_func(self, x_train):
        fU = self.dyn_func(x_train)
        gU = self.grad_U(x_train)
        return self.lamda * tf.keras.backend.sum(tf.abs(gU * fU + gU ** 2)) - \
               tf.keras.backend.sum(tf.abs(gU))

    @tf.function
    def train_step(self, x_train):
        with tf.GradientTape() as tape:
            loss = self.loss_func(x_train)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def train(self, x_ranges, n_x=1000, n_epochs=50):

        x_train = np.linspace(x_ranges[0], x_ranges[1], n_x)
        x_train = tf.cast(x_train, tf.float32)
        x_train = tf.reshape(x_train, [n_x, 1])

        # Training loop
        for epoch in range(n_epochs):
            self.train_step(x_train)

        return x_train, self(x_train)


if __name__ == '__main__':

    # Test the function approximator
    def func(x):
        return np.exp(0.2 * x) * np.sin(x)
    model_approximator = FunctionApproximator(func, 1)

    _, y_before = model_approximator.train((-10, 10), n_epochs=0)
    x_train, y_after = model_approximator.train((-10, 10), n_epochs=500)
    y_train = [func(x) for x in x_train]

    plt.plot(x_train, y_train, marker='.', linewidth=0, label="True function")
    plt.plot(x_train, y_before, label="Before")
    plt.plot(x_train, y_after, label="After")
    plt.legend()
    plt.show()

    # Test the landscape estimation
    def f(x):
        return x - x ** 3 + 0.1

    def true_U(x):
        return -0.5*x**2 + 0.25*x**4 - 0.1*x
    landscape_model = Landscape(f)

    _, y_before = landscape_model.train((-2.0, 2.0), n_epochs=0)
    x_train, y_after = landscape_model.train((-2.0, 2.0), n_epochs=500)
    y_true = [true_U(x) for x in x_train]

    plt.plot(x_train, y_true, marker='.', linewidth=0, label="True landscape")
    plt.plot(x_train, y_before, label="Before")
    plt.plot(x_train, y_after, label="After")
    plt.legend()
    plt.show()
