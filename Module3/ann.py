# Todd Bartoszkiewicz
# CSC510: Foundations of Artificial Intelligence
# Module 3: Critical Thinking
#
# Hand-Made Shallow ANN in Python
#
# Using your research and resources, write a basic 2-layer Artificial Neural Network utilizing static backpropagation
# using Numpy in Python. Your neural network can perform a basic function, such as guessing the next number in a series.
# Using the activation function of your choice to calculate the predicted output ŷ, known as the feedforward function,
# and updating the weights and biases through gradient descent (backpropagation) based on your choice of a basic loss
# function.
#
# Your ANN should include the following features:
#
# An input layer that takes input data as a matrix receives and passes it on,
# A hidden layer,
# An output layer, and
# Weights between the layers.
# Also, your ANN should demonstrate it can perform the following functions:
#
# Multiply the input by a set of weights (via matrix multiplication);
# Apply deliberate activation function for every hidden layer;
# Return an output;
# Calculate error by taking the difference from the desired output and the predicted output,
# giving us the gradient descent to provide our loss function;
# Apply loss function to weights; and
# Repeat this process no less than 1,000 times to train the ANN.
#
import numpy as np
import tensorflow as tf


class ArtificialNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initialize artificial neural network with weights
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weight1 = tf.Variable(np.random.randn(input_size, hidden_size) * 0.1, dtype=tf.float32)
        self.bias1 = tf.Variable(np.zeros((1, hidden_size)), dtype=tf.float32)
        self.weight2 = tf.Variable(np.random.randn(hidden_size, output_size) * 0.1, dtype=tf.float32)
        self.bias2 = tf.Variable(np.zeros((1, output_size)), dtype=tf.float32)

        # Activation function
        self.hidden_activation = tf.nn.relu
        self.output_activation = lambda x: x

    def feedforward_propagation(self, x):
        """
        Forward propagation through the artificial neural network.
        :param x: input data matrix (TensorFlow tensor)
        :return: prediction output (TensorFlow tensor)
        """
        # Layer 1 (input to hidden)
        matrix1 = tf.matmul(x, self.weight1) + self.bias1
        ha1 = self.hidden_activation(matrix1)

        # Layer 2 (hidden to output)
        matrix2 = tf.matmul(ha1, self.weight2) + self.bias2

        return self.output_activation(matrix2)

    def train_step(self, x, y):
        """
        Perform one training step (forward pass + backward pass + weight update).
        :param x: input data matrix (TensorFlow tensor)
        :param y: target output matrix (TensorFlow tensor)
        :return: loss value (float)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            y_prediction = self.feedforward_propagation(x)

            # Compute loss
            loss = tf.reduce_mean(tf.square(y_prediction - y))

        # Compute gradients
        gradients = tape.gradient(loss, [self.weight1, self.bias1, self.weight2, self.bias2])

        # Update weights and biases using gradient descent
        self.weight1.assign_sub(self.learning_rate * gradients[0])
        self.bias1.assign_sub(self.learning_rate * gradients[1])
        self.weight2.assign_sub(self.learning_rate * gradients[2])
        self.bias2.assign_sub(self.learning_rate * gradients[3])

        return loss.numpy

    def train(self, x, y, epochs=1000):
        """
        Train the artificial neural network
        :param x: input data matrix (NumPy array)
        :param y: target output data matrix (NumPy array)
        :param epochs: number of training iterations
        :return: training_step_losses
        """

        # Convert to TensorFlow tensors
        x_tf = tf.constant(x, dtype=tf.float32)
        y_tf = tf.constant(y, dtype=tf.float32)

        training_step_losses = []

        print(f"Training artificial neural network for {epochs} epochs.")

        for epoch in range(epochs):
            # Perform training step
            training_step_loss = self.train_step(x_tf, y_tf)
            training_step_losses.append(training_step_loss)

        return training_step_losses

    def predict(self, x):
        """
        Make predictions using the trained artificial neural network.
        :param x: input data matrix (NumPy array)
        :return: predicted output (NumPy array)
        """
        x_tf = tf.constant(x, dtype=tf.float32)
        return self.feedforward_propagation(x_tf).numpy()


def create_training_data():
    x = []
    y = []

    for j in range(1, 20):
        x.append([j, j+1, j+2])
        y.append([j+3])

    return np.array(x, dtype=float), np.array(y, dtype=float)


if __name__ == "__main__":
    print(f"Please enter a list of the numbers that you would like the next number predicted for in this format:")
    print(f"1,2,3")
    test_input_string = input()
    test_inputs = []
    elements = [int(z) for z in test_input_string.split(',')]
    test_inputs.append(elements)

    # Create sample data for training.
    # Use a simple pattern when given 3 consecutive numbers, predict the 4th
    # For example: [1, 2, 3] -> 4
    x_array, y_array = create_training_data()

    print(f"Training data shape: x_array={x_array.shape}, y_array={y_array.shape}")
    print("Sample training data:")
    for i in range(5):
        print(f"Input: {x_array[i]}, Target: {y_array[i][0]}")

    # Normalize data
    x_min = x_array.min()
    x_max = x_array.max()
    y_min = y_array.min()
    y_max = y_array.max()

    x_normalized = (x_array - x_min) / (x_max - x_min)
    y_normalized = (y_array - y_min) / (y_max - y_min)

    # Create and train artificial neural network
    ann = ArtificialNeuralNetwork(input_size=3, hidden_size=8, output_size=1, learning_rate=0.5)

    # Train for at least 1,000 times
    losses = ann.train(x_normalized, y_normalized, epochs=2000)

    # Test the artificial neural network
    print(f"Testing the trained artificial neural network:")
    # test_inputs = [
    #    [4, 5, 6],  # Should predict 7
    #    [7, 8, 9],  # Should predict 10
    #    [10, 11, 12],  # Should predict 13
    #    [13, 14, 15]  # Should predict 16
    # ]
    test_inputs_array = np.array(test_inputs, dtype=float)

    # Normalize the test inputs
    test_inputs_normalized = (test_inputs_array - x_min) / (x_max - x_min)

    # Make predictions
    predictions_normalized = ann.predict(test_inputs_normalized)

    # Denormalize predictions
    predictions = predictions_normalized * (y_max - y_min) + y_min

    # Display results
    for i, test_input in enumerate(test_inputs):
        expected = test_input[-1] + 1
        predicted = predictions[i][0]
        error = abs(predicted - expected)
        print(f"Input: {test_input} -> Predicted: {predicted:.4f}, Expected: {expected}, Error: {error:.4f}")
