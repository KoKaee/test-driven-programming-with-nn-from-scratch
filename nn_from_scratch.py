import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_mnist_from_csv(train_csv_path, test_csv_path, val_split=0.1):
    """
    Load MNIST dataset from CSV files and split into train, validation, and test sets

    Arguments:
    train_csv_path -- path to the training CSV file
    test_csv_path -- path to the test CSV file
    val_split -- fraction of training data to use for validation

    Returns:
    X_train -- training features
    y_train -- training labels
    X_val -- validation features
    y_val -- validation labels
    X_test -- test features
    y_test -- test labels
    """
    print("Loading training data...")
    train_data = pd.read_csv(train_csv_path)

    print("Loading test data...")
    test_data = pd.read_csv(test_csv_path)

    # Extract labels and features
    y_train_full = train_data.iloc[:, 0].values
    X_train_full = (
        train_data.iloc[:, 1:].values / 255.0
    )  # Normalize pixel values to [0,1]

    y_test = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values / 255.0

    # Create validation set from training data
    n_val = int(len(X_train_full) * val_split)

    # Random indices for validation set
    np.random.seed(42)
    val_indices = np.random.choice(len(X_train_full), n_val, replace=False)

    # Create a mask for training indices
    train_mask = np.ones(len(X_train_full), dtype=bool)
    train_mask[val_indices] = False

    # Split into train and validation sets
    X_val = X_train_full[val_indices]
    y_val = y_train_full[val_indices]

    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_from_csv(
    "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
)


class Layer:
    """
    Base class for neural network layers

    Each layer performs two key operations:
    - Forward pass: Compute outputs from inputs
    - Backward pass: Compute gradients and update parameters
    """

    def __init__(self):
        """Initialize layer parameters (if any)"""
        pass

    def forward(self, input):
        """
        Forward pass: Compute outputs from inputs

        Arguments:
        input -- Input data, shape (batch_size, input_dimension)

        Returns:
        output -- Layer output, shape (batch_size, output_dimension)
        """
        # Base implementation (identity function)
        return input

    def backward(self, input, grad_output):
        """
        Backward pass: Compute gradient of loss with respect to inputs

        Arguments:
        input -- Input data from forward pass
        grad_output -- Gradient of loss with respect to layer output

        Returns:
        grad_input -- Gradient of loss with respect to layer input
        """
        # Base implementation (pass gradient unchanged)
        return grad_output


class ReLU(Layer):
    """
    ReLU activation layer

    Forward: f(x) = max(0, x)
    Backward gradient: f'(x) = 1 if x > 0 else 0
    """

    def forward(self, input):
        """Apply ReLU activation function"""
        # TODO: Implement the forward pass for ReLU activation
        # Hint: Use np.maximum to apply the ReLU function element-wise
        return  np.maximum(0, input)

    def backward(self, input, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        # TODO: Implement the backward pass for ReLU activation
        # Hint: ReLU gradient is 1 where input > 0, and 0 elsewhere
        relu_grad =  input > 0 
        return grad_output * relu_grad


class Dense(Layer):
    """
    Fully connected (dense) layer

    Forward: output = input · weights + bias
    """

    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
        Initialize dense layer

        Arguments:
        input_units -- dimension of input
        output_units -- dimension of output
        learning_rate -- learning rate for gradient descent
        """
        self.learning_rate = learning_rate

        # Initialize weights with small random values (Xavier/Glorot initialization)
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(
            2.0 / input_units
        )
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        Forward pass: compute output = input · weights + bias
        """
        # Store input for backward pass
        self.input = input

        # TODO: Implement the forward pass for a dense layer
        # Hint: Use np.dot for matrix multiplication
        return  np.dot(input, self.weights) + self.biases 

    def backward(self, input, grad_output):
        """
        Backward pass: compute gradients and update parameters
        """
        # Gradient of loss w.r.t. weights: input^T · grad_output
        grad_weights = np.dot(input.T, grad_output) / input.shape[0]

        # Gradient of loss w.r.t. biases: sum grad_output over batch dimension
        grad_biases = np.mean(grad_output, axis=0)

        # Gradient of loss w.r.t. input: grad_output · weights^T
        grad_input = np.dot(grad_output, self.weights.T)

        # Update parameters using gradient descent
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input


def softmax_crossentropy_with_logits(logits, labels):
    """
    Compute softmax cross-entropy loss and its gradient

    Arguments:
    logits -- raw model outputs, shape (batch_size, num_classes)
    labels -- true labels, shape (batch_size,)

    Returns:
    loss -- scalar value, cross-entropy loss
    grad -- gradient of loss w.r.t. logits, shape (batch_size, num_classes)
    """
    # Create one-hot vectors from labels
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    # Compute softmax (with numeric stability)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy loss
    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size

    # Gradient of cross-entropy loss w.r.t. logits
    grad = (softmax_probs - one_hot_labels) / batch_size

    return loss, grad


def forward(network, X):
    """
    Perform forward propagation through the network

    Arguments:
    network -- list of layers
    X -- input data

    Returns:
    layer_activations -- list of activations from each layer
    """
    activations = []
    input = X

    # Pass input through each layer
    for layer in network:
        activations.append(layer.forward(input))
        input = activations[-1]  # Output of current layer becomes input to next layer

    return activations


def predict(network, X):
    """Get predictions from the network"""
    # Get the output of the last layer
    logits = forward(network, X)[-1]
    # Return the class with highest score
    return np.argmax(logits, axis=-1)


def train(network, X, y):
    """Train the network on a batch of examples"""
    # Forward pass
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations[:-1]  # Inputs to each layer
    logits = layer_activations[-1]

    # Compute loss and initial gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    # Backward pass (backpropagation)
    grad_output = grad_logits
    for i in range(len(network))[::-1]:  # Reversed order
        layer = network[i]
        layer_input = layer_inputs[i]
        grad_output = layer.backward(layer_input, grad_output)

    return loss


def train_mnist_network(
    X_train, y_train, X_val, y_val, num_epochs=5, batch_size=32, subset_size=10000
):
    """Train a neural network on MNIST dataset"""
    # Initialize the network
    network = [
        Dense(784, 64),  # Input layer -> Hidden layer 1
        ReLU(),  # Activation function
        Dense(64, 32),  # Hidden layer 1 -> Hidden layer 2
        ReLU(),  # Activation function
        Dense(32, 10),  # Hidden layer 2 -> Output layer
    ]

    print("Network architecture:")
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            print(
                f"Layer {i}: Dense ({layer.weights.shape[0]} -> {layer.weights.shape[1]})"
            )
        else:
            print(f"Layer {i}: {layer.__class__.__name__}")

    # Training history
    train_loss_history = []
    val_accuracy_history = []

    # Use a subset for faster training during demo
    if subset_size and subset_size < len(X_train):
        subset_indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = X_train[subset_indices]
        y_train_subset = y_train[subset_indices]
    else:
        X_train_subset = X_train
        y_train_subset = y_train

    print(f"\nTraining on {len(X_train_subset)} examples")

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Shuffle the training data
        indices = np.random.permutation(len(X_train_subset))
        X_shuffled = X_train_subset[indices]
        y_shuffled = y_train_subset[indices]

        # Mini-batch training
        num_batches = (len(X_train_subset) + batch_size - 1) // batch_size
        epoch_losses = []

        for batch in range(num_batches):
            # Extract batch
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, len(X_train_subset))

            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Train on batch
            loss = train(network, X_batch, y_batch)
            epoch_losses.append(loss)
            train_loss_history.append(loss)

            # Print progress
            if batch % 20 == 0:
                print(f"  Batch {batch+1}/{num_batches}, Loss: {loss:.4f}")

        # Evaluate on validation set
        val_subset_size = min(
            1000, len(X_val)
        )  # Use a smaller subset for faster evaluation
        val_indices = np.random.choice(len(X_val), val_subset_size, replace=False)

        val_predictions = predict(network, X_val[val_indices])
        val_accuracy = np.mean(val_predictions == y_val[val_indices])
        val_accuracy_history.append(val_accuracy)

        avg_loss = np.mean(epoch_losses)
        print(
            f"  Epoch complete - Avg Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

    return network, val_accuracy


network, val_accuracy = train_mnist_network(X_train, y_train, X_val, y_val)
assert val_accuracy > 0.65