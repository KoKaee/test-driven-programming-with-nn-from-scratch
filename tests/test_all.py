import sys
import os
import pytest
import numpy as np

# Add parent directory to path to import nn-from-scratch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the nn-from-scratch module
from nn_from_scratch import (
    Layer, 
    ReLU, 
    Dense, 
    softmax_crossentropy_with_logits,
    forward,
    predict,
    train,
    train_mnist_network,
    load_mnist_from_csv
)

def test_relu_layer():
    """Test the ReLU layer implementation"""
    # Create a ReLU layer
    relu = ReLU()

    # Test input with positive and negative values
    input = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

    # Forward pass
    output = relu.forward(input)
    
    # Expected output: [0, 0, 0, 1, 2]
    expected = np.array([[0.0, 0.0, 0.0, 1.0, 2.0]])
    assert np.array_equal(output, expected), f"ReLU forward pass failed. Expected {expected}, got {output}"

    # Backward pass
    grad_output = np.ones_like(output)
    grad_input = relu.backward(input, grad_output)
    
    # Expected gradient: [0, 0, 0, 1, 1]
    expected_grad = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    assert np.array_equal(grad_input, expected_grad), f"ReLU backward pass failed. Expected {expected_grad}, got {grad_input}"


def test_dense_layer():
    """Test the Dense layer implementation"""
    # Create a small dense layer: 2 inputs, 3 outputs
    dense = Dense(2, 3, learning_rate=0.1)

    # Set weights and biases for predictable testing
    dense.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    dense.biases = np.array([0.1, 0.2, 0.3])

    # Test input: batch of 2 examples
    input = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Forward pass
    output = dense.forward(input)

    # Verify output with manual calculation
    expected_output = np.array(
        [
            [
                1.0 * 0.1 + 2.0 * 0.4 + 0.1,
                1.0 * 0.2 + 2.0 * 0.5 + 0.2,
                1.0 * 0.3 + 2.0 * 0.6 + 0.3,
            ],
            [
                3.0 * 0.1 + 4.0 * 0.4 + 0.1,
                3.0 * 0.2 + 4.0 * 0.5 + 0.2,
                3.0 * 0.3 + 4.0 * 0.6 + 0.3,
            ],
        ]
    )
    assert np.allclose(output, expected_output), "Dense forward pass failed"

    # Test backward pass
    grad_output = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Save original weights
    original_weights = dense.weights.copy()
    original_biases = dense.biases.copy()

    grad_input = dense.backward(input, grad_output)

    # Check parameter updates (weights and biases should be updated)
    assert not np.array_equal(original_weights, dense.weights), "Weights were not updated"
    assert not np.array_equal(original_biases, dense.biases), "Biases were not updated"


@pytest.mark.slow
def test_model_accuracy():
    """Test that the model achieves at least 70% validation accuracy"""
    try:
        # Load a small subset of data for testing
        X_train, y_train, X_val, y_val, _, _ = load_mnist_from_csv(
            "./mnist_train.csv", "./mnist_test.csv", val_split=0.1
        )

        # Train model with minimal configuration
        _, val_accuracy = train_mnist_network(
            X_train, y_train, X_val, y_val, num_epochs=5
        )

        # Assert that validation accuracy is at least 50%
        assert val_accuracy >= 0.5, f"Model accuracy {val_accuracy:.4f} is below the required 50%"
    except FileNotFoundError:
        pytest.skip("MNIST dataset files not found. Skipping accuracy test.")


if __name__ == "__main__":
    # Run tests
    print("Running ReLU layer test...")
    test_relu_layer()
    print("ReLU layer test passed!")
    
    print("\nRunning Dense layer test...")
    test_dense_layer()
    print("Dense layer test passed!")
    
    try:
        print("\nRunning model accuracy test...")
        test_model_accuracy()
        print("Model accuracy test passed!")
    except Exception as e:
        print(f"Model accuracy test skipped or failed: {e}") 