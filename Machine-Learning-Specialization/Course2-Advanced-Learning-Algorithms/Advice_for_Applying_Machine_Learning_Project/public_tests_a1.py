import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np

def test_eval_mse(target):
    y_hat = np.array([2.4, 4.2], dtype=np.float64)
    y_tmp = np.array([2.3, 4.1], dtype=np.float64)
    result = target(y_hat, y_tmp)
    
    # Use np.isclose and ensure result is cast to float for comparison
    assert np.isclose(float(result), 0.005, atol=1e-6), f"Wrong value. Expected 0.005, got {result}"
    
    y_hat = np.array([3.] * 10)
    y_tmp = np.array([3.] * 10)
    result = target(y_hat, y_tmp)
    assert np.isclose(float(result), 0.), f"Wrong value. Expected 0.0, but got {result}"
    
    y_hat = np.array([3.])
    y_tmp = np.array([0.])
    result = target(y_hat, y_tmp)
    assert np.isclose(float(result), 4.5), f"Wrong value. Expected 4.5, but got {result}."
    
    y_hat = np.array([3.] * 5)
    y_tmp = np.array([2.] * 5)
    result = target(y_hat, y_tmp)
    assert np.isclose(float(result), 0.5), f"Wrong value. Expected 0.5, but got {result}."
    
    print("\033[92m All tests passed.")

def test_eval_cat_err(target):
    y_hat = np.array([1, 0, 1, 1, 1, 0])
    y_tmp = np.array([0, 1, 0, 0, 0, 1])
    result = target(y_hat, y_tmp)
    assert not np.isclose(float(result), 6.), "Wrong value. Did you divide by m?"
    
    y_hat = np.array([1, 2, 0])
    y_tmp = np.array([1, 2, 3])
    result = target(y_hat, y_tmp)
    assert np.isclose(float(result), 1./3., atol=1e-6), f"Wrong value. Expected 0.333, but got {result}"
    
    y_hat = np.array([[1], [2], [0], [3]])
    y_tmp = np.array([[1], [2], [1], [3]])
    res_tmp = target(y_hat, y_tmp)
    # Check if it's a scalar (compatible with modern numpy/tf)
    assert np.isscalar(res_tmp) or res_tmp.shape == (), f"Output must be a scalar but got {type(res_tmp)}"
    
    print("\033[92m All tests passed.")

def model_test(target, classes, input_size):
    # Ensure model is built to check weights/layers
    if not target.built:
        target.build(input_shape=(None, input_size))
        
    expected_lr = 0.01
    assert len(target.layers) == 3, f"Expected 3 layers, got {len(target.layers)}"
    
    # In modern Keras, activation can be a function or a string
    # We normalize to compare names
    expected = [[Dense, [None, 120], "relu"],
                [Dense, [None, 40], "relu"],
                [Dense, [None, classes], "linear"]]

    for i, layer in enumerate(target.layers):
        assert isinstance(layer, expected[i][0]), f"Layer {i} type mismatch"
        
        # Use output.shape.as_list() which is more reliable across TF versions
        # or build the shape from the layer's units
        actual_shape = layer.get_config().get('units')
        expected_units = expected[i][1][1] # Extracts the unit count (e.g., 120)
        
        assert actual_shape == expected_units, \
            f"Wrong number of units in layer {i}. Expected {expected_units} but got {actual_shape}"
        
        # Modern Keras activation check
        # We check the __name__ of the function or the string in the config
        act_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
        
        # Clean up string if it's formatted like <function relu at 0x...>
        if "relu" in act_name.lower():
            act_comparison = "relu"
        elif "linear" in act_name.lower():
            act_comparison = "linear"
        else:
            act_comparison = act_name
            
        assert act_comparison == expected[i][2], \
            f"Layer {i} activation mismatch. Expected {expected[i][2]} but got {act_comparison}"
        
    assert isinstance(target.loss, SparseCategoricalCrossentropy)
    # Corrected learning rate access for modern Adam
    lr = target.optimizer.learning_rate
    if hasattr(lr, 'numpy'): lr = lr.numpy()
    
    assert np.isclose(lr, expected_lr, atol=1e-8), f"Expected LR {expected_lr}, got {lr}"
    assert target.loss.get_config()['from_logits'], "Set from_logits=True in loss function"

    print("\033[92mAll tests passed!")
def model_s_test(target, classes, input_size):
    if not target.built:
        target.build(input_shape=(None, input_size))
    
    expected_lr = 0.01
    assert len(target.layers) == 2, f"Wrong number of layers. Expected 2 but got {len(target.layers)}"
    
    expected = [[Dense, 6, "relu"],
                [Dense, classes, "linear"]]

    for i, layer in enumerate(target.layers):
        assert isinstance(layer, expected[i][0]), f"Layer {i} type mismatch"
        units = layer.get_config().get('units')
        assert units == expected[i][1], f"Layer {i} units mismatch. Expected {expected[i][1]}, got {units}"
        
        act = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
        assert expected[i][2] in act.lower(), f"Layer {i} activation mismatch"

    # Learning rate check
    lr = target.optimizer.learning_rate
    if hasattr(lr, 'numpy'): lr = lr.numpy()
    
    assert np.isclose(float(lr), expected_lr, atol=1e-8), f"Expected LR {expected_lr}, got {lr}"
    assert target.loss.get_config()['from_logits'], "Set from_logits=True in loss function"
    print("\033[92mAll tests passed!")

def model_r_test(target, classes, input_size):
    if not target.built:
        target.build(input_shape=(None, input_size))
        
    expected_lr = 0.01
    assert len(target.layers) == 3, f"Expected 3 layers, got {len(target.layers)}"
    
    # [LayerType, units, activation, (regularizer_type, factor)]
    expected = [[Dense, 120, "relu", 0.1],
                [Dense, 40, "relu", 0.1],
                [Dense, classes, "linear", None]]

    for i, layer in enumerate(target.layers):
        assert isinstance(layer, expected[i][0]), f"Layer {i} type mismatch"
        assert layer.get_config().get('units') == expected[i][1], f"Layer {i} units mismatch"
        
        # Regularization check
        if expected[i][3] is not None:
            reg = layer.kernel_regularizer
            assert reg is not None, f"Layer {i} should have L2 regularization"
            # Get the l2 value (works for both legacy and Keras 3)
            l2_val = reg.get_config().get('l2', 0.0)
            assert np.isclose(l2_val, expected[i][3]), f"Layer {i} expected L2 of {expected[i][3]}, got {l2_val}"
        else:
            assert layer.kernel_regularizer is None, f"Layer {i} should not have regularization"

    lr = target.optimizer.learning_rate
    if hasattr(lr, 'numpy'): lr = lr.numpy()
    
    assert np.isclose(float(lr), expected_lr, atol=1e-8)
    assert target.loss.get_config()['from_logits'], "Set from_logits=True"
    print("\033[92mAll tests passed!")
# Apply similar logic to model_s_test and model_r_test...