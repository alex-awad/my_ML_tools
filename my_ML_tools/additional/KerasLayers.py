from keras import regularizers # For adding L1 and L2 regularization
from keras import layers

def get_layer_list(input_shape):
    """Returns list of pre-defined layers for Keras Sequential models."""
    # Layers for different models
    layer_list_0 = [
        # Input layer
        layers.Dense(
            16,
            activation="relu",
            input_shape=(input_shape,)
        ),
        # First hidden layer
        layers.Dense(
            32,
            activation="relu"
        ),
        # Second hidden layer
        layers.Dense(
            16,
            activation="relu"
        ),
        # Output layer
        layers.Dense(1)
    ]

    layer_list_1 = [
        # Input layer
        layers.Dense(
            32,
            activation="relu",
            input_shape=(input_shape,)
        ),
        # First hidden layer
        layers.Dense(
            32,
            activation="relu"
        ),
        # Second hidden layer
        layers.Dense(
            16,
            activation="relu"
        ),
        # Output layer
        layers.Dense(1)
    ]

    layer_list_2 = [
        # Input layer
        layers.Dense(
            32,
            activation="relu",
            input_shape=(input_shape,)
        ),
        # First hidden layer
        layers.Dense(
            32,
            activation="relu"
        ),
        # Second hidden layer
        layers.Dense(
            32,
            activation="relu"
        ),
        # Output layer
        layers.Dense(1)
    ]

    layer_list_3 = [
        # Input layer
        layers.Dense(
            32,
            activation="relu",
            input_shape=(input_shape,)
        ),
        # hidden layer
        layers.Dense(
            16,
            activation="relu"
        ),
        # Output layer
        layers.Dense(1)
    ]

    layer_list_4 = [
        # Input layer
        layers.Dense(
            4,
            activation="relu",
            input_shape=(input_shape,)
        ),
        # First hidden layer
        layers.Dense(
            4,
            activation="relu"
        ),
        # Output layer
        layers.Dense(1)
    ]

    layer_list_5 = [
        # Input layer
        layers.Dense(
            32,
            kernel_regularizer=regularizers.l2(0.001),
            activation="relu",
            input_shape=(input_shape,)
        ),
        # First hidden layer
        layers.Dense(
            32,
            kernel_regularizer=regularizers.l2(0.001),
            activation="relu"
        ),
        # Second hidden layer
        layers.Dense(
            16,
            kernel_regularizer=regularizers.l2(0.001),
            activation="relu"
        ),
        # Output layer
        layers.Dense(1)
    ]

    layer_list_6 = [
        # Input layer
        layers.Dense(
            32,
            kernel_regularizer=regularizers.l1(0.0001),
            activation="relu",
            input_shape=(input_shape,)
        ),
        # First hidden layer
        layers.Dense(
            32,
            kernel_regularizer=regularizers.l1(0.0001),
            activation="relu"
        ),
        # Second hidden layer
        layers.Dense(
            16,
            kernel_regularizer=regularizers.l1(0.0001),
            activation="relu"
        ),
        # Output layer
        layers.Dense(1)
    ]


    layer_list_7 = [
        # Input layer
        layers.Dense(
            32,
            activation="relu",
            input_shape=(input_shape,)
        ),
        # Dropout
        layers.Dropout(0.5),
        # First hidden layer
        layers.Dense(
            32,
            activation="relu"
        ),
        # Dropout
        layers.Dropout(0.5),
        # Second hidden layer
        layers.Dense(
            16,
            activation="relu"
        ),
        # Dropout
        layers.Dropout(0.5),
        # Output layer
        layers.Dense(1)
    ]


    layers_list = [
        layer_list_0,
        layer_list_1,
        layer_list_2,
        layer_list_3,
        layer_list_4,
        layer_list_5,
        layer_list_6,
        layer_list_7,
    ]

    return layers_list