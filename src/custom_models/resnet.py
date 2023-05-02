"""Functions for the resnet model."""


def apply_top_layers(x):
    """Apply the top layers of the resnet model."""
    # resnet just applies a global average pooling,
    # but this is already included when instantiating the base model
    return x
