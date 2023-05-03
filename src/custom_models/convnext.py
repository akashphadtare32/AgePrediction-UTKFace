"""Functions for the convnext model."""


def apply_top_layers(x):
    """Apply the top layers of the convnext model."""
    # convnext has Layer Normalization, but here we
    # directly use the output of the base model
    return x
