"""Simple MLP model for testing quantization."""

from flax import linen as nn


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for testing.
    
    Attributes:
        dhidden: Hidden layer dimension
        dout: Output dimension
    """
    dhidden: int
    dout: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dhidden, use_bias=False)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dout, use_bias=False)(x)
        return x

