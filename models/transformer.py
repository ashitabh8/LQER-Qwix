"""Simple decoder-only Transformer model for testing quantization."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class SingleHeadAttention(nn.Module):
    """Single-head attention mechanism.
    
    All Q, K, V, and output projections are Dense layers that will be quantized.
    
    Attributes:
        d_model: Model dimension
        use_bias: Whether to use bias in projections
    """
    d_model: int
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x, mask: Optional[jax.Array] = None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V projections - these are Dense layers that will be quantized
        q = nn.Dense(self.d_model, use_bias=self.use_bias, name='query')(x)
        k = nn.Dense(self.d_model, use_bias=self.use_bias, name='key')(x)
        v = nn.Dense(self.d_model, use_bias=self.use_bias, name='value')(x)
        
        # Compute attention scores
        # q @ k.T scaled by sqrt(d_model)
        scale = jnp.sqrt(self.d_model).astype(x.dtype)
        attn_scores = jnp.einsum('bqd,bkd->bqk', q, k) / scale
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            attn_scores = jnp.where(mask, attn_scores, -1e10)
        
        # Softmax over key dimension
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        
        # Apply attention to values
        attn_output = jnp.einsum('bqk,bkd->bqd', attn_weights, v)
        
        # Output projection - another Dense layer to quantize
        output = nn.Dense(self.d_model, use_bias=self.use_bias, name='out')(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with two Dense layers.
    
    Attributes:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        use_bias: Whether to use bias
    """
    d_model: int
    d_ff: int
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # First Dense layer: d_model -> d_ff
        x = nn.Dense(self.d_ff, use_bias=self.use_bias, name='fc1')(x)
        x = nn.gelu(x)
        
        # Second Dense layer: d_ff -> d_model
        x = nn.Dense(self.d_model, use_bias=self.use_bias, name='fc2')(x)
        
        return x


class TransformerBlock(nn.Module):
    """Single Transformer decoder block.
    
    Attributes:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        use_bias: Whether to use bias in Dense layers
    """
    d_model: int
    d_ff: int
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, x, mask: Optional[jax.Array] = None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = SingleHeadAttention(
            d_model=self.d_model,
            use_bias=self.use_bias,
            name='attention'
        )(x, mask=mask)
        x = nn.LayerNorm(name='ln1')(x + attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = FeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            use_bias=self.use_bias,
            name='ffn'
        )(x)
        x = nn.LayerNorm(name='ln2')(x + ff_output)
        
        return x


class SimpleTransformer(nn.Module):
    """Simple decoder-only Transformer model.
    
    This model has multiple Dense layers that LQER should quantize:
    - Token embedding projection
    - In each attention: Q, K, V, and output projections (4 Dense layers)
    - In each FFN: 2 Dense layers
    - Final output projection
    
    Total Dense layers per block: 4 (attention) + 2 (FFN) = 6
    Total for model: 1 (embed) + 6 * n_layers + 1 (output) = 2 + 6 * n_layers
    
    Attributes:
        vocab_size: Vocabulary size
        d_model: Model dimension
        d_ff: Feed-forward dimension
        n_layers: Number of transformer blocks
        max_seq_len: Maximum sequence length
        use_bias: Whether to use bias in Dense layers
    """
    vocab_size: int
    d_model: int
    d_ff: int
    n_layers: int
    max_seq_len: int = 512
    use_bias: bool = False
    
    @nn.compact
    def __call__(self, tokens, train: bool = False):
        """
        Args:
            tokens: Input token IDs of shape (batch, seq_len)
            train: Whether in training mode (for dropout, not used here)
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = tokens.shape
        
        # Token embedding - this is a Dense layer that will be quantized
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model, name='embed')(tokens)
        
        # Add positional encoding (learnable)
        pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.02),
            (1, self.max_seq_len, self.d_model)
        )
        x = x + pos_embed[:, :seq_len, :]
        
        # Create causal mask (lower triangular)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = mask[None, :, :]  # Add batch dimension
        
        # Apply transformer blocks
        for i in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                d_ff=self.d_ff,
                use_bias=self.use_bias,
                name=f'block_{i}'
            )(x, mask=mask)
        
        # Final layer norm
        x = nn.LayerNorm(name='ln_final')(x)
        
        # Output projection to vocabulary - another Dense layer to quantize
        logits = nn.Dense(self.vocab_size, use_bias=False, name='output')(x)
        
        return logits
    
    def count_dense_layers(self):
        """Returns the number of Dense layers in the model."""
        # 1 embedding + n_layers * (4 attention + 2 ffn) + 1 output
        return 1 + self.n_layers * 6 + 1

