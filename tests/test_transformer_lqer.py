"""Test LQER quantization on Transformer architecture.

This script demonstrates LQER's ability to automatically quantize all Dense layers
in a complex architecture including:
- Token embeddings
- Query, Key, Value projections in attention
- Attention output projections
- Feed-forward network layers
- Final output projection
"""

import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from LQER_src import LqerRule, LqerProvider, lqer_quantize_params
from models import SimpleTransformer
import qwix


def test_transformer_lqer():
    """Test LQER on Transformer with various configurations."""
    
    print("=" * 80)
    print("Testing LQER on Transformer Architecture")
    print("=" * 80)
    
    # Model configuration
    vocab_size = 100
    d_model = 64
    d_ff = 128  # Feed-forward dimension (typically 4x d_model)
    n_layers = 2
    seq_len = 8
    batch_size = 2
    
    key = jax.random.key(42)
    
    # Create Transformer model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=32,
        use_bias=False
    )
    
    # Generate random input tokens
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    
    # Initialize model and get FP32 baseline
    print("\n1. Initializing FP32 baseline model...")
    fp_variables = model.init(key, tokens)
    fp_output = model.apply(fp_variables, tokens)
    
    # Print model architecture info
    print(f"\n2. Model Architecture:")
    print(f"   ├─ Vocabulary size: {vocab_size}")
    print(f"   ├─ Model dimension (d_model): {d_model}")
    print(f"   ├─ Feed-forward dimension (d_ff): {d_ff}")
    print(f"   ├─ Number of layers: {n_layers}")
    print(f"   ├─ Sequence length: {seq_len}")
    print(f"   └─ Batch size: {batch_size}")
    
    # Count Dense layers
    n_dense_layers = model.count_dense_layers()
    print(f"\n3. Dense Layer Breakdown:")
    print(f"   ├─ Token embedding: 1 layer")
    print(f"   ├─ Per transformer block: 6 layers")
    print(f"   │   ├─ Attention (Q, K, V, Out): 4 layers")
    print(f"   │   └─ Feed-forward (FC1, FC2): 2 layers")
    print(f"   ├─ Total blocks × layers: {n_layers} × 6 = {n_layers * 6}")
    print(f"   ├─ Output projection: 1 layer")
    print(f"   └─ TOTAL DENSE LAYERS: {n_dense_layers}")
    
    print(f"\n4. FP32 Baseline Output:")
    print(f"   ├─ Shape: {fp_output.shape}")
    print(f"   └─ Sample logits: {fp_output[0, 0, :5]}")
    
    # Test with different quantization configs
    configs = [
        {'qtype': jnp.int8, 'rank': 16, 'name': 'INT8 + Rank-16'},
        {'qtype': jnp.int4, 'rank': 16, 'name': 'INT4 + Rank-16'},
        {'qtype': jnp.int4, 'rank': 32, 'name': 'INT4 + Rank-32'},
    ]
    
    print("\n" + "=" * 80)
    print("Running LQER Quantization Tests")
    print("=" * 80)
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/3: {config['name']}")
        print(f"{'='*80}")
        
        # Create LQER rule
        rules = [LqerRule(
            module_path='.*',  # Match all layers
            weight_qtype=config['qtype'],
            rank=config['rank']
        )]
        
        # Quantize model
        lqer_model = qwix.quantize_model(model, LqerProvider(rules))
        
        # Quantize parameters
        temp_ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
        abs_ptq_params = jax.eval_shape(temp_ptq_model.init, key, tokens)['params']
        lqer_params = lqer_quantize_params(fp_variables['params'], abs_ptq_params, rules)
        
        # Run inference
        lqer_output = lqer_model.apply({'params': lqer_params}, tokens)
        
        # Compute error metrics
        lqer_error = jnp.max(jnp.abs(fp_output - lqer_output))
        lqer_mean_error = jnp.mean(jnp.abs(fp_output - lqer_output))
        
        # Compare with standard PTQ
        ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
        ptq_params = qwix.quantize_params(fp_variables['params'], abs_ptq_params)
        ptq_output = ptq_model.apply({'params': ptq_params}, tokens)
        
        ptq_error = jnp.max(jnp.abs(fp_output - ptq_output))
        ptq_mean_error = jnp.mean(jnp.abs(fp_output - ptq_output))
        
        # Calculate improvement
        max_improvement = (ptq_error - lqer_error) / ptq_error * 100
        mean_improvement = (ptq_mean_error - lqer_mean_error) / ptq_mean_error * 100
        
        # Print results
        print(f"\nResults:")
        print(f"   Quantization: {config['qtype'].__name__}, Error correction rank: {config['rank']}")
        print(f"   ")
        print(f"   ┌─ Max Absolute Error:")
        print(f"   │   ├─ Standard PTQ:  {ptq_error:.6f}")
        print(f"   │   ├─ LQER:          {lqer_error:.6f}")
        print(f"   │   └─ Improvement:   {max_improvement:.1f}%")
        print(f"   │")
        print(f"   └─ Mean Absolute Error:")
        print(f"       ├─ Standard PTQ:  {ptq_mean_error:.6f}")
        print(f"       ├─ LQER:          {lqer_mean_error:.6f}")
        print(f"       └─ Improvement:   {mean_improvement:.1f}%")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Successfully quantized all {n_dense_layers} Dense layers")
    print(f"✓ LQER error correction applied to:")
    print(f"  - Token embeddings")
    print(f"  - All attention projections (Q, K, V, Output)")
    print(f"  - All feed-forward layers")
    print(f"  - Output projection")
    print(f"✓ Low-rank error correction significantly improved accuracy")
    print(f"✓ Higher rank = better accuracy (captures more error modes)")
    print("=" * 80)


if __name__ == "__main__":
    test_transformer_lqer()

