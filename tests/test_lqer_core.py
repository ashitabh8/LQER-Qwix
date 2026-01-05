"""Test suite for LQER core functionality.

This module contains tests for:
- Simple MLP models
- Transformer models
- Integration tests comparing LQER vs PTQ
- Profiling support
"""

import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from LQER_src.lqer_core import (
    LqerRule,
    LqerProvider,
    lqer_quantize_params,
    profile_lqer_model,
)
from LQER_src import lqer_core
from experiments.models.transformer import SimpleTransformer
from experiments.models.simple_mlp import SimpleMLP
import qwix


def _run_single_test(model, fp_variables, fp_output, model_input, rules, key, enable_profiling=True):
    """Run a single LQER test with the given rules.
    
    Args:
        model: The model to test
        fp_variables: FP32 model variables
        fp_output: FP32 model output for comparison
        model_input: Input to the model
        rules: LQER quantization rules
        key: JAX random key
        enable_profiling: If True, enable JAX profiling
    """
    # Quantize the Model (Architecture only)
    lqer_model = qwix.quantize_model(model, LqerProvider(rules))
    
    # Quantize the Params (The SVD happens here!)
    temp_ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
    abs_ptq_params = jax.eval_shape(temp_ptq_model.init, key, model_input)['params']
    lqer_params = lqer_quantize_params(fp_variables['params'], abs_ptq_params, rules, debug=False)
    
    # Run Inference with JIT compilation
    LqerProvider.reset_counts()
    
    # JIT compile the apply function for better performance
    @jax.jit
    def apply_model(variables, inputs):
        return lqer_model.apply(variables, inputs)
    
    # Warmup run to compile JIT
    _ = apply_model({'params': lqer_params}, model_input)
    _.block_until_ready()
    
    # Profile if enabled
    if enable_profiling:
        trace_dir = lqer_core._DEFAULT_TRACE_DIR / f"jax-trace-lqer-{id(model)}"
        profile_lqer_model(
            lqer_model, 
            lqer_params, 
            model_input, 
            trace_dir=trace_dir,
            create_perfetto_link=False  # Disable server to avoid port conflicts
        )
    
    # Actual inference
    lqer_output = apply_model({'params': lqer_params}, model_input)
    lqer_output.block_until_ready()
    
    # Show which code paths were executed
    LqerProvider.print_counts()
    
    # Compare with FP32
    max_diff = jnp.max(jnp.abs(fp_output - lqer_output))
    print(f"LQER max error: {max_diff:.6f}")
    
    # Also test standard PTQ for comparison
    ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
    ptq_params = qwix.quantize_params(fp_variables['params'], abs_ptq_params)
    
    @jax.jit
    def apply_ptq_model(variables, inputs):
        return ptq_model.apply(variables, inputs)
    
    ptq_output = apply_ptq_model({'params': ptq_params}, model_input)
    ptq_output.block_until_ready()
    
    ptq_diff = jnp.max(jnp.abs(fp_output - ptq_output))
    print(f"PTQ max error:  {ptq_diff:.6f}")
    print(f"LQER improvement: {(ptq_diff - max_diff) / ptq_diff * 100:.1f}%")


def test_simple_mlp():
    """Test LQER on a simple MLP."""
    print("=" * 70)
    print("TEST 1: SimpleMLP (2 Dense layers)")
    print("=" * 70)
    
    key = jax.random.key(0)
    model_input = jax.random.normal(key, (1, 16))
    
    model = SimpleMLP(dhidden=64, dout=10)
    fp_variables = model.init(key, model_input)
    fp_output = model.apply(fp_variables, model_input)
    
    print(f"\nFP32 Output (first 5): {fp_output[0, :5]}")
    print(f"Model has 2 Dense layers")
    
    # Test with different ranks
    for rank in [8, 16]:
        print(f"\n--- Testing with rank={rank} ---")
        rules = [LqerRule(module_path='.*', weight_qtype=jnp.int4, rank=rank)]
        _run_single_test(model, fp_variables, fp_output, model_input, rules, key, enable_profiling=True)


def test_transformer():
    """Test LQER on a Transformer model with multiple Dense layers."""
    print("\n" + "=" * 70)
    print("TEST 2: SimpleTransformer (Multiple Dense layers)")
    print("=" * 70)
    
    # Small transformer config
    vocab_size = 100
    d_model = 64
    d_ff = 128
    n_layers = 2
    seq_len = 8
    batch_size = 2
    
    key = jax.random.key(42)
    
    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=32,
        use_bias=False
    )
    
    # Random token IDs
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
    
    # Initialize and get FP32 output
    fp_variables = model.init(key, tokens)
    fp_output = model.apply(fp_variables, tokens)
    
    print(f"\nModel config:")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - d_model: {d_model}, d_ff: {d_ff}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - Sequence length: {seq_len}")
    
    # Count Dense layers
    n_dense_layers = model.count_dense_layers()
    print(f"  - Total Dense layers: {n_dense_layers}")
    print(f"    (1 embed + {n_layers} × (4 attn + 2 ffn) + 1 output)")
    
    print(f"\nFP32 logits shape: {fp_output.shape}")
    print(f"FP32 logits sample: {fp_output[0, 0, :5]}")
    
    # Test LQER with rank=16
    rank = 16
    print(f"\n--- Testing with rank={rank} ---")
    rules = [LqerRule(module_path='.*', weight_qtype=jnp.int4, rank=rank)]
    _run_single_test(model, fp_variables, fp_output, tokens, rules, key, enable_profiling=True)


def run_lqer_test():
    """Run all LQER tests."""
    print("=" * 70)
    print("LQER (Low-rank Quantization Error Reconstruction) Test Suite")
    print("=" * 70)
    
    # Test 1: Simple MLP
    test_simple_mlp()
    
    # Test 2: Transformer
    test_transformer()
    
    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_lqer_test()

