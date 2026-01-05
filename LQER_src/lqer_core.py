"""Core LQER (Low-rank Quantization Error Reconstruction) implementation.

This module provides the core functionality for LQER quantization:
- LqerRule: Quantization rule with rank parameter
- LqerWeight: Container for quantized weight + error correction matrices
- lqer_quantize_params: Function to quantize parameters with LQER
- LqerProvider: Inference provider that applies error correction
- benchmark_lqer_model: Helper function for accurate per-layer timing

Timing/Benchmarking:
    For per-layer runtime benchmarking, use:
    
    1. Enable timing in LqerProvider:
       provider = LqerProvider(rules, enable_timing=True)
    
    2. Use benchmark_lqer_model() for accurate results:
       results = benchmark_lqer_model(model, params, input, num_runs=10)
    
    3. Or access timing data directly:
       LqerProvider.print_timings()
       timings = LqerProvider.get_timings()
    
    Note: For most accurate per-layer timing, consider using JAX's profiler:
    ```python
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        output = model.apply(variables, input)
        output.block_until_ready()
    ```
"""

import dataclasses
from typing import Any, Callable
import re
import time
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import qwix
from qwix._src.providers import ptq
from qwix._src import qconfig
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


from experiments.models.transformer import SimpleTransformer
from experiments.models.simple_mlp import SimpleMLP


# 1. Custom Rule to hold the Rank 'k'
@dataclasses.dataclass(frozen=True, kw_only=True)
class LqerRule(qwix.QuantizationRule):
    """
    Extends standard rules to include the target rank for SVD reconstruction.
    rank: Number of singular values to keep for error correction.
    """
    rank: int = 32


# 2. Custom Data Structure to hold the multi-matrix weight
# This is NOT wrapped in WithAux - it's used directly in the params tree
@flax.struct.dataclass
class LqerWeight:
    """Container for LQER quantized weight with low-rank error correction."""
    w_q: qwix.QArray  # Quantized weight
    error_a: jax.Array  # Low-rank error matrix A (U * S)
    error_b: jax.Array  # Low-rank error matrix B (Vh)
    layer_id: str = ""  # Identifier for this layer (e.g., parameter path)
    
    @property
    def shape(self):
        return self.w_q.shape
    
    @property
    def dtype(self):
        return self.w_q.dtype


def lqer_quantize_params(fp_params, abs_ptq_params, rules: list[LqerRule], debug=False):
    """
    Quantize parameters using LQER: quantized weight + low-rank error correction.
    
    Args:
        fp_params: The original floating-point parameters
        abs_ptq_params: Abstract quantized params from eval_shape (contains WithAux with HowToQuantize)
        rules: List of LqerRule objects that define which layers to quantize and with what rank
        debug: If True, print debug information
    
    Returns:
        Dict with LqerWeight containers for quantized layers, original values for others
    """
    import re
    
    flat_fp = flax.traverse_util.flatten_dict(fp_params)
    flat_abs = flax.traverse_util.flatten_dict(abs_ptq_params)
    quantized_params = {}
    
    def find_matching_rule(path_str: str) -> LqerRule | None:
        """Find the first LqerRule that matches the given path."""
        for rule in rules:
            if isinstance(rule, LqerRule) and re.match(rule.module_path, path_str):
                return rule
        return None
    
    if debug:
        print("\n[DEBUG lqer_quantize_params] Abstract params structure:")
        for path, node in flat_abs.items():
            print(f"  {'/'.join(path)}: type={type(node).__name__}")
            if isinstance(node, ptq.WithAux):
                print(f"    -> how: {type(node.how).__name__}")

    for path, W in flat_fp.items():
        abs_node = flat_abs.get(path)
        path_str = '/'.join(path)
        
        # Find matching LqerRule for this path
        lqer_rule = find_matching_rule(path_str)
        
        if debug:
            is_withaux = isinstance(abs_node, ptq.WithAux)
            print(f"  Processing {path_str}: WithAux={is_withaux}, LqerRule match={lqer_rule is not None}")
        
        # Check if this param should be quantized with LQER
        # Condition: abs_node is WithAux (means it's a weight) AND we have a matching LqerRule
        if isinstance(abs_node, ptq.WithAux) and lqer_rule is not None:
            # 1. Standard Quantization
            # qwix.quantize public API accepts keyword args: qtype, channelwise_axes, etc.
            # We can get these from abs_node.how (the HowToQuantize created by PtqProvider)
            how = abs_node.how
            w_q_array = qwix.quantize(
                W,
                qtype=how.qtype,
                channelwise_axes=tuple(how.channelwise_axes),
                tiled_axes=how.tiled_axes or None,
                calibration_method=how.calibration_method,
            )
            
            # 2. SVD Error Calculation
            W_dequant = qwix.dequantize(w_q_array)
            E = W - W_dequant
            U, S, Vh = jnp.linalg.svd(E, full_matrices=False)
            
            # 3. Get rank from the LqerRule
            k = lqer_rule.rank
            error_a = U[:, :k] * S[:k]  # (in_features, k)
            error_b = Vh[:k, :]          # (k, out_features)
            
            if debug:
                print(f"    -> LQER quantized: rank={k}, error_a={error_a.shape}, error_b={error_b.shape}")
            
            # 4. Store as LqerWeight directly (NOT inside WithAux)
            # Include layer_id for timing/benchmarking purposes
            lqer_container = LqerWeight(
                w_q=w_q_array, 
                error_a=error_a, 
                error_b=error_b,
                layer_id=path_str
            )
            quantized_params[path] = lqer_container
        else:
            # Keep non-LQER params as-is
            quantized_params[path] = W
            
    return flax.traverse_util.unflatten_dict(quantized_params)


# 3. Simple MLP Model for testing
class LqerProvider(qconfig.QuantizationProvider):
    """
    Custom provider for LQER inference.
    
    Handles LqerWeight params directly, using:
    - Quantized weight path: y_q = x @ dequant(w_q)
    - Error correction path: correction = (x @ A) @ B
    - Final output: y = y_q + correction
    """
    
    def __init__(self, rules, enable_timing=False):
        super().__init__(rules)
        self.enable_timing = enable_timing
    
    # Class-level counters to track which code paths are executed
    _path_counts = {'lqer': 0, 'withaux_lqer': 0, 'withaux_qarray': 0, 'fallback': 0}
    
    # Class-level timing data: {layer_id: [time1, time2, ...]}
    _layer_timings = {}
    
    def promote_dtype(self, *args, **kwargs):
        """Intercept promote_dtype to skip LqerWeight (handled later in dot_general)."""
        from collections.abc import Sequence
        if len(args) == 1 and isinstance(args[0], Sequence):
            args = args[0]  # nnx version
        # Skip LqerWeight and WithAux - they will be handled in dot_general
        array_args = [x if isinstance(x, jax.Array) else None for x in args]
        array_args = flax.linen.dtypes.promote_dtype(*array_args, **kwargs)
        return [x if x is not None else y for x, y in zip(array_args, args)]
    
    @classmethod
    def reset_counts(cls):
        cls._path_counts = {'lqer': 0, 'withaux_lqer': 0, 'withaux_qarray': 0, 'fallback': 0}
    
    @classmethod
    def reset_timings(cls):
        """Reset all layer timing data."""
        cls._layer_timings = {}
    
    @classmethod
    def print_counts(cls):
        print("\n[DEBUG] Code path execution counts:")
        for path, count in cls._path_counts.items():
            status = "✓ CALLED" if count > 0 else "✗ NOT CALLED"
            print(f"  - {path}: {count} times ({status})")
    
    @classmethod
    def get_timings(cls) -> dict[str, list[float]]:
        """Get timing data for all layers.
        
        Returns:
            Dict mapping layer_id to list of execution times (in seconds).
        """
        return cls._layer_timings.copy()
    
    @classmethod
    def print_timings(cls, summary=True):
        """Print timing statistics for all layers.
        
        Args:
            summary: If True, print summary statistics. If False, print all individual timings.
        """
        if not cls._layer_timings:
            print("\n[Timing] No timing data collected.")
            return
        
        print("\n" + "=" * 70)
        print("LAYER TIMING BENCHMARKS")
        print("=" * 70)
        
        # Calculate statistics
        stats = []
        for layer_id, times in cls._layer_timings.items():
            if times:
                total = sum(times)
                mean = total / len(times)
                min_time = min(times)
                max_time = max(times)
                stats.append({
                    'layer_id': layer_id,
                    'count': len(times),
                    'total_ms': total * 1000,
                    'mean_ms': mean * 1000,
                    'min_ms': min_time * 1000,
                    'max_ms': max_time * 1000,
                })
        
        # Sort by total time (descending)
        stats.sort(key=lambda x: x['total_ms'], reverse=True)
        
        # Print table
        print(f"\n{'Layer ID':<40} {'Count':<8} {'Total (ms)':<12} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 100)
        for stat in stats:
            print(f"{stat['layer_id']:<40} {stat['count']:<8} {stat['total_ms']:<12.3f} {stat['mean_ms']:<12.3f} {stat['min_ms']:<12.3f} {stat['max_ms']:<12.3f}")
        
        if not summary:
            print("\nDetailed timings:")
            for layer_id, times in cls._layer_timings.items():
                print(f"\n  {layer_id}:")
                for i, t in enumerate(times):
                    print(f"    Call {i+1}: {t*1000:.3f} ms")
        
        total_time = sum(sum(times) for times in cls._layer_timings.values())
        print(f"\nTotal execution time: {total_time*1000:.3f} ms")
        print("=" * 70)
    
    def _record_timing(self, layer_id: str, duration: float):
        """Record timing for a layer (side effect, safe in JAX)."""
        if self.enable_timing:
            if layer_id not in LqerProvider._layer_timings:
                LqerProvider._layer_timings[layer_id] = []
            LqerProvider._layer_timings[layer_id].append(duration)
    
    def dot_general(
        self,
        lhs: jax.Array,
        rhs: Any,
        dimension_numbers: jax.lax.DotDimensionNumbers,
        precision: jax.lax.PrecisionLike = None,
        preferred_element_type: jax.typing.DTypeLike | None = None,
        **kwargs,
    ) -> jax.Array:
        """Handle dot_general with LQER weights or fall back to standard."""
        
        # PATH 1: LqerWeight directly in params tree (THIS IS THE MAIN LQER PATH)
        # This is called when lqer_quantize_params stores LqerWeight directly
        if isinstance(rhs, LqerWeight):
            LqerProvider._path_counts['lqer'] += 1
            
            # Start timing if enabled
            start_time = time.perf_counter() if self.enable_timing else None
            
            # 1. Dequantize and compute main quantized path
            w_dequant = qwix.dequantize(rhs.w_q)
            y_q = jax.lax.dot_general(
                lhs, w_dequant, dimension_numbers,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )
            
            # 2. Compute error correction: (lhs @ A) @ B
            # This is the key LQER contribution - low-rank error reconstruction
            (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
            
            # For simple matmul (no batch dims): just do sequential matmuls
            if not lhs_ba and not rhs_ba and len(lhs_ca) == 1 and len(rhs_ca) == 1:
                intermediate = lhs @ rhs.error_a  # (..., k)
                correction = intermediate @ rhs.error_b  # (..., out)
            else:
                # Fallback for more complex dimension cases
                intermediate = jax.lax.dot_general(
                    lhs, rhs.error_a, dimension_numbers,
                    precision=precision,
                    preferred_element_type=preferred_element_type,
                )
                correction = intermediate @ rhs.error_b
            
            result = y_q + correction
            
            # Record timing if enabled
            if self.enable_timing and start_time is not None:
                layer_id = rhs.layer_id if rhs.layer_id else f"unknown_{id(rhs)}"
                duration = time.perf_counter() - start_time
                self._record_timing(layer_id, duration)
            
            return result
        
        # PATH 2: WithAux wrapper (for compatibility/future use)
        # This handles cases where LqerWeight might be wrapped in WithAux,
        # or for mixed PTQ/LQER models. Currently NOT called in basic usage.
        if isinstance(rhs, ptq.WithAux):
            rhs_array = rhs.array
            if isinstance(rhs_array, LqerWeight):
                LqerProvider._path_counts['withaux_lqer'] += 1
                return self.dot_general(
                    lhs, rhs_array, dimension_numbers,
                    precision=precision,
                    preferred_element_type=preferred_element_type,
                    **kwargs
                )
            # Standard QArray from PTQ
            LqerProvider._path_counts['withaux_qarray'] += 1
            rhs_dequant = qwix.dequantize(rhs_array)
            return jax.lax.dot_general(
                lhs, rhs_dequant, dimension_numbers,
                precision=precision,
                preferred_element_type=preferred_element_type,
            )
        
        # PATH 3: Fallback for regular jax arrays (non-quantized layers)
        LqerProvider._path_counts['fallback'] += 1
        return jax.lax.dot_general(
            lhs, rhs, dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
    
    def get_intercept_map(self) -> dict[str, Callable[..., Any]]:
        """Define which JAX functions to intercept."""
        return super().get_intercept_map() | {
            'jax.lax.dot_general': self.dot_general,
            'flax.linen.dtypes.promote_dtype': self.promote_dtype,
            'flax.nnx.nn.dtypes.promote_dtype': self.promote_dtype,
        }


# --- TESTING THE COMPLETE PIPELINE ---

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
        _run_single_test(model, fp_variables, fp_output, model_input, rules, key, enable_timing=True)


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
    _run_single_test(model, fp_variables, fp_output, tokens, rules, key, enable_timing=True)


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


def benchmark_lqer_model(
    model,
    lqer_params: dict,
    model_input: Any,
    num_warmup: int = 3,
    num_runs: int = 10,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Benchmark LQER model with proper JAX blocking for accurate per-layer timing.
    
    This function runs the model multiple times and properly blocks after each execution
    to get accurate timing measurements. It's the recommended way to benchmark LQER models.
    
    IMPORTANT: The model must be created with a LqerProvider that has enable_timing=True
    for this function to collect per-layer timings.
    
    Args:
        model: The quantized LQER model (must be created with LqerProvider(enable_timing=True))
        lqer_params: Quantized parameters dict
        model_input: Input to the model
        num_warmup: Number of warmup runs (to compile JIT, etc.)
        num_runs: Number of benchmark runs
        verbose: If True, print timing results
    
    Returns:
        Dict with timing statistics: {
            'layer_timings': {layer_id: [times...]},
            'total_time': total execution time (seconds),
            'mean_time': mean execution time per run (seconds),
            'num_runs': number of benchmark runs,
        }
    
    Example:
        >>> rules = [LqerRule(module_path='.*', weight_qtype=jnp.int4, rank=16)]
        >>> provider = LqerProvider(rules, enable_timing=True)
        >>> model = qwix.quantize_model(base_model, provider)
        >>> params = lqer_quantize_params(fp_params, abs_params, rules)
        >>> results = benchmark_lqer_model(model, params, input_data, num_runs=20)
    """
    # Run warmup
    if verbose:
        print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        output = model.apply({'params': lqer_params}, model_input)
        output.block_until_ready()
    
    # Reset timings before benchmark
    LqerProvider.reset_timings()
    
    # Benchmark runs
    if verbose:
        print(f"Benchmarking ({num_runs} runs)...")
    
    total_start = time.perf_counter()
    for run_idx in range(num_runs):
        output = model.apply({'params': lqer_params}, model_input)
        output.block_until_ready()  # Block to ensure computation completes
        
        if verbose and (run_idx + 1) % max(1, num_runs // 10) == 0:
            print(f"  Run {run_idx + 1}/{num_runs} completed")
    
    total_end = time.perf_counter()
    total_time = total_end - total_start
    mean_time = total_time / num_runs
    
    timings = LqerProvider.get_timings()
    
    if verbose:
        print(f"\nBenchmark complete:")
        print(f"  Total time: {total_time*1000:.2f} ms")
        print(f"  Mean time per run: {mean_time*1000:.2f} ms")
        if timings:
            LqerProvider.print_timings()
        else:
            print("  Note: No per-layer timings collected. Ensure model was created with enable_timing=True")
    
    return {
        'layer_timings': timings,
        'total_time': total_time,
        'mean_time': mean_time,
        'num_runs': num_runs,
    }


def _run_single_test(model, fp_variables, fp_output, model_input, rules, key, enable_timing=False):
    """Run a single LQER test with the given rules.
    
    Args:
        model: The model to test
        fp_variables: FP32 model variables
        fp_output: FP32 model output for comparison
        model_input: Input to the model
        rules: LQER quantization rules
        key: JAX random key
        enable_timing: If True, collect and print timing benchmarks
    """
    # Quantize the Model (Architecture only)
    lqer_model = qwix.quantize_model(model, LqerProvider(rules, enable_timing=enable_timing))
    
    # Quantize the Params (The SVD happens here!)
    temp_ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
    abs_ptq_params = jax.eval_shape(temp_ptq_model.init, key, model_input)['params']
    lqer_params = lqer_quantize_params(fp_variables['params'], abs_ptq_params, rules, debug=False)
    
    # Run Inference
    LqerProvider.reset_counts()
    if enable_timing:
        LqerProvider.reset_timings()
    
    lqer_output = lqer_model.apply({'params': lqer_params}, model_input)
    
    # Show which code paths were executed
    LqerProvider.print_counts()
    
    # Print timing if enabled
    if enable_timing:
        LqerProvider.print_timings()
    
    # Compare with FP32
    max_diff = jnp.max(jnp.abs(fp_output - lqer_output))
    print(f"LQER max error: {max_diff:.6f}")
    
    # Also test standard PTQ for comparison
    ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
    ptq_params = qwix.quantize_params(fp_variables['params'], abs_ptq_params)
    ptq_output = ptq_model.apply({'params': ptq_params}, model_input)
    
    ptq_diff = jnp.max(jnp.abs(fp_output - ptq_output))
    print(f"PTQ max error:  {ptq_diff:.6f}")
    print(f"LQER improvement: {(ptq_diff - max_diff) / ptq_diff * 100:.1f}%")


if __name__ == "__main__":
    run_lqer_test()
