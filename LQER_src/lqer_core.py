"""Core LQER (Low-rank Quantization Error Reconstruction) implementation.

This module provides the core functionality for LQER quantization:
- LqerRule: Quantization rule with rank parameter
- LqerWeight: Container for quantized weight + error correction matrices
- lqer_quantize_params: Function to quantize parameters with LQER
- LqerProvider: Inference provider that applies error correction
- profile_lqer_model: Function to profile LQER models using JAX profiler

Profiling:
    To profile the LQER model and analyze dot_general performance:
    
    >>> from LQER_src.lqer_core import profile_lqer_model
    >>> trace_dir = profile_lqer_model(model, params, input_data)
    >>> # View trace at https://ui.perfetto.dev
    
    The trace will include annotations for:
    - LQER_dot_general: Main LQER computation
    - LQER_dequantize_and_matmul: Quantized weight path
    - LQER_error_correction: Error correction computation
    - LQER_error_matmul_simple/complex: Error matrix multiplications
    - LQER_final_add: Final addition step
"""

import dataclasses
from typing import Any, Callable
import re
import os
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import qwix
from qwix._src.providers import ptq
from qwix._src import qconfig
import sys
from pathlib import Path

# Get project root directory
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Default trace directory in project tmp folder
_DEFAULT_TRACE_DIR = _PROJECT_ROOT / "tmp"


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
            lqer_container = LqerWeight(
                w_q=w_q_array, 
                error_a=error_a, 
                error_b=error_b
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
    
    def __init__(self, rules):
        super().__init__(rules)
    
    # Class-level counters to track which code paths are executed
    _path_counts = {'lqer': 0, 'withaux_lqer': 0, 'withaux_qarray': 0, 'fallback': 0}
    
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
    def print_counts(cls):
        print("\n[DEBUG] Code path execution counts:")
        for path, count in cls._path_counts.items():
            status = "✓ CALLED" if count > 0 else "✗ NOT CALLED"
            print(f"  - {path}: {count} times ({status})")
    
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
            
            # Add profiler annotation for LQER path
            with jax.named_scope("LQER_dot_general"):
                # 1. Dequantize and compute main quantized path
                with jax.named_scope("LQER_dequantize_and_matmul"):
                    w_dequant = qwix.dequantize(rhs.w_q)
                    y_q = jax.lax.dot_general(
                        lhs, w_dequant, dimension_numbers,
                        precision=precision,
                        preferred_element_type=preferred_element_type,
                    )
                
                # 2. Compute error correction: (lhs @ A) @ B
                # This is the key LQER contribution - low-rank error reconstruction
                (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
                
                with jax.named_scope("LQER_error_correction"):
                    # For simple matmul (no batch dims): just do sequential matmuls
                    if not lhs_ba and not rhs_ba and len(lhs_ca) == 1 and len(rhs_ca) == 1:
                        with jax.named_scope("LQER_error_matmul_simple"):
                            intermediate = lhs @ rhs.error_a  # (..., k)
                            correction = intermediate @ rhs.error_b  # (..., out)
                    else:
                        # Fallback for more complex dimension cases
                        with jax.named_scope("LQER_error_matmul_complex"):
                            intermediate = jax.lax.dot_general(
                                lhs, rhs.error_a, dimension_numbers,
                                precision=precision,
                                preferred_element_type=preferred_element_type,
                            )
                            correction = intermediate @ rhs.error_b
                
                with jax.named_scope("LQER_final_add"):
                    result = y_q + correction
            
            return result
        
        # PATH 2: WithAux wrapper (for compatibility/future use)
        # This handles cases where LqerWeight might be wrapped in WithAux,
        # or for mixed PTQ/LQER models. Currently NOT called in basic usage.
        if isinstance(rhs, ptq.WithAux):
            rhs_array = rhs.array
            if isinstance(rhs_array, LqerWeight):
                LqerProvider._path_counts['withaux_lqer'] += 1
                with jax.named_scope("LQER_withaux_path"):
                    return self.dot_general(
                        lhs, rhs_array, dimension_numbers,
                        precision=precision,
                        preferred_element_type=preferred_element_type,
                        **kwargs
                    )
            # Standard QArray from PTQ
            LqerProvider._path_counts['withaux_qarray'] += 1
            with jax.named_scope("PTQ_withaux_path"):
                rhs_dequant = qwix.dequantize(rhs_array)
                return jax.lax.dot_general(
                    lhs, rhs_dequant, dimension_numbers,
                    precision=precision,
                    preferred_element_type=preferred_element_type,
                )
        
        # PATH 3: Fallback for regular jax arrays (non-quantized layers)
        LqerProvider._path_counts['fallback'] += 1
        with jax.named_scope("LQER_fallback_path"):
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


def profile_lqer_model(
    model,
    lqer_params: dict,
    model_input: Any,
    trace_dir: str | Path = None,
    num_warmup: int = 3,
    num_runs: int = 5,
    create_perfetto_link: bool = False,
):
    """
    Profile LQER model using JAX profiler.
    
    This function runs the model with JAX profiling enabled and generates a trace
    that can be viewed in Perfetto (https://ui.perfetto.dev).
    
    Args:
        model: The quantized LQER model
        lqer_params: Quantized parameters dict
        model_input: Input to the model
        trace_dir: Directory to save trace files (default: project_root/tmp/jax-trace)
        num_warmup: Number of warmup runs before profiling
        num_runs: Number of profiled runs
        create_perfetto_link: If True, tries to create a local server link (may fail if port in use)
    
    Returns:
        Path to the trace directory (as string)
    """
    # Use default trace directory if not specified
    if trace_dir is None:
        trace_dir = str(_DEFAULT_TRACE_DIR / "jax-trace")
    else:
        trace_dir = str(trace_dir)
    # JIT compile the apply function
    @jax.jit
    def apply_model(variables, inputs):
        return model.apply(variables, inputs)
    
    # Warmup runs to compile JIT
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        _ = apply_model({'params': lqer_params}, model_input)
        _.block_until_ready()
    
    # Create trace directory
    os.makedirs(trace_dir, exist_ok=True)
    
    # Profile the model
    print(f"Profiling ({num_runs} runs)...")
    print(f"Trace will be saved to: {trace_dir}")
    
    # Try to create perfetto link, but handle port conflicts gracefully
    try:
        with jax.profiler.trace(trace_dir, create_perfetto_link=create_perfetto_link):
            for run_idx in range(num_runs):
                with jax.profiler.TraceAnnotation(f"LQER_inference_run_{run_idx}"):
                    output = apply_model({'params': lqer_params}, model_input)
                    output.block_until_ready()
    except OSError as e:
        if "Address already in use" in str(e) and create_perfetto_link:
            # Port conflict - retry without the server
            print(f"Warning: Port conflict when creating Perfetto link. Retrying without server...")
            with jax.profiler.trace(trace_dir, create_perfetto_link=False):
                for run_idx in range(num_runs):
                    with jax.profiler.TraceAnnotation(f"LQER_inference_run_{run_idx}"):
                        output = apply_model({'params': lqer_params}, model_input)
                        output.block_until_ready()
        else:
            raise
    
    print(f"\nProfiling complete!")
    print(f"Trace saved to: {trace_dir}")
    print(f"\nTo view the trace:")
    print(f"  1. Go to https://ui.perfetto.dev")
    print(f"  2. Click 'Open trace file'")
    print(f"  3. Navigate to: {trace_dir}")
    print(f"  4. Select the trace file (usually named with a timestamp)")
    
    return trace_dir
