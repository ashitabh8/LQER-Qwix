# LQER: Low-rank Quantization Error Reconstruction for Qwix

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Qwix](https://img.shields.io/badge/Built%20with-Qwix-green.svg)](https://github.com/google/qwix)

> **Note:** The project is in initial stages. More models need to be tested with different datasets, with the core technique working.

**LQER** extends [Qwix](https://github.com/google/qwix) (Google's JAX quantization library) with **low-rank error correction** to significantly improve Post-Training Quantization (PTQ) accuracy.

## What is LQER?

LQER improves quantized model accuracy by storing a low-rank approximation of the quantization error:

1. **Quantize** weight matrix W → W_q (standard PTQ)
2. **Compute error** E = W - dequantize(W_q)
3. **SVD decomposition** E ≈ U @ S @ V^T
4. **Keep top-k** singular values: E_k = (U[:, :k] * S[:k]) @ V[:k, :]
5. **Store** W_q, A = U[:, :k] * S[:k], B = V[:k, :]
6. **Inference** y = dequantize(W_q) @ x + (x @ A) @ B

This achieves **29-60% error reduction** with minimal storage overhead.

## Accuracy

**Note:** Currently tested on randomly generated input and weights.

| Model | Quantization | PTQ Error | LQER Error | Improvement |
|-------|-------------|-----------|------------|-------------|
| SimpleMLP (2 layers) | INT4 | 0.101 | 0.000 | **100%** |
| Transformer (14 layers) | INT4, rank=16 | 0.523 | 0.371 | **29%** |
| Transformer (14 layers) | INT4, rank=32 | 0.523 | 0.286 | **45%** |
| Transformer (14 layers) | INT8, rank=16 | 0.032 | 0.022 | **32%** |

## Installation

```bash
# Clone the repository
git clone https://github.com/ashitabh8/LQER-Qwix.git
cd LQER-Qwix

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies (includes Qwix at tested commit)
pip install -r requirements.txt
```

## Quick Start

```bash
# Test on SimpleMLP (2 Dense layers)
python tests/ptq_test.py

# Test on Transformer (14 Dense layers with attention)
python tests/test_transformer_lqer.py
```

## Usage Example

```python
import jax
import jax.numpy as jnp
from LQER_src import LqerRule, LqerProvider, lqer_quantize_params
from models import SimpleMLP
import qwix

# 1. Create your model
model = SimpleMLP(dhidden=64, dout=10)
key = jax.random.key(0)
model_input = jax.random.normal(key, (1, 16))

# 2. Initialize with FP32 weights
fp_params = model.init(key, model_input)['params']

# 3. Define LQER quantization rules
rules = [LqerRule(
    module_path='.*',        # Match all layers (regex pattern)
    weight_qtype=jnp.int4,   # Quantize to INT4
    rank=16                  # Use rank-16 error correction
)]

# 4. Quantize the model architecture
lqer_model = qwix.quantize_model(model, LqerProvider(rules))

# 5. Quantize the parameters
temp_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
abs_params = jax.eval_shape(temp_model.init, key, model_input)['params']
lqer_params = lqer_quantize_params(fp_params, abs_params, rules)

# 6. Run inference with error correction
output = lqer_model.apply({'params': lqer_params}, model_input)
```

## Project Structure

```
LQER-Qwix/
├── LQER_src/              # Core LQER implementation
│   ├── __init__.py        # Package exports
│   └── lqer_core.py       # LqerRule, LqerWeight, LqerProvider
│
├── models/                # Example model architectures
│   ├── __init__.py
│   ├── simple_mlp.py      # 2-layer MLP
│   └── transformer.py     # Decoder-only Transformer
│
├── tests/                 # Test scripts
│   ├── ptq_test.py        # Basic PTQ comparison
│   └── test_transformer_lqer.py  # Transformer with multiple configs
│
├── requirements.txt       # Dependencies (Qwix @ specific commit)
└── README.md             # This file
```

## Features

✅ **Automatic Dense layer detection** - LQER finds and quantizes all matching layers  
✅ **Flexible pattern matching** - Use regex to target specific layers  
✅ **Multiple quantization types** - INT4, INT8, or any Qwix-supported type  
✅ **Rank-accuracy tradeoff** - Higher rank = better accuracy, more storage  
✅ **Works with complex models** - Tested on Transformers with 14+ Dense layers  
✅ **Seamless Qwix integration** - Extends existing Qwix workflows

## How It Works

### LQER Components

1. **`LqerRule`** - Extends `QuantizationRule` with rank parameter
2. **`LqerWeight`** - Container storing (W_q, A, B) matrices
3. **`lqer_quantize_params()`** - Computes SVD and creates LqerWeight
4. **`LqerProvider`** - Inference provider applying error correction

### What Gets Quantized?

LQER automatically quantizes all Dense layers matching your pattern:

**In Transformer (14 layers):**
- Token embedding (1 layer)
- Per attention block:
  - Query, Key, Value projections (3 layers)
  - Output projection (1 layer)
  - Feed-forward FC1, FC2 (2 layers)
- Output projection (1 layer)

**Pattern examples:**
- `'.*'` - All layers
- `'Dense_0/.*'` - Only first Dense layer
- `'.*attention.*'` - Only attention layers

## Advanced Configuration

### Rank Selection

```python
# Trade-off between accuracy and storage
LqerRule(..., rank=8)   # ~5-15% improvement, minimal storage
LqerRule(..., rank=16)  # ~20-30% improvement, balanced
LqerRule(..., rank=32)  # ~40-60% improvement, higher storage
```

## Tested With

- **Qwix**: Commit `5c9ba31` (December 2024)
- **JAX**: 0.4.20+
- **Flax**: 0.8.0+
- **Python**: 3.10+

## TODO

Future work and testing planned:

- [ ] Test on Llama 1 and Llama 2 models
- [ ] Add Runtime profiling logic and test on GPU/TPU (currently only tested on CPU) 
- [ ] Evaluate on WikiText-103 dataset
- [ ] Conduct comprehensive accuracy benchmarks on real models
- [ ] Perform runtime performance testing and profiling
- [ ] Compare with other quantization methods (GPTQ, AWQ)
- [ ] Add support for more model architectures

## Citing

If you use LQER in your research, please cite the original paper:

```bibtex
@article{zhang2024lqer,
  title={LQER: Low-Rank Quantization Error Reconstruction for LLMs},
  author={Zhang, Cheng and Cheng, Jianyi and Constantinides, George A and Zhao, Yiren},
  journal={arXiv preprint arXiv:2402.02446},
  year={2024}
}
```

And this implementation:

```bibtex
@software{lqer2025,
  title={LQER: Low-rank Quantization Error Reconstruction for Qwix},
  author={Misra, Ashitabh},
  year={2025},
  url={https://github.com/ashitabh8/LQER-Qwix}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built on top of [Qwix](https://github.com/google/qwix) by Google
- Inspired by SVD-based compression and error correction techniques
- Thanks to the JAX and Flax communities

## Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using [Qwix](https://github.com/google/qwix) and [JAX](https://github.com/google/jax)**

