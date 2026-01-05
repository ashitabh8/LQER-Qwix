import jax
import jax.numpy as jnp
from flax import linen as nn
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import SimpleMLP
import qwix

model = SimpleMLP(dhidden=64, dout=16)
model_input = jax.random.uniform(jax.random.key(0), (8, 16))

fp_params = model.init(jax.random.key(1), model_input)['params']
fp_output = model.apply({'params': fp_params}, model_input)

rules = [
    qwix.QuantizationRule(
        module_path='.*',
        weight_qtype='int8',
        act_qtype='int8',
    )
]

ptq_model = qwix.quantize_model(model, qwix.PtqProvider(rules))
abs_ptq_params = jax.eval_shape(ptq_model.init, jax.random.key(0), model_input)['params']
ptq_params = qwix.quantize_params(fp_params, abs_ptq_params)
ptq_output = ptq_model.apply({'params': ptq_params}, model_input)

print("Input shape:", model_input.shape)
print("FP32 output shape:", fp_output.shape)
print("PTQ output shape:", ptq_output.shape)
print()
print("FP32 output (first 5):", fp_output[0, :5])
print("PTQ output (first 5):", ptq_output[0, :5])
print()
print("Max abs diff:", jnp.max(jnp.abs(fp_output - ptq_output)))

