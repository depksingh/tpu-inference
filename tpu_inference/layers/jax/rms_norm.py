from collections.abc import Iterable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import dtypes
from flax.typing import Array, Axes, Dtype
from jax import lax
from jax.sharding import PartitionSpec as P

def rmsnorm_forward(x, residual, weight, epsilon) -> jax.Array | tuple[jax.Array, jax.Array]:
    orig_dtype = x.dtype
    x_f32 = jnp.asarray(x, jnp.float32)
    if residual is not None:
        x_f32 += jnp.asarray(residual, jnp.float32)
        residual = x_f32.astype(orig_dtype)
    mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
    y = jnp.asarray(x_f32 * lax.rsqrt(mean2 + epsilon), jnp.float32)
    output = (y * jnp.asarray(weight, jnp.float32)).astype(orig_dtype)
    if residual is None:
        return output
    else:
        return output, residual


def dual_rmsnorm_forward(
    x: jax.Array,
    residual: jax.Array,
    weight1: nnx.Param[jax.Array],
    weight2: nnx.Param[jax.Array],
    epsilon: float,
) -> tuple[jax.Array, jax.Array]:
    """Apply two RMSNorms with shared residual path, returning (y2, residual).

    Equivalent to fused_dual_residual_rmsnorm: first adds residual, applies
    norm with weight1 to produce y1 (discarded), then norm with weight2 to produce y2.
    """
    y1 = rmsnorm_forward(x, None, weight1, epsilon)
    y2, residual_out = rmsnorm_forward(y1, residual, weight2, epsilon)
    return y2, residual_out
