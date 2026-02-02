import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .sgl_weight_utils import WeightMapping

def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,
    expert_type_names: tuple[str, str, str] = (
        "gate_proj",
        "up_proj",
        "down_proj",
    ),  # expert source names [gate, up, down]
    expert_concat_axis_map: dict[
        str, int
    ] = None,  # Map from source weight name to its concatenation axis (default is None)
    moe_backend: str = "epmoe",
    moe_path: str = "mlp",
    source_expert_pattern: str = "experts.{i}",
) -> dict:
    """Generate a unified mapping dictionary for MoE layer expert weights."""
    if moe_backend == "epmoe":
        expert_type_map = {
            expert_type_names[0]: "wi_0",
            expert_type_names[1]: "wi_1",
            expert_type_names[2]: "wo",
        }
    elif moe_backend == "fused":
        expert_type_map = {
            expert_type_names[0]: "w1",
            expert_type_names[1]: "w3",
            expert_type_names[2]: "w2",
        }
    else:
        raise ValueError(f"Unsupported MoE backend: {moe_backend}")

    if expert_concat_axis_map is None:
        expert_concat_axis_map = {}

    mappings = {}
    for source_name, target_name in expert_type_map.items():
        # Target path for JAX model parameters (matching EPMoE internal variables)
        target_path_base = f"{target_prefix}.{moe_path}.{target_name}"

        # Source weight paths for all experts to be loaded and concatenated
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.weight"
            for i in range(num_experts)
        ]

        if moe_backend == "epmoe":
            # Sharding logic based on EPMoE PartitionSpec:
            # wi_0/wi_1 (Input projections) use P("expert", "tensor", None)
            # wo (Output projection) uses P("expert", None, "tensor")
            sharding = (
                ("expert", None, "tensor") if target_name == "wo" else ("expert", "tensor", None)
            )
            transpose = False
        elif moe_backend == "fused":
            # Fused MoE kernel shards experts across the full EP mesh, i.e. the
            # product of ("data", "tensor"). Shard expert dim (axis=0) across
            # both mesh axes so each device owns a disjoint expert slice.
            sharding = (("data", "tensor"), None, None)
            transpose = True
        else:
            raise ValueError(f"Unsupported MoE backend: {moe_backend}")

        concat_axis = expert_concat_axis_map.get(source_name)

        # Use __MOE_EXPERTS__ prefix to indicate aggregated MoE weight loading
        mappings[f"__MOE_EXPERTS__{target_path_base}"] = WeightMapping(
            target_path=[target_path_base] + expert_keys,
            sharding=sharding,
            transpose=transpose,
            concat_axis=concat_axis,
        )

    return mappings
