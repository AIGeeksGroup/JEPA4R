import torch

from jepa4r.policy.bc.lewm_cortex import _remap_vit_state_dict_for_runtime


def test_remap_vit_state_dict_matches_legacy_transformers_keys():
    source = {
        "vit.layers.0.attention.q_proj.weight": torch.ones(2, 2),
        "vit.layers.0.attention.k_proj.bias": torch.zeros(2),
        "vit.layers.0.attention.v_proj.weight": torch.ones(2, 2),
        "vit.layers.0.attention.o_proj.bias": torch.zeros(2),
        "vit.layers.0.mlp.fc1.weight": torch.ones(2, 2),
        "vit.layers.0.mlp.fc2.bias": torch.zeros(2),
        "vit.layernorm.weight": torch.ones(2),
    }
    target_keys = {
        "vit.encoder.layer.0.attention.attention.query.weight",
        "vit.encoder.layer.0.attention.attention.key.bias",
        "vit.encoder.layer.0.attention.attention.value.weight",
        "vit.encoder.layer.0.attention.output.dense.bias",
        "vit.encoder.layer.0.intermediate.dense.weight",
        "vit.encoder.layer.0.output.dense.bias",
        "vit.layernorm.weight",
    }

    remapped = _remap_vit_state_dict_for_runtime(source, target_keys)

    assert set(remapped) == target_keys
    assert torch.equal(
        remapped["vit.encoder.layer.0.attention.attention.query.weight"], source["vit.layers.0.attention.q_proj.weight"]
    )
