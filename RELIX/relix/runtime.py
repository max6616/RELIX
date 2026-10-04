"""Unified model loader; scientific components retain their frozen parameters."""
from pathlib import Path
from types import SimpleNamespace

import torch


def load(checkpoint, device="cpu"):
    from lattice_relation.consensus_model import ConsensusRelationTransformer
    from lattice_relation.robust_inference import RobustRelationIndexer
    from lattice_relation.profile_symmetry import ProfileSymmetryTransformer
    from lattice_relation.candidate_ranker import CandidateRanker
    from lattice_relation.hybrid_measured import HybridMeasuredRelationIndexer
    from lattice_relation.reflection_set import ReflectionSetSpaceGroup
    from lattice_relation.continuous_postprocess import ContinuousPostprocessor
    from lattice_relation.relix_no_rescue import BudgetedIndexer

    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("format") != "relix-unified-checkpoint-v1":
        raise ValueError("Expected a RELIX unified checkpoint")
    config, states = payload["config"], payload["components"]

    state = states["geometry"]
    network = ConsensusRelationTransformer(**state["config"]).to(device).eval()
    network.load_state_dict(state["model"], strict=True)
    geometry = RobustRelationIndexer(network, state["tokens"], device,
                                     state.get("sampling", "uniform"))
    geometry.radial_fraction = state.get("radial_fraction", .5)
    for name, value in config["geometry"].items():
        if not hasattr(geometry, name):
            raise ValueError("Unknown geometry option: " + name)
        setattr(geometry, name, value)

    state = states["symmetry"]
    symmetry = ProfileSymmetryTransformer(
        state["base_mean"], state["base_std"],
        profile_mean=state["profile_mean"], profile_std=state["profile_std"],
        **state["config"]).to(device)
    symmetry.load_state_dict(state["model"], strict=True)
    symmetry.eval()
    base = HybridMeasuredRelationIndexer(geometry, symmetry, device, **config["measured"])
    base.native_beam_plane = config["instrument"]["native_beam_plane"]
    base.rigid_instrument = config["instrument"]["single_scan_rigid_detector"]

    state = states["ranker"]
    ranker = CandidateRanker(state["mean"], state["std"], state["width"],
                              config["rank_strength"]).to(device)
    ranker.load_state_dict(state["model"], strict=True)
    base.geometry.candidate_ranker = ranker.eval()

    state = states["reflection"]
    expert = ReflectionSetSpaceGroup(**state["config"]).to(device)
    expert.load_state_dict(state["model"], strict=True)
    post = ContinuousPostprocessor(base, expert.eval(), config["reflection_set_strength"],
                                   config["posterior_threshold"])
    root = next(p for p in Path(__file__).resolve().parents
                if (p / "RELIX_PACKAGE.json").is_file())
    return BudgetedIndexer(SimpleNamespace(postprocessor=post), config["runtime"], root)
