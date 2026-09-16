"""Data-parallel ranks: ownership, reductions, and checkpoint interchange.

Two gloo ranks on CPU exercise the real collective path: the same in-place
reduce-scatter, reduce, all-reduce, all-gather, broadcast, and send/recv
calls the CUDA ranks make, with the same site table and the same optimizer
partition.
"""

import math
import os

import pytest
import torch
import torch.multiprocessing as mp

from delta_feedback_experiment.data import DEFAULT_SOURCE, write_synthetic
from delta_feedback_experiment.distributed import Topology
from delta_feedback_experiment.model import DeltaModel, condition_config
from delta_feedback_experiment.optim import NorMuonH, build_optimizers
from delta_feedback_experiment.sites import ParameterSites
from delta_feedback_experiment.train import (
    Trainer,
    parse_run_args,
    read_checkpoint,
    train,
)

GEOMETRY = {
    "vocab_size": 31,
    "dim": 16,
    "layers": 8,
    "heads": 2,
    "kv_heads": 2,
    "head_dim": 8,
    "expert_intermediate": 8,
    "num_routed_experts": 3,
    "experts_per_token": 2,
    "pkda_heads": 2,
    "pkda_head_dim": 8,
    "max_seq_len": 5,
}


def tiny(condition="fl"):
    torch.manual_seed(5)
    return DeltaModel(condition_config(condition, **GEOMETRY))


def test_topology_reads_the_launcher_environment(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert Topology.from_environment() == Topology(0, 1, 0)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    topology = Topology.from_environment()
    assert (topology.rank, topology.world, topology.local_rank) == (3, 4, 1)
    assert not topology.main
    assert topology.chunk(15) == 4 and topology.padded(15) == 16
    monkeypatch.setenv("RANK", "4")
    with pytest.raises(ValueError, match="outside"):
        Topology.from_environment()


def test_ownership_is_a_partition_every_rank_computes_alike():
    """Banks chunk by expert with the shared expert first; dense sites go
    whole to one rank; every rank derives the same table; NAdam parameters
    are replicated and the arena all-reduces."""
    model = tiny()
    world = 4
    tables = [ParameterSites(model, Topology(rank, world)) for rank in range(world)]
    owner = tables[0].owner
    assert all(table.owner == owner for table in tables)
    trainable = {p for p in model.parameters() if p.requires_grad}
    assert set(owner) == tables[0].sharded
    assert tables[0].sharded | tables[0].replicated == trainable
    owned = [table.owned for table in tables]
    assert set().union(*owned) == tables[0].sharded
    assert all(not a & b for i, a in enumerate(owned) for b in owned[i + 1 :])
    bank = model.blocks[0].mlp
    members = [bank.shared.gate_up_proj.weight] + [
        expert.gate_up_proj.weight for expert in bank.experts
    ]
    # Four entries over four ranks: one expert each, the shared one on rank 0.
    assert [owner[member] for member in members] == [0, 1, 2, 3]
    attention = model.blocks[3].attn
    gate = model.attention_gates[model.blocks[3].global_gate_index].weight
    assert gate in tables[0].replicated and attention.qkv_proj.weight in owner
    assert model.embed_tokens.weight in tables[0].replicated
    loads = [
        sum(p.numel() for p in table.owned) for table in tables
    ]
    assert max(loads) - min(loads) < 0.25 * max(loads)


def test_gradient_norm_is_global_unclipped_and_finite_checked():
    model = tiny("f")
    sites = ParameterSites(model, Topology())
    sites.allocate()
    gradients = sites.gradients()
    expected = 0.0
    for index, buffer in enumerate(gradients.values()):
        buffer.fill_(0.01 * (index + 1))
        expected += buffer.square().sum().item()
    assert sites.gradient_norm() == pytest.approx(math.sqrt(expected), rel=1e-6)
    assert all(buffer.abs().sum() > 0 for buffer in gradients.values())
    next(iter(gradients.values()))[0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        sites.gradient_norm()


def test_adopted_parameters_are_their_working_views_and_masters_follow():
    """After adoption the optimizer writes the master and the parameter agrees."""
    model = tiny("f")
    sites = ParameterSites(model, Topology())
    optimizers = build_optimizers(model, owned=sites.owned)
    normuonh = optimizers[0]
    assert isinstance(normuonh, NorMuonH)
    sites.allocate()
    trainer = Trainer(model, optimizers, sites, parse_run_args(["t", "--steps", "1"]), Topology())
    matrix = model.blocks[0].attn.o_proj.weight
    assert matrix.data_ptr() == model.blocks[0].attn.o_shadow.data_ptr()
    master = normuonh.master_of(matrix)
    assert master.data_ptr() != matrix.data_ptr()
    assert torch.equal(master, matrix)
    trainer.gradients[matrix].normal_()
    trainer.prepare_optimizer(frozenset(trainer.gradients))
    normuonh.step()
    assert torch.equal(normuonh.master_of(matrix), matrix)
    assert not torch.equal(master, torch.zeros_like(master))


# -- two gloo ranks ------------------------------------------------------------


def _settings(tmp_path):
    return {
        "condition": "fl",
        "data-root": tmp_path / "data",
        "out-dir": tmp_path / "runs",
        "vocab-size": 31,
        "dim": 16,
        "layers": 8,
        "heads": 2,
        "kv-heads": 2,
        "head-dim": 8,
        "expert-intermediate": 8,
        "num-routed-experts": 3,
        "experts-per-token": 2,
        "pkda-heads": 2,
        "pkda-head-dim": 8,
        "seq-len": 4,
        "batch-rows": 4,
        "micro-rows": 1,
        "loop-iterations": 2,
        "steps": 3,
        "warmup-frac": 0,
        "cooldown-frac": 0,
        "recurrence-start": 0,
        "three-rate": 0,
        "eval-every": 3,
        "eval-rows": 3,
        "snapshot-every": 1,
        "device": "cpu",
    }


def _flags(settings):
    return [item for key, value in settings.items() for item in (f"--{key}", str(value))]


def _rank_main(rank, world, port, argv, result_path):
    os.environ.update(
        {
            "WORLD_SIZE": str(world),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    torch.set_num_threads(1)
    summary = train(argv)
    if rank == 0:
        torch.save(summary, result_path)


def _run_ranks(world, argv, tmp_path, port):
    result_path = tmp_path / f"summary-{port}.pt"
    mp.spawn(
        _rank_main, args=(world, port, argv, str(result_path)), nprocs=world, join=True
    )
    return torch.load(result_path, weights_only=False)


@pytest.mark.parametrize("port", [29611])
def test_two_ranks_match_one_and_exchange_checkpoints(tmp_path, port):
    """Two ranks reproduce the single-process trajectory to reduction-order
    rounding, write the same single-file checkpoint schema, resume from a
    one-rank snapshot, and hand a two-rank snapshot back to one rank."""
    write_synthetic(
        tmp_path / "data" / DEFAULT_SOURCE, train_tokens=160, val_tokens=40, vocab=31
    )
    settings = _settings(tmp_path)
    flags = _flags(settings)
    single = train(["single", *flags])
    paired = _run_ranks(2, ["paired", *flags, "--ranks", "2"], tmp_path, port)
    for key in ("loss", "val", "val_fused", "val_mtp", "val_one"):
        assert paired[key] == pytest.approx(single[key], rel=2e-4), key
    one = read_checkpoint(tmp_path / "runs/single.pt.3")
    two = read_checkpoint(tmp_path / "runs/paired.pt.3")
    assert set(one["state"]) == set(two["state"])
    for name in one["state"]:
        assert one["state"][name].dtype == two["state"][name].dtype
        assert one["state"][name].shape == two["state"][name].shape
        torch.testing.assert_close(one["state"][name], two["state"][name], rtol=2e-3, atol=2e-3)
    one_opt, two_opt = one["optimizer"]["stack"][0], two["optimizer"]["stack"][0]
    assert set(one_opt["state"]) == set(two_opt["state"])
    assert one_opt["param_groups"] == two_opt["param_groups"]
    for index, entry in one_opt["state"].items():
        assert set(entry) == set(two_opt["state"][index])
        for kind, value in entry.items():
            assert value.shape == two_opt["state"][index][kind].shape
    # A one-rank snapshot resumes on two ranks, and a two-rank snapshot on one.
    half = train(["half", *flags, "--max-steps", "2"])
    assert half["step"] == 2
    resumed_paired = _run_ranks(
        2, ["half", *flags, "--ranks", "2", "--resume"], tmp_path, port + 1
    )
    assert resumed_paired["step"] == 3
    assert resumed_paired["loss"] == pytest.approx(single["loss"], rel=2e-4)
    partial_paired = _run_ranks(
        2, ["quarter", *flags, "--ranks", "2", "--max-steps", "2"], tmp_path, port + 2
    )
    assert partial_paired["step"] == 2
    resumed_single = train(["quarter", *flags, "--resume"])
    assert resumed_single["step"] == 3
    assert resumed_single["loss"] == pytest.approx(single["loss"], rel=2e-4)
