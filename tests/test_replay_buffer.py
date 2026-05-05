"""Tests for `model.replay_buffer`. Generates a small parquet shard
via the Rust `selfplay-stub` binary, ingests it, exercises sampling
and eviction. Skipped if the Rust binary hasn't been built yet."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from model.encoder import MOVE_SPACE
from model.replay_buffer import ReplayBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]
SELFPLAY_STUB = REPO_ROOT / "target" / "release" / "selfplay-stub"


def _build_selfplay_stub() -> Path:
    """Build the stub binary if not present. Skip the test if cargo
    isn't available."""
    if SELFPLAY_STUB.exists():
        return SELFPLAY_STUB
    if shutil.which("cargo") is None:
        pytest.skip("cargo not available; can't build selfplay-stub")
    subprocess.run(
        ["cargo", "build", "--release", "-p", "abalone-selfplay", "--bin", "selfplay-stub"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return SELFPLAY_STUB


@pytest.fixture(scope="module")
def shard_path(tmp_path_factory) -> Path:
    binary = _build_selfplay_stub()
    out_dir = tmp_path_factory.mktemp("rb_shards")
    out = out_dir / "shard.parquet"
    subprocess.run(
        [
            str(binary),
            "--games",
            "2",
            "--simulations",
            "20",
            "--out",
            str(out),
            "--seed",
            "42",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return out


def test_ingest_decodes_planes_with_correct_shape(shard_path: Path):
    rb = ReplayBuffer()
    n = rb.ingest_shard(shard_path, gen=0)
    assert n > 0
    assert rb.total_size() == n


def test_sample_returns_correctly_shaped_batch(shard_path: Path):
    rb = ReplayBuffer(augment=True)
    rb.ingest_shard(shard_path, gen=0)
    rng = np.random.default_rng(0)
    batch = rb.sample(8, rng)
    assert batch.planes.shape == (8, 6, 9, 9)
    assert batch.policy.shape == (8, MOVE_SPACE)
    assert batch.legal_mask.shape == (8, MOVE_SPACE)
    assert batch.z.shape == (8,)
    assert batch.q.shape == (8,)
    # Policy on legal moves sums to 1 (when there's at least one visit).
    legal_sums = batch.policy.sum(axis=1)
    assert np.all((legal_sums > 0.999) & (legal_sums < 1.001))
    # All policy mass lands on legal moves.
    assert np.all(batch.policy * (1 - batch.legal_mask) == 0)


def test_sample_without_augment_is_deterministic(shard_path: Path):
    rb = ReplayBuffer(augment=False)
    rb.ingest_shard(shard_path, gen=0)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    b1 = rb.sample(16, rng1)
    b2 = rb.sample(16, rng2)
    np.testing.assert_array_equal(b1.planes, b2.planes)
    np.testing.assert_array_equal(b1.policy, b2.policy)


def test_evict_below_drops_old_chunks(shard_path: Path):
    rb = ReplayBuffer()
    rb.ingest_shard(shard_path, gen=0)
    rb.ingest_shard(shard_path, gen=5)
    rb.ingest_shard(shard_path, gen=10)
    n0 = rb.total_size()
    removed = rb.evict_below(5)
    assert removed > 0
    assert rb.total_size() == n0 - removed
    assert rb.generations() == [5, 10]


def test_concat_within_same_generation(shard_path: Path):
    rb = ReplayBuffer()
    rb.ingest_shard(shard_path, gen=0)
    n_first = rb.total_size()
    rb.ingest_shard(shard_path, gen=0)
    assert rb.total_size() == 2 * n_first
    assert rb.generations() == [0]


def test_augmentation_changes_planes(shard_path: Path):
    # With augmentation on, repeated samples of the same buffer should
    # produce different plane content (vanishingly unlikely to match by
    # chance over a batch of 16).
    rb = ReplayBuffer(augment=True)
    rb.ingest_shard(shard_path, gen=0)
    rng = np.random.default_rng(0)
    b1 = rb.sample(16, rng)
    b2 = rb.sample(16, rng)
    # At least one element should differ between the two batches.
    assert not np.array_equal(b1.planes, b2.planes)
