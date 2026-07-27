"""Tests for `model.encoder`: plane layout + D6 symmetry math.

Coverage rationale: with the symmetry implementation moved entirely to
Python, this is now the *spec*. We test it from multiple angles so a
regression surfaces immediately:

  * Plane layout: the `(14, 9, 9)` encoding of MODEL.md §5 — marbles,
    the two loss thermometers, the ply normaliser, the valid mask, and
    the deliberate choice to fill the constant planes across all 81
    cells rather than masking them.
  * Group axioms: identity, R has order 6, F has order 2, F is an
    involution, all 12 elements are distinct on a generic cell.
  * Closure: composition of any two elements is in the group.
  * Concrete fixed points and orbits: R cycles the 6 corners, F fixes
    the NE-SW diagonal, the centre cell E5 is fixed by every element.
  * Direction permutations match the canonical engine convention.
  * Move-perm respects the inline/broadside split (no cross-leakage).
  * Plane permutation: identity is no-op; centre marble is invariant;
    a marble at A1 maps under R to A5 as the cell math predicts; the
    marble planes permute while the constant planes do not.
  * Policy permutation: mass on move `i` lands on `MOVE_PERM[s, i]`.
  * Combined consistency: rotating planes + rotating policy is
    equivalent to applying the symmetry to the underlying state.

Agreement with the *Rust* encoder is a separate concern and lives in
`tests/test_conformance.py`; nothing here can catch the two languages
drifting apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from model.encoder import (
    BROADSIDE_DIRS,
    BROADSIDE_OFFSET,
    CELL_PERM,
    DIR_PERM,
    LOSS_THERMOMETER_PLANES,
    MOVE_PERM,
    MOVE_SPACE,
    NUM_CELLS,
    NUM_DIRS,
    NUM_INPUT_CHANNELS,
    NUM_SYMS,
    OPP_LOSSES_PLANE,
    OPP_MARBLES_PLANE,
    OWN_LOSSES_PLANE,
    OWN_MARBLES_PLANE,
    PLANE_SIZE,
    PLY_PLANE,
    POSITIVE_DIRS,
    SYM_MATS,
    VALID_CELL_MASK,
    VALID_MASK_PLANE,
    PositionRecord,
    apply_sym_to_planes,
    apply_sym_to_policy,
    bb_to_plane,
    decode_broadside,
    decode_inline,
    encode_broadside,
    encode_inline,
    encode_position,
)

# ----- helpers ---------------------------------------------------------------


def cell(name: str) -> int:
    """Cell index from name like "A1", "E5", "I9"."""
    assert len(name) == 2
    r = ord(name[0]) - ord("A")
    q = ord(name[1]) - ord("1")
    return r * 9 + q


def is_valid_cell(c: int) -> bool:
    if c < 0 or c >= NUM_CELLS:
        return False
    r, q = divmod(c, 9)
    return abs(q - r) <= 4


def bb(*names: str) -> int:
    """Bitboard from cell names."""
    out = 0
    for n in names:
        out |= 1 << cell(n)
    return out


def record(
    own_bb: int = 0,
    opp_bb: int = 0,
    own_losses: int = 0,
    opp_losses: int = 0,
    ply: int = 0,
    max_plies: int = 200,
    turn: int = 0,
) -> PositionRecord:
    return PositionRecord(
        own_bb=own_bb,
        opp_bb=opp_bb,
        own_losses=own_losses,
        opp_losses=opp_losses,
        ply=ply,
        max_plies=max_plies,
        turn=turn,
    )


# Indexes into the 12-element group, matching the construction in
# encoder.py: 0..6 are R^k (60° rotations), 6..12 are R^k * F.
IDENTITY = 0
R = 1
F = 6

# Every plane except the two marble planes is a broadcast constant.
CONSTANT_PLANES = tuple(range(OWN_LOSSES_PLANE, NUM_INPUT_CHANNELS))


# ----- plane layout ----------------------------------------------------------


class TestPlaneLayout:
    """The `(14, 9, 9)` contract of MODEL.md §5. The Rust encoder mirrors
    this exactly; `tests/test_conformance.py` proves it."""

    def test_shape_and_dtype(self):
        planes = encode_position(record())
        assert planes.shape == (NUM_INPUT_CHANNELS, 9, 9)
        assert planes.dtype == np.float32
        assert planes.size == PLANE_SIZE == 14 * 81

    def test_plane_indices_are_the_documented_layout(self):
        assert (OWN_MARBLES_PLANE, OPP_MARBLES_PLANE) == (0, 1)
        assert OWN_LOSSES_PLANE == 2
        assert OPP_LOSSES_PLANE == 7
        assert LOSS_THERMOMETER_PLANES == 5
        assert PLY_PLANE == 12
        assert VALID_MASK_PLANE == 13

    def test_marble_planes_are_placed_by_cell(self):
        planes = encode_position(
            record(own_bb=bb("A1", "E5"), opp_bb=bb("I9", "C3"))
        )
        own, opp = planes[OWN_MARBLES_PLANE], planes[OPP_MARBLES_PLANE]
        assert own.sum() == 2 and opp.sum() == 2
        assert own[0, 0] == 1.0 and own[4, 4] == 1.0
        assert opp[8, 8] == 1.0 and opp[2, 2] == 1.0
        # No leakage between the two.
        assert float((own * opp).sum()) == 0.0

    def test_marble_planes_are_side_to_move_relative(self):
        # The record carries `own`/`opp`, never black/white: the same
        # bitboards with the turn flipped produce identical planes.
        a = encode_position(record(own_bb=bb("A1"), opp_bb=bb("I9"), turn=0))
        b = encode_position(record(own_bb=bb("A1"), opp_bb=bb("I9"), turn=1))
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_own_loss_thermometer(self, n: int):
        planes = encode_position(record(own_losses=n))
        for k in range(LOSS_THERMOMETER_PLANES):
            expected = 1.0 if n >= k + 1 else 0.0
            assert planes[OWN_LOSSES_PLANE + k].min() == expected
            assert planes[OWN_LOSSES_PLANE + k].max() == expected
        # ...and the opponent's group stays cold.
        assert planes[OPP_LOSSES_PLANE : OPP_LOSSES_PLANE + 5].sum() == 0.0

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
    def test_opp_loss_thermometer(self, n: int):
        planes = encode_position(record(opp_losses=n))
        for k in range(LOSS_THERMOMETER_PLANES):
            expected = 1.0 if n >= k + 1 else 0.0
            assert planes[OPP_LOSSES_PLANE + k].min() == expected
        assert planes[OWN_LOSSES_PLANE : OWN_LOSSES_PLANE + 5].sum() == 0.0

    def test_thermometer_is_monotone(self):
        for n in range(6):
            planes = encode_position(record(own_losses=n, opp_losses=5 - n))
            for group in (OWN_LOSSES_PLANE, OPP_LOSSES_PLANE):
                vals = [planes[group + k][0, 0] for k in range(LOSS_THERMOMETER_PLANES)]
                assert vals == sorted(vals, reverse=True), (
                    f"thermometer at {group} is not monotone for {n} losses: {vals}"
                )

    def test_asymmetric_losses_land_in_their_own_plane_group(self):
        """The regression test for the swapped-capture-plane bug: with the
        two counters different, the groups must not be interchangeable."""
        planes = encode_position(record(own_losses=4, opp_losses=1))
        own = planes[OWN_LOSSES_PLANE : OWN_LOSSES_PLANE + 5][:, 0, 0]
        opp = planes[OPP_LOSSES_PLANE : OPP_LOSSES_PLANE + 5][:, 0, 0]
        np.testing.assert_array_equal(own, [1, 1, 1, 1, 0])
        np.testing.assert_array_equal(opp, [1, 0, 0, 0, 0])
        assert not np.array_equal(own, opp)

    def test_ply_plane_uses_the_records_own_cap(self):
        # Not a hardcoded 400: the same ply under two caps reads differently.
        assert encode_position(record(ply=50, max_plies=200))[PLY_PLANE][0, 0] == 0.25
        assert encode_position(record(ply=50, max_plies=100))[PLY_PLANE][0, 0] == 0.5

    def test_ply_plane_clamps_to_one(self):
        planes = encode_position(record(ply=999, max_plies=200))
        assert planes[PLY_PLANE].min() == 1.0
        assert planes[PLY_PLANE].max() == 1.0

    def test_zero_cap_is_zero_not_nan(self):
        planes = encode_position(record(ply=5, max_plies=0))
        assert np.isfinite(planes).all()
        assert planes[PLY_PLANE].sum() == 0.0

    def test_valid_mask_plane_covers_exactly_61_cells(self):
        planes = encode_position(record())
        np.testing.assert_array_equal(planes[VALID_MASK_PLANE], VALID_CELL_MASK)
        assert planes[VALID_MASK_PLANE].sum() == 61.0

    def test_constant_planes_are_not_masked(self):
        """Deliberate: planes 2-12 cover all 81 cells, off-board slots
        included. The network masks internally; if the encoder masked here
        too it would silently diverge from Rust."""
        planes = encode_position(
            record(own_losses=5, opp_losses=5, ply=100, max_plies=200)
        )
        for p in range(OWN_LOSSES_PLANE, VALID_MASK_PLANE):
            values = np.unique(planes[p])
            assert len(values) == 1, f"plane {p} is not constant across all 81 cells"
        assert planes[OWN_LOSSES_PLANE].sum() == 81.0
        assert planes[PLY_PLANE].sum() == 81.0 * 0.5

    def test_marbles_never_appear_off_board_for_a_real_bitboard(self):
        # Off-board bits cannot be set by the engine; the encoder simply
        # copies the bitboard, so this pins the cell->(r, q) mapping.
        planes = encode_position(record(own_bb=bb("A1", "E1", "E9", "I9")))
        assert float((planes[OWN_MARBLES_PLANE] * (1 - VALID_CELL_MASK)).sum()) == 0.0
        assert planes[OWN_MARBLES_PLANE].sum() == 4.0

    def test_all_values_are_zero_or_one_except_the_ply_plane(self):
        planes = encode_position(
            record(own_bb=bb("E5"), opp_bb=bb("A1"), own_losses=3, opp_losses=2, ply=7)
        )
        rest = np.delete(planes, PLY_PLANE, axis=0)
        assert set(np.unique(rest).tolist()) <= {0.0, 1.0}


# ----- group axioms ----------------------------------------------------------


class TestGroupStructure:
    def test_identity_fixes_every_cell(self):
        for c in range(NUM_CELLS):
            if not is_valid_cell(c):
                continue
            assert CELL_PERM[IDENTITY, c] == c

    def test_identity_fixes_every_direction(self):
        for d in range(NUM_DIRS):
            assert DIR_PERM[IDENTITY, d] == d

    def test_identity_fixes_every_move_index(self):
        for idx in range(MOVE_SPACE):
            assert MOVE_PERM[IDENTITY, idx] == idx

    def test_R_has_order_6_on_cells(self):
        for c in range(NUM_CELLS):
            if not is_valid_cell(c):
                continue
            cur = c
            for _ in range(6):
                cur = int(CELL_PERM[R, cur])
            assert cur == c, f"R^6 != identity on cell {c}"

    def test_R_does_not_fix_corners_before_full_period(self):
        # A1 (a corner) cycles through the 6 corners under R; it shouldn't
        # return to itself in fewer than 6 applications.
        a1 = cell("A1")
        cur = a1
        for k in range(1, 6):
            cur = int(CELL_PERM[R, cur])
            assert cur != a1, f"R^{k} unexpectedly fixed A1"

    def test_F_is_involution(self):
        for c in range(NUM_CELLS):
            if not is_valid_cell(c):
                continue
            assert CELL_PERM[F, CELL_PERM[F, c]] == c

    def test_F_is_involution_on_directions(self):
        for d in range(NUM_DIRS):
            assert DIR_PERM[F, DIR_PERM[F, d]] == d

    def test_F_is_involution_on_moves(self):
        for idx in range(MOVE_SPACE):
            assert MOVE_PERM[F, MOVE_PERM[F, idx]] == idx

    def test_all_twelve_elements_distinct_on_generic_cell(self):
        # B3 (q=2, r=1) sits off every reflection axis, so its full orbit
        # under D6 has 12 distinct images.
        b3 = cell("B3")
        images = {int(CELL_PERM[s, b3]) for s in range(NUM_SYMS)}
        assert len(images) == NUM_SYMS

    def test_group_closure_via_action_on_two_cells(self):
        # Group closure: for every (i, j), there's some k such that the
        # composition `S_i ∘ S_j` matches `S_k` on the entire valid set.
        # We probe with two cells whose joint stabilizer is trivial.
        probes = [cell("A1"), cell("B3")]
        for i in range(NUM_SYMS):
            for j in range(NUM_SYMS):
                target = tuple(int(CELL_PERM[i, int(CELL_PERM[j, p])]) for p in probes)
                hit = False
                for k in range(NUM_SYMS):
                    if tuple(int(CELL_PERM[k, p]) for p in probes) == target:
                        hit = True
                        break
                assert hit, f"S_{i} ∘ S_{j} not in the group"

    def test_every_element_has_an_inverse_in_the_group(self):
        for s in range(NUM_SYMS):
            inverse_found = False
            for t in range(NUM_SYMS):
                # Compose s∘t on a probe cell that has trivial stabilizer.
                b3 = cell("B3")
                if int(CELL_PERM[s, int(CELL_PERM[t, b3])]) == b3 and (
                    int(CELL_PERM[t, int(CELL_PERM[s, b3])]) == b3
                ):
                    inverse_found = True
                    break
            assert inverse_found, f"sym {s} has no inverse"

    def test_matrices_have_unit_determinant(self):
        # All D6 elements are isometries: |det| = 1. Rotations det = +1,
        # reflections det = -1.
        for s in range(NUM_SYMS):
            det = int(np.round(np.linalg.det(SYM_MATS[s].astype(float))))
            assert det in (1, -1)
            # First six are rotations, last six reflections.
            expected = 1 if s < 6 else -1
            assert det == expected


# ----- concrete fixed points & orbits ----------------------------------------


class TestConcreteCellAction:
    def test_R_cycles_corners_in_order(self):
        # The 6 corners form a single orbit of size 6 under R.
        corners = ["A1", "A5", "E9", "I9", "I5", "E1"]
        for i, name in enumerate(corners):
            src = cell(name)
            dst = cell(corners[(i + 1) % 6])
            assert CELL_PERM[R, src] == dst, f"R({name}) != {corners[(i + 1) % 6]}"

    def test_F_fixes_NE_SW_diagonal(self):
        # F is the reflection that fixes the line q == r (A1, B2, ..., I9).
        for name in ["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8", "I9"]:
            c = cell(name)
            assert CELL_PERM[F, c] == c, f"F should fix {name}"

    def test_centre_is_fixed_by_every_element(self):
        e5 = cell("E5")
        for s in range(NUM_SYMS):
            assert CELL_PERM[s, e5] == e5, f"sym {s} should fix E5"


class TestConcreteDirectionAction:
    def test_R_cycles_directions(self):
        # Engine direction order: E=0, NE=1, NW=2, W=3, SW=4, SE=5
        cycle = [0, 1, 2, 3, 4, 5]
        for i in range(6):
            assert DIR_PERM[R, cycle[i]] == cycle[(i + 1) % 6]

    def test_F_swaps_E_and_NW_and_W_and_SE(self):
        # F (the reflection through the NE-SW axis) swaps E↔NW, W↔SE,
        # and fixes NE, SW.
        E, NE, NW, W_, SW, SE = 0, 1, 2, 3, 4, 5
        assert DIR_PERM[F, E] == NW
        assert DIR_PERM[F, NW] == E
        assert DIR_PERM[F, W_] == SE
        assert DIR_PERM[F, SE] == W_
        assert DIR_PERM[F, NE] == NE
        assert DIR_PERM[F, SW] == SW


# ----- move-index permutation ------------------------------------------------


class TestMovePerm:
    def test_inline_stays_inline(self):
        # Indices < BROADSIDE_OFFSET map back to indices < BROADSIDE_OFFSET.
        for s in range(NUM_SYMS):
            for idx in range(BROADSIDE_OFFSET):
                assert MOVE_PERM[s, idx] < BROADSIDE_OFFSET, (
                    f"sym {s} sent inline {idx} to broadside {MOVE_PERM[s, idx]}"
                )

    def test_broadside_stays_broadside(self):
        for s in range(NUM_SYMS):
            for idx in range(BROADSIDE_OFFSET, MOVE_SPACE):
                assert MOVE_PERM[s, idx] >= BROADSIDE_OFFSET, (
                    f"sym {s} sent broadside {idx} to inline {MOVE_PERM[s, idx]}"
                )

    def test_R_has_order_6_on_moves(self):
        for idx in range(MOVE_SPACE):
            cur = idx
            for _ in range(6):
                cur = int(MOVE_PERM[R, cur])
            assert cur == idx

    def test_size_is_preserved(self):
        # The size of an inline or broadside move is invariant under
        # symmetry — only the anchor and direction(s) change.
        for s in range(NUM_SYMS):
            for idx in range(MOVE_SPACE):
                if MOVE_PERM[s, idx] == idx:
                    # Identity-mapped (e.g., structurally invalid encoding
                    # or s == identity).
                    continue
                if idx < BROADSIDE_OFFSET:
                    _, _, size_before = decode_inline(idx)
                    _, _, size_after = decode_inline(int(MOVE_PERM[s, idx]))
                else:
                    _, _, _, size_before = decode_broadside(idx)
                    _, _, _, size_after = decode_broadside(int(MOVE_PERM[s, idx]))
                assert size_before == size_after

    def test_inline_at_centre_with_E_dir_round_trips_with_R(self):
        # An inline-1 move at E5 in direction E maps to E5 in direction NE
        # under R (centre is fixed; E rotates to NE).
        e5 = cell("E5")
        idx_before = encode_inline(e5, dir_idx=0, size=1)  # E5, E, size 1
        idx_after = encode_inline(e5, dir_idx=1, size=1)  # E5, NE, size 1
        assert MOVE_PERM[R, idx_before] == idx_after


# ----- plane permutation -----------------------------------------------------


class TestPlanePerm:
    def _empty_planes(self) -> np.ndarray:
        return np.zeros((NUM_INPUT_CHANNELS, 9, 9), dtype=np.float32)

    def test_identity_is_noop(self):
        rng = np.random.default_rng(0)
        planes = rng.random((NUM_INPUT_CHANNELS, 9, 9), dtype=np.float32) * VALID_CELL_MASK
        out = apply_sym_to_planes(planes, IDENTITY)
        np.testing.assert_array_equal(out, planes)

    def test_centre_marble_is_invariant(self):
        planes = self._empty_planes()
        planes[OWN_MARBLES_PLANE] = bb_to_plane(1 << cell("E5"))
        for s in range(NUM_SYMS):
            out = apply_sym_to_planes(planes, s)
            np.testing.assert_array_equal(
                out[OWN_MARBLES_PLANE],
                planes[OWN_MARBLES_PLANE],
                f"sym {s} should not move the E5 marble",
            )

    def test_A1_marble_lands_at_A5_under_R(self):
        planes = self._empty_planes()
        planes[OWN_MARBLES_PLANE] = bb_to_plane(1 << cell("A1"))
        out = apply_sym_to_planes(planes, R)
        # A1 → A5 under R
        expected = bb_to_plane(1 << cell("A5"))
        np.testing.assert_array_equal(out[OWN_MARBLES_PLANE], expected)

    def test_both_marble_planes_permute_by_CELL_PERM(self):
        """The two marble planes are the only spatial channels, and each
        must follow `CELL_PERM` independently — the own/opp split survives
        augmentation untouched."""
        planes = self._empty_planes()
        own_cells = ["A1", "B3", "E5", "I9"]
        opp_cells = ["A5", "C4", "G7"]
        planes[OWN_MARBLES_PLANE] = bb_to_plane(bb(*own_cells))
        planes[OPP_MARBLES_PLANE] = bb_to_plane(bb(*opp_cells))
        for s in range(NUM_SYMS):
            out = apply_sym_to_planes(planes, s)
            for ch, cells in (
                (OWN_MARBLES_PLANE, own_cells),
                (OPP_MARBLES_PLANE, opp_cells),
            ):
                expected = bb_to_plane(
                    int(np.bitwise_or.reduce([1 << int(CELL_PERM[s, cell(n)]) for n in cells]))
                )
                np.testing.assert_array_equal(
                    out[ch], expected, f"sym {s} mispermuted marble plane {ch}"
                )
            # The channel axis itself is never touched: own stays own.
            assert out[OWN_MARBLES_PLANE].sum() == len(own_cells)
            assert out[OPP_MARBLES_PLANE].sum() == len(opp_cells)

    def test_constant_planes_are_invariant_on_board(self):
        # Channels 2–13 (the two loss thermometers, ply, valid mask) are
        # constants broadcast across the board. Sym is defined only over the
        # 61 valid cells, so the test asserts invariance restricted to the
        # on-board mask — off-board entries in the output are zeroed by the
        # sym op regardless of input, which is fine because the network has
        # the valid-mask channel and never relies on off-board values.
        planes = self._empty_planes()
        for i, ch in enumerate(CONSTANT_PLANES[:-1]):
            planes[ch].fill(0.05 * (i + 1))
        planes[VALID_MASK_PLANE] = VALID_CELL_MASK
        for s in range(NUM_SYMS):
            out = apply_sym_to_planes(planes, s)
            for ch in CONSTANT_PLANES:
                np.testing.assert_array_equal(
                    out[ch] * VALID_CELL_MASK,
                    planes[ch] * VALID_CELL_MASK,
                    f"sym {s} disturbed constant channel {ch} on the board",
                )

    def test_real_encoded_position_keeps_its_constant_planes_under_sym(self):
        """End-to-end: a position straight out of `encode_position` must
        come back from any symmetry with the same counters and ply. Only the
        board rotates; the game state does not."""
        planes = encode_position(
            record(
                own_bb=bb("A1", "B3", "E5"),
                opp_bb=bb("I9", "G7"),
                own_losses=4,
                opp_losses=1,
                ply=50,
                max_plies=200,
            )
        )
        for s in range(NUM_SYMS):
            out = apply_sym_to_planes(planes, s)
            for ch in CONSTANT_PLANES:
                np.testing.assert_array_equal(
                    out[ch] * VALID_CELL_MASK,
                    planes[ch] * VALID_CELL_MASK,
                    f"sym {s} changed channel {ch}",
                )
            # Material is conserved even though the marbles moved.
            assert out[OWN_MARBLES_PLANE].sum() == 3
            assert out[OPP_MARBLES_PLANE].sum() == 2
            if s != IDENTITY:
                assert not np.array_equal(
                    out[OWN_MARBLES_PLANE], planes[OWN_MARBLES_PLANE]
                ), f"sym {s} left this asymmetric position unchanged"

    def test_six_R_applications_are_identity(self):
        rng = np.random.default_rng(1)
        planes = rng.random((NUM_INPUT_CHANNELS, 9, 9), dtype=np.float32) * VALID_CELL_MASK
        cur = planes.copy()
        for _ in range(6):
            cur = apply_sym_to_planes(cur, R)
        np.testing.assert_allclose(cur, planes)

    def test_F_squared_is_identity(self):
        rng = np.random.default_rng(2)
        planes = rng.random((NUM_INPUT_CHANNELS, 9, 9), dtype=np.float32) * VALID_CELL_MASK
        out = apply_sym_to_planes(apply_sym_to_planes(planes, F), F)
        np.testing.assert_allclose(out, planes)

    def test_marble_count_is_preserved(self):
        # Sym is a permutation of valid cells; the total number of set
        # marbles is invariant under any sym.
        rng = np.random.default_rng(3)
        board = int(rng.integers(0, 1 << 60)) & ((1 << 81) - 1)
        planes = self._empty_planes()
        planes[OWN_MARBLES_PLANE] = bb_to_plane(board)
        before = int((planes[OWN_MARBLES_PLANE] * VALID_CELL_MASK).sum())
        for s in range(NUM_SYMS):
            out = apply_sym_to_planes(planes, s)
            after = int((out[OWN_MARBLES_PLANE] * VALID_CELL_MASK).sum())
            assert after == before, f"sym {s}: {after} != {before}"


# ----- policy permutation ----------------------------------------------------


class TestPolicyPerm:
    def test_identity_is_noop(self):
        rng = np.random.default_rng(0)
        policy = rng.random((MOVE_SPACE,), dtype=np.float32)
        out = apply_sym_to_policy(policy, IDENTITY)
        np.testing.assert_array_equal(out, policy)

    def test_mass_on_move_lands_on_permuted_index(self):
        # One-hot mass on idx i; after sym s, mass should be on MOVE_PERM[s, i].
        for src in [0, 100, 500, 1000, BROADSIDE_OFFSET, BROADSIDE_OFFSET + 50, MOVE_SPACE - 1]:
            policy = np.zeros((MOVE_SPACE,), dtype=np.float32)
            policy[src] = 1.0
            for s in range(NUM_SYMS):
                out = apply_sym_to_policy(policy, s)
                expected = int(MOVE_PERM[s, src])
                assert out[expected] == pytest.approx(1.0), (
                    f"sym {s}, src {src}: expected mass at {expected}"
                )
                # Total mass is conserved.
                assert out.sum() == pytest.approx(1.0)

    def test_total_mass_preserved(self):
        rng = np.random.default_rng(0)
        policy = rng.random((MOVE_SPACE,), dtype=np.float32)
        original_sum = float(policy.sum())
        for s in range(NUM_SYMS):
            out = apply_sym_to_policy(policy, s)
            assert float(out.sum()) == pytest.approx(original_sum, rel=1e-5)

    def test_six_R_applications_are_identity(self):
        rng = np.random.default_rng(0)
        policy = rng.random((MOVE_SPACE,), dtype=np.float32)
        cur = policy.copy()
        for _ in range(6):
            cur = apply_sym_to_policy(cur, R)
        np.testing.assert_allclose(cur, policy)

    def test_F_squared_is_identity(self):
        rng = np.random.default_rng(0)
        policy = rng.random((MOVE_SPACE,), dtype=np.float32)
        out = apply_sym_to_policy(apply_sym_to_policy(policy, F), F)
        np.testing.assert_allclose(out, policy)

    def test_batched_application(self):
        # apply_sym_to_policy supports leading batch dims by spec.
        rng = np.random.default_rng(0)
        policy = rng.random((4, MOVE_SPACE), dtype=np.float32)
        for s in range(NUM_SYMS):
            out = apply_sym_to_policy(policy, s)
            for b in range(4):
                np.testing.assert_array_equal(out[b], apply_sym_to_policy(policy[b], s))


# ----- consistency between cell perm, dir perm, move perm --------------------


class TestPermConsistency:
    def test_inline_perm_via_cell_and_dir_matches_move_perm(self):
        # For an inline move at a specific (anchor, dir, size), the result
        # of decoding, perming the anchor and dir, and re-encoding should
        # equal MOVE_PERM[s, idx].
        e5 = cell("E5")
        for size in (1, 2, 3):
            for d in range(6):
                idx = encode_inline(e5, d, size)
                for s in range(NUM_SYMS):
                    expected_anchor = int(CELL_PERM[s, e5])
                    expected_dir = int(DIR_PERM[s, d])
                    expected = encode_inline(expected_anchor, expected_dir, size)
                    assert MOVE_PERM[s, idx] == expected, (
                        f"sym {s}, anchor E5, dir {d}, size {size}: "
                        f"got {MOVE_PERM[s, idx]}, expected {expected}"
                    )

    def test_broadside_at_centre_perm_matches_cell_and_dir(self):
        # Broadside at E5: pick group_dir E and a sideways move_dir.
        e5 = cell("E5")
        gd = 0  # E (in POSITIVE_DIRS)
        md = BROADSIDE_DIRS[POSITIVE_DIRS.index(gd)][0]  # NE
        size = 2
        idx = encode_broadside(e5, gd, md, size)
        for s in range(NUM_SYMS):
            target = int(MOVE_PERM[s, idx])
            anchor_t, gd_t, md_t, size_t = decode_broadside(target)
            assert size_t == size
            # Move dir under sym
            assert md_t == int(DIR_PERM[s, md])
            # The anchor of the new broadside is one of the two cells of
            # the rotated 2-line, specifically the one with positive
            # group_dir orientation.
            new_gd_naive = int(DIR_PERM[s, gd])
            if new_gd_naive in POSITIVE_DIRS:
                # No re-anchoring; anchor is sym(e5).
                assert anchor_t == int(CELL_PERM[s, e5])
                assert gd_t == new_gd_naive
            else:
                # Re-anchored at the other end. Group_dir flipped.
                assert gd_t == ((new_gd_naive + 3) % 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
