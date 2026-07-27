//! Emit cross-language conformance fixtures as JSON.
//!
//! The plane encoding is duplicated in Rust (`selfplay::encoder`) and Python
//! (`model/encoder.py`). A silent divergence between them — swapped capture
//! planes — was live for three generations and survived a 43-test Python
//! suite, because nothing ever compared the two implementations against each
//! other. This binary is half of the guard that makes that impossible:
//! it writes the Rust encoder's output for a broad, *deterministic* set of
//! positions, and `tests/test_conformance.py` asserts that Python reproduces
//! every float exactly.
//!
//! Coverage is deliberate, not incidental (see `positions()`):
//!   * both standard openings,
//!   * the full 6×6 handicap grid for both sides, both openings — including
//!     asymmetric seeds like (0, 4) and (5, 1) that a symmetric fixture set
//!     would let a plane swap slip through,
//!   * positions with each side to move,
//!   * seeded random playouts sampled at varied plies,
//!   * near-terminal positions,
//!   * several distinct `max_plies` caps plus a past-the-cap position, so the
//!     ply normaliser is pinned rather than assumed.
//!
//! Everything is driven by a FIXED seed, so re-running reproduces the file
//! byte-for-byte and a fixture diff means a real encoder change.
//!
//! Run:
//!   cargo run --release -p abalone-selfplay --bin dump-golden -- <out.json>

use std::path::PathBuf;

use abalone_game::{
    encode, Game, Opening, Side, DEFAULT_MAX_PLIES, MOVE_SPACE, NO_PROGRESS_DISABLED,
};
use abalone_selfplay::encoder::{encode_planes, NUM_INPUT_CHANNELS, PLANE_SIZE};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

/// Fixed so the fixture file is reproducible.
const SEED: u64 = 0x6011_1DEA_C0FF_EE01;
/// Guard rail: the conformance test is only as good as its coverage.
const MIN_RECORDS: usize = 200;

#[derive(Serialize)]
struct Record {
    /// Human-readable provenance, so a conformance failure names the case.
    label: String,
    /// Side-to-move-relative bitboards, as the u64 halves of the u128.
    own_bb_lo: u64,
    own_bb_hi: u64,
    opp_bb_lo: u64,
    opp_bb_hi: u64,
    /// Marbles each side has **LOST** (had pushed off the board) — named for
    /// losses, never for who did the pushing. `Board::pushed_off[s]` holds the
    /// opposite (opponent marbles pushed off *by* `s`); reading that field as
    /// "s's losses" is the exact mistake this fixture set exists to catch.
    black_losses: u8,
    white_losses: u8,
    /// 0 = Black to move, 1 = White to move.
    turn: u8,
    ply: u32,
    max_plies: u32,
    /// The flattened `(14, 9, 9)` encoder output: 1134 floats.
    planes: Vec<f32>,
    /// Every legal move at this position, as flat move indices, sorted.
    legal_move_indices: Vec<u16>,
}

#[derive(Serialize)]
struct Fixture {
    /// Bump when the record schema changes so a stale file fails loudly.
    schema_version: u32,
    num_input_channels: usize,
    board_h: usize,
    board_w: usize,
    plane_size: usize,
    move_space: usize,
    seed: u64,
    positions: Vec<Record>,
}

struct Builder {
    records: Vec<Record>,
    rng: SmallRng,
}

impl Builder {
    fn new() -> Self {
        Self {
            records: Vec::new(),
            rng: SmallRng::seed_from_u64(SEED),
        }
    }

    fn push(&mut self, label: impl Into<String>, g: &Game) {
        let mut planes = vec![0f32; PLANE_SIZE];
        encode_planes(g, &mut planes);

        let own = g.board.bb(g.turn);
        let opp = g.board.bb(g.turn.other());

        let mut legal: Vec<u16> = g.legal_moves().iter().map(|&m| encode(m)).collect();
        legal.sort_unstable();
        let before = legal.len();
        legal.dedup();
        assert_eq!(before, legal.len(), "move encoding collided on legal moves");

        self.records.push(Record {
            label: label.into(),
            own_bb_lo: own as u64,
            own_bb_hi: (own >> 64) as u64,
            opp_bb_lo: opp as u64,
            opp_bb_hi: (opp >> 64) as u64,
            // `lost(side)` does the `pushed_off` flip for us.
            black_losses: g.board.lost(Side::Black),
            white_losses: g.board.lost(Side::White),
            turn: g.turn.idx() as u8,
            ply: g.ply,
            max_plies: g.max_plies,
            planes,
            legal_move_indices: legal,
        });
    }

    /// Apply one uniformly-random legal move. Returns false at a terminal.
    fn step(&mut self, g: &mut Game) -> bool {
        if g.is_terminal() {
            return false;
        }
        let moves = g.legal_moves();
        if moves.is_empty() {
            return false;
        }
        let pick = self.rng.gen_range(0..moves.len());
        g.apply(moves[pick]);
        true
    }
}

fn opening_name(o: Opening) -> &'static str {
    match o {
        Opening::Standard => "standard",
        Opening::BelgianDaisy => "belgian",
    }
}

fn positions() -> Vec<Record> {
    let mut b = Builder::new();

    // --- 1. the two openings, untouched -------------------------------------
    b.push("standard-opening", &Game::new_standard());
    b.push("belgian-daisy-opening", &Game::new_belgian_daisy());

    // --- 2. the full handicap grid, both openings, both sides to move -------
    // 0..=5 on each axis independently, so the asymmetric seeds — (0,4),
    // (5,1), (4,0) — are all present. Under a plane swap these are exactly
    // the records that differ, which is the point.
    for &opening in &[Opening::Standard, Opening::BelgianDaisy] {
        for black_lost in 0..=5u8 {
            for white_lost in 0..=5u8 {
                let g = Game::with_handicap(
                    opening,
                    DEFAULT_MAX_PLIES,
                    NO_PROGRESS_DISABLED,
                    black_lost,
                    white_lost,
                    &mut b.rng,
                );
                let name = opening_name(opening);
                b.push(
                    format!("handicap-{name}-b{black_lost}-w{white_lost}-black-to-move"),
                    &g,
                );
                // One ply flips the side to move, so every handicap pair is
                // also seen from White's point of view.
                let mut g2 = g;
                if b.step(&mut g2) {
                    assert_eq!(g2.turn, Side::White);
                    b.push(
                        format!("handicap-{name}-b{black_lost}-w{white_lost}-white-to-move"),
                        &g2,
                    );
                }
            }
        }
    }

    // --- 3. varied ply caps ------------------------------------------------
    // The ply normaliser is `ply / max_plies`. Sampling several caps pins the
    // division itself, not just the numerator.
    for &cap in &[DEFAULT_MAX_PLIES, 100, 60, 137, 7] {
        let mut g = Game::new(Opening::BelgianDaisy, cap, NO_PROGRESS_DISABLED);
        for ply in 0..10 {
            b.push(format!("cap{cap}-ply{ply}"), &g);
            if !b.step(&mut g) {
                break;
            }
        }
    }

    // --- 4. seeded random playouts, sampled at varied plies -----------------
    for game_id in 0..14u32 {
        let opening = if game_id % 2 == 0 {
            Opening::Standard
        } else {
            Opening::BelgianDaisy
        };
        let cap = [DEFAULT_MAX_PLIES, 120, 240][(game_id % 3) as usize];
        let a = b.rng.gen_range(0..=5u8);
        let w = b.rng.gen_range(0..=5u8);
        let mut g =
            Game::with_handicap(opening, cap, NO_PROGRESS_DISABLED, a, w, &mut b.rng);
        // Sample every few plies at a game-dependent stride so the recorded
        // plies are not all multiples of one number.
        let stride = 3 + (game_id % 5);
        let mut ply = 0u32;
        loop {
            if ply.is_multiple_of(stride) {
                b.push(format!("playout{game_id}-ply{ply}"), &g);
            }
            if !b.step(&mut g) {
                break;
            }
            ply += 1;
        }
        // The terminal position itself.
        b.push(format!("playout{game_id}-terminal-ply{}", g.ply), &g);
    }

    // --- 5. near-terminal positions ----------------------------------------
    // At (5, 5) the very next capture ends the game, so the last few plies
    // before termination are genuinely near-terminal rather than merely late.
    for trial in 0..8u32 {
        let mut g = Game::with_handicap(
            Opening::BelgianDaisy,
            DEFAULT_MAX_PLIES,
            NO_PROGRESS_DISABLED,
            5,
            5,
            &mut b.rng,
        );
        let mut recent: Vec<Game> = Vec::new();
        loop {
            recent.push(g);
            if recent.len() > 3 {
                recent.remove(0);
            }
            if !b.step(&mut g) {
                break;
            }
        }
        for (i, snap) in recent.iter().enumerate() {
            b.push(format!("near-terminal{trial}-minus{}", recent.len() - i), snap);
        }
        b.push(format!("near-terminal{trial}-final"), &g);
    }

    // --- 6. past the cap, and a degenerate cap ------------------------------
    // `ply > max_plies` must saturate at 1.0 in both encoders, and a zero cap
    // must not produce a NaN. Reached by hand because normal play cannot.
    let mut over = Game::new(Opening::Standard, 40, NO_PROGRESS_DISABLED);
    over.ply = 137;
    b.push("ply-past-the-cap", &over);
    let mut zero_cap = Game::new(Opening::BelgianDaisy, 0, NO_PROGRESS_DISABLED);
    zero_cap.ply = 5;
    b.push("zero-cap", &zero_cap);

    b.records
}

fn out_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" | "-o" => {
                path = Some(PathBuf::from(
                    args.next().expect("--out requires a path argument"),
                ))
            }
            "-h" | "--help" => {
                eprintln!("usage: dump-golden [--out] <path.json>");
                std::process::exit(0);
            }
            other => path = Some(PathBuf::from(other)),
        }
    }
    path.unwrap_or_else(|| {
        eprintln!("usage: dump-golden [--out] <path.json>");
        std::process::exit(2);
    })
}

fn main() {
    let path = out_path();
    let positions = positions();
    assert!(
        positions.len() >= MIN_RECORDS,
        "fixture coverage regressed: {} records, expected at least {MIN_RECORDS}",
        positions.len()
    );
    for r in &positions {
        assert_eq!(r.planes.len(), PLANE_SIZE);
        assert!(r.legal_move_indices.iter().all(|&i| (i as usize) < MOVE_SPACE));
    }

    let fixture = Fixture {
        schema_version: 1,
        num_input_channels: NUM_INPUT_CHANNELS,
        board_h: 9,
        board_w: 9,
        plane_size: PLANE_SIZE,
        move_space: MOVE_SPACE,
        seed: SEED,
        positions,
    };

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("could not create output directory");
        }
    }
    let json = serde_json::to_string(&fixture).expect("serialising fixtures failed");
    std::fs::write(&path, json).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    eprintln!(
        "dump-golden: wrote {} positions to {}",
        fixture.positions.len(),
        path.display()
    );
}
