//! Measure what the browser's game review is actually measuring.
//!
//! The review panel grades a played move by how much it cost. There are two
//! ways to compute that and they are not the same number:
//!
//!   * **within one search** — the played move's Q against the best move's Q,
//!     both read off the same tree. Zero, by definition, when the move played
//!     was the search's own pick.
//!   * **across the move** — the root eval before against the root eval after,
//!     two independent searches of two different positions.
//!
//! The panel shipped with the second one, in `P(win)` rather than expected
//! score, and produced a column of small negatives on *every* move including
//! the ones it labelled best. This binary exists to say which part of that was
//! draw-mass bookkeeping, which part was the search failing to converge, and
//! which part was real — by measuring, not by reasoning about the source.
//!
//! It plays a game and then sweeps every position it passed through, exactly as
//! `ReviewView` does: same simulation count, same `c_puct`, same batch size,
//! same per-ply seed, `track_outcome_stats` on. One JSON line per position, to
//! stdout or `--out`; analysis is left to the reader.
//!
//! ```text
//! review-probe --model web/public/models/best.onnx \
//!              --review-sims 200 --games 2 --out /tmp/probe.jsonl
//! ```
//!
//! `--review-sims` may be repeated: every sweep is run at each count, from one
//! game record, which is the convergence test. If the cost of a best move falls
//! toward zero as simulations rise, the number the panel was showing was the
//! search's horizon moving, not the player's mistake.
//!
//! `ABALONE_USE_COREML=1` is strongly advised — a 3200-simulation sweep is
//! minutes on the ANE and most of an hour on one CPU thread.

use std::io::Write;
use std::path::PathBuf;

use abalone_game::{Game, GameState, Move, Opening, Side};
use abalone_mcts::{Search, SearchConfig};
use abalone_selfplay::ort_eval::{use_coreml, OrtEvaluator};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Everything one searched position yields. Mirrors `PlyRead` in
/// `web/lib/engine/review.ts` plus the per-move detail the panel throws away.
struct Read {
    ply: usize,
    turn: Side,
    /// Q of the most-visited root child, White's POV. The number the graph plots.
    root_eval_white: f32,
    /// Visit-weighted `(win, draw, loss)` of that same child, White's POV.
    root_wdl_white: [f32; 3],
    best: Move,
    best_visits: u32,
    total_visits: u32,
    /// Q and visits of every legal move, White's POV.
    per_move: Vec<(Move, f32, u32)>,
}

impl Read {
    fn q_of(&self, mv: Move) -> Option<(f32, u32)> {
        self.per_move
            .iter()
            .find(|(m, _, _)| *m == mv)
            .map(|&(_, q, n)| (q, n))
    }
}

/// The browser's config, restated. Any drift here and the numbers stop being
/// about the panel: `crates/wasm/src/lib.rs::begin_search` is the original.
fn review_config(sims: u32, batch: usize) -> SearchConfig {
    SearchConfig {
        simulations: sims,
        c_puct: 1.4,
        batch_size: batch,
        dirichlet_eps: 0.0,
        track_outcome_stats: true,
        ..Default::default()
    }
}

/// The worker's per-position seed, so a position searched here and in the
/// browser builds the same tree: `(plies * 2654435761 + simulations) >>> 0`.
fn worker_seed(plies: usize, sims: u32) -> u64 {
    let mixed = (plies as u64)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(sims as u64);
    u64::from(mixed as u32)
}

fn analyse(
    game: &Game,
    cfg: &SearchConfig,
    seed: u64,
    ev: &mut OrtEvaluator,
    ply: usize,
) -> Option<Read> {
    let mut s = Search::begin(game, cfg, seed);
    loop {
        // Copied out of the search before calling the evaluator: `next_batch`
        // borrows the tree, and the tree is what `submit_with_stats` mutates.
        let batch: Vec<Game> = s.next_batch().to_vec();
        if batch.is_empty() {
            break;
        }
        let (evals, wdl) = ev
            .evaluate_batch_with_wdl(&batch)
            .expect("evaluate review batch");
        // The margin head is a readout this probe does not use; the search
        // ignores it either way, and `submit_with_stats` only requires the
        // slice be parallel.
        let margins = vec![0.0f32; evals.len()];
        s.submit_with_stats(&evals, &wdl, &margins);
    }
    let res = s.result()?;
    let to_white = if game.turn == Side::White { 1.0 } else { -1.0 };
    let best_slot = res
        .visits
        .iter()
        .position(|&(mv, _)| mv == res.best)
        .expect("best move is a root child");
    let flip = |w: [f32; 3]| {
        if to_white > 0.0 {
            w
        } else {
            [w[2], w[1], w[0]]
        }
    };
    Some(Read {
        ply,
        turn: game.turn,
        root_eval_white: res.q_parent_pov[best_slot] * to_white,
        root_wdl_white: flip(res.wdl_parent_pov[best_slot]),
        best: res.best,
        best_visits: res.visits[best_slot].1,
        total_visits: res.visits.iter().map(|&(_, n)| n).sum(),
        per_move: res
            .visits
            .iter()
            .zip(res.q_parent_pov.iter())
            .map(|(&(mv, n), &q)| (mv, q * to_white, n))
            .collect(),
    })
}

/// Play one game. `weak` plays a move sampled from the root visit counts rather
/// than the argmax, which is a stand-in for the human on the other side of the
/// games this panel is built to review — a record of nothing but engine picks
/// would answer a narrower question than the one asked.
fn play_game(
    opening: Opening,
    sims: u32,
    batch: usize,
    weak: Option<Side>,
    temperature: f32,
    max_plies: usize,
    ev: &mut OrtEvaluator,
    rng: &mut SmallRng,
) -> (Game, Vec<Move>) {
    let start = Game::new(opening, 300, u32::MAX);
    let mut g = start;
    let mut moves = Vec::new();
    let cfg = review_config(sims, batch);
    while moves.len() < max_plies && g.state() == GameState::InProgress {
        let seed = rng.gen();
        let Some(read) = analyse(&g, &cfg, seed, ev, moves.len()) else {
            break;
        };
        let mv = if weak == Some(g.turn) {
            sample_by_visits(&read, temperature, rng)
        } else {
            read.best
        };
        g.apply(mv);
        moves.push(mv);
    }
    (start, moves)
}

fn sample_by_visits(read: &Read, temperature: f32, rng: &mut SmallRng) -> Move {
    let weights: Vec<f64> = read
        .per_move
        .iter()
        .map(|&(_, _, n)| (n as f64).powf(1.0 / temperature as f64))
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return read.best;
    }
    let mut t = rng.gen_range(0.0..total);
    for (i, w) in weights.iter().enumerate() {
        t -= w;
        if t <= 0.0 {
            return read.per_move[i].0;
        }
    }
    read.best
}

struct Args {
    model: PathBuf,
    review_sims: Vec<u32>,
    play_sims: u32,
    batch: usize,
    games: usize,
    max_plies: usize,
    temperature: f32,
    seed: u64,
    opening: Opening,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("web/public/models/best.onnx"),
        review_sims: Vec::new(),
        play_sims: 200,
        batch: 16,
        games: 1,
        max_plies: 120,
        temperature: 1.0,
        seed: 20260729,
        opening: Opening::Standard,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().expect("flag expects a value");
        match flag.as_str() {
            "--model" => a.model = PathBuf::from(val()),
            "--review-sims" => a.review_sims.push(val().parse().expect("--review-sims")),
            "--play-sims" => a.play_sims = val().parse().expect("--play-sims"),
            "--batch" => a.batch = val().parse().expect("--batch"),
            "--games" => a.games = val().parse().expect("--games"),
            "--max-plies" => a.max_plies = val().parse().expect("--max-plies"),
            "--temperature" => a.temperature = val().parse().expect("--temperature"),
            "--seed" => a.seed = val().parse().expect("--seed"),
            "--belgian" => a.opening = Opening::BelgianDaisy,
            "--out" => a.out = Some(PathBuf::from(val())),
            other => panic!("unknown flag {other}"),
        }
    }
    if a.review_sims.is_empty() {
        a.review_sims.push(200);
    }
    a
}

fn main() {
    let args = parse_args();
    let mut ev = OrtEvaluator::from_onnx(&args.model).expect("load model");
    // CoreML compiles one shape-specialised graph per input width, so an
    // unpadded search spends its life recompiling. See `set_fixed_batch`.
    if use_coreml() {
        ev.set_fixed_batch(Some(args.batch));
    }

    let mut sink: Box<dyn Write> = match &args.out {
        Some(p) => Box::new(std::fs::File::create(p).expect("create --out")),
        None => Box::new(std::io::stdout()),
    };

    for game_idx in 0..args.games {
        let mut rng = SmallRng::seed_from_u64(args.seed ^ (game_idx as u64) << 32);
        // Whichever side is not the engine's own picks alternates, so a
        // systematic effect cannot hide as a property of one colour.
        let weak = if game_idx % 2 == 0 {
            Some(Side::Black)
        } else {
            Some(Side::White)
        };
        let (start, moves) = play_game(
            args.opening,
            args.play_sims,
            args.batch,
            weak,
            args.temperature,
            args.max_plies,
            &mut ev,
            &mut rng,
        );
        eprintln!(
            "game {game_idx}: {} plies, weak side {:?}",
            moves.len(),
            weak
        );

        for &sims in &args.review_sims {
            let cfg = review_config(sims, args.batch);
            for ply in 0..=moves.len() {
                let mut g = start;
                for &mv in &moves[..ply] {
                    g.apply(mv);
                }
                if g.state() != GameState::InProgress {
                    break;
                }
                let Some(read) = analyse(&g, &cfg, worker_seed(ply, sims), &mut ev, ply) else {
                    break;
                };
                let played = moves.get(ply).copied();
                let played_q = played.and_then(|mv| read.q_of(mv));
                writeln!(
                    sink,
                    "{}",
                    json_line(game_idx, sims, weak, &read, played, played_q)
                )
                .expect("write");
                if ply % 10 == 0 {
                    eprintln!("  sims {sims} ply {ply}/{}", moves.len());
                }
            }
            sink.flush().expect("flush");
        }
    }
}

fn json_line(
    game: usize,
    sims: u32,
    weak: Option<Side>,
    r: &Read,
    played: Option<Move>,
    played_q: Option<(f32, u32)>,
) -> String {
    let side = if r.turn == Side::White { 1 } else { 0 };
    let played_str = played.map(|m| m.to_string()).unwrap_or_default();
    let (pq, pn) = played_q.unwrap_or((f32::NAN, 0));
    format!(
        concat!(
            r#"{{"game":{},"sims":{},"weak_is_white":{},"ply":{},"side":{},"#,
            r#""root_eval_white":{:.6},"win_white":{:.6},"draw":{:.6},"loss_white":{:.6},"#,
            r#""best":"{}","best_visits":{},"total_visits":{},"#,
            r#""played":"{}","played_is_best":{},"played_q_white":{},"played_visits":{}}}"#
        ),
        game,
        sims,
        weak == Some(Side::White),
        r.ply,
        side,
        r.root_eval_white,
        r.root_wdl_white[0],
        r.root_wdl_white[1],
        r.root_wdl_white[2],
        r.best,
        r.best_visits,
        r.total_visits,
        played_str,
        played == Some(r.best),
        if pq.is_nan() {
            "null".to_string()
        } else {
            format!("{pq:.6}")
        },
        pn,
    )
}
