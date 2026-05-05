//! Binary shard format for self-play trajectories. Tuned for fast Python
//! ingest via `numpy.fromfile` and explicit struct unpacking — not a
//! general serialization format.
//!
//! # Layout
//!
//! Each shard begins with a 16-byte header:
//!
//! ```text
//! [0..4]   "ABSP"  -- magic
//! [4..8]   u32     -- version (currently 1)
//! [8..12]  u32     -- simulations per move (sanity check; see entry.n_visits)
//! [12..16] u32     -- reserved (0)
//! ```
//!
//! Then a sequence of records. Each record is one self-play position:
//!
//! ```text
//! offset   bytes  field
//! 0        16     own_bb       u128 (LE) — bitboard of side-to-move
//! 16       16     opp_bb       u128 (LE) — bitboard of opponent
//! 32       1      pushed_off_black
//! 33       1      pushed_off_white
//! 34       1      turn         (0 = Black, 1 = White)
//! 35       1      _pad
//! 36       2      ply          u16
//! 38       2      move_played  u16 (encoded move idx, 0..2562)
//! 40       2      n_children   u16
//! 42       2      _pad2
//! 44       4      z            f32 (final outcome from this POV)
//! 48       4      q            f32 (visit-weighted root Q from this POV)
//! 52       8 * n_children      ChildVisit array:
//!                                 [u16 move_idx, u16 _pad, u32 visits]
//! ```
//!
//! Records are not aligned across record boundaries; the reader reads
//! the 52-byte header first, then `n_children * 8` more bytes.

use std::io::{BufWriter, Write};
use std::path::Path;

use abalone_game::Side;

use crate::{GameOutcome, TrajectoryEntry};

pub const SHARD_MAGIC: &[u8; 4] = b"ABSP";
pub const SHARD_VERSION: u32 = 1;
pub const SHARD_HEADER_BYTES: usize = 16;
pub const ENTRY_FIXED_BYTES: usize = 52;
pub const CHILD_VISIT_BYTES: usize = 8;

pub struct ShardWriter {
    out: BufWriter<std::fs::File>,
    pub bytes_written: u64,
    pub games_written: u32,
    pub entries_written: u64,
}

impl ShardWriter {
    pub fn create<P: AsRef<Path>>(path: P, simulations: u32) -> std::io::Result<Self> {
        let f = std::fs::File::create(path.as_ref())?;
        let mut out = BufWriter::new(f);
        out.write_all(SHARD_MAGIC)?;
        out.write_all(&SHARD_VERSION.to_le_bytes())?;
        out.write_all(&simulations.to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?;
        Ok(Self {
            out,
            bytes_written: SHARD_HEADER_BYTES as u64,
            games_written: 0,
            entries_written: 0,
        })
    }

    pub fn write_game(&mut self, outcome: &GameOutcome) -> std::io::Result<()> {
        for entry in &outcome.trajectory {
            let z = outcome.z_for(entry);
            self.write_entry(entry, z)?;
        }
        self.games_written += 1;
        Ok(())
    }

    fn write_entry(&mut self, e: &TrajectoryEntry, z: f32) -> std::io::Result<()> {
        let g = &e.state;
        let own_bb = g.board.bb(g.turn);
        let opp_bb = g.board.bb(g.turn.other());
        let pushed_off_black = g.board.pushed_off[Side::Black.idx()];
        let pushed_off_white = g.board.pushed_off[Side::White.idx()];
        let turn_byte: u8 = match g.turn {
            Side::Black => 0,
            Side::White => 1,
        };
        let n_children = e.child_visits.len() as u16;

        self.out.write_all(&own_bb.to_le_bytes())?;
        self.out.write_all(&opp_bb.to_le_bytes())?;
        self.out.write_all(&[pushed_off_black, pushed_off_white, turn_byte, 0])?;
        self.out.write_all(&(g.ply as u16).to_le_bytes())?;
        self.out.write_all(&e.move_played.to_le_bytes())?;
        self.out.write_all(&n_children.to_le_bytes())?;
        self.out.write_all(&[0u8; 2])?;
        self.out.write_all(&z.to_le_bytes())?;
        self.out.write_all(&e.q.to_le_bytes())?;
        for &(mv, v) in &e.child_visits {
            self.out.write_all(&mv.to_le_bytes())?;
            self.out.write_all(&[0u8; 2])?;
            self.out.write_all(&v.to_le_bytes())?;
        }

        self.bytes_written +=
            ENTRY_FIXED_BYTES as u64 + (n_children as u64) * CHILD_VISIT_BYTES as u64;
        self.entries_written += 1;
        Ok(())
    }

    pub fn finish(mut self) -> std::io::Result<()> {
        self.out.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{play_game, SelfPlayConfig};
    use abalone_mcts::heuristic;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn round_trip_header() {
        let dir = std::env::temp_dir();
        let path = dir.join("abalone-test-shard.bin");
        let writer = ShardWriter::create(&path, 200).expect("create");
        writer.finish().expect("finish");
        let bytes = std::fs::read(&path).expect("read");
        assert_eq!(&bytes[0..4], SHARD_MAGIC);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            SHARD_VERSION
        );
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            200
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_real_game_and_count_entries() {
        let cfg = SelfPlayConfig {
            simulations: 15,
            c_puct: 1.4,
            temperature_plies: 4,
            temperature: 1.0,
        };
        let mut rng = SmallRng::seed_from_u64(0);
        let outcome = play_game(&cfg, &mut rng, heuristic);

        let path = std::env::temp_dir().join("abalone-test-game.bin");
        let mut writer = ShardWriter::create(&path, cfg.simulations).expect("create");
        writer.write_game(&outcome).expect("write_game");
        let entries = writer.entries_written;
        let bytes = writer.bytes_written;
        writer.finish().expect("finish");

        assert_eq!(entries as usize, outcome.trajectory.len());
        let on_disk = std::fs::metadata(&path).expect("metadata").len();
        assert_eq!(on_disk, bytes);

        std::fs::remove_file(&path).ok();
    }
}
