//! Self-play shards as Apache Parquet files. One row per trajectory
//! position; schema is stable enough to be a public artifact and is
//! checked at read time.
//!
//! # Why parquet
//!
//! - **Zero-copy numpy reads.** `pyarrow.parquet.read_table` decodes
//!   directly into Arrow buffers; columnar primitives surface as numpy
//!   arrays without per-row Python object allocation.
//! - **Compression on low-entropy data.** Bitboards repeat heavily
//!   within a game's trajectory; snappy + dictionary encoding shrinks
//!   shards 2–5× vs raw bytes.
//! - **Schema evolution is documented.** Adding a column later doesn't
//!   require versioning logic on our side — pyarrow + arrow-rs
//!   handle missing columns natively.
//!
//! # Schema
//!
//! ```text
//! own_bb_lo:        u64    bits 0..64 of the side-to-move bitboard
//! own_bb_hi:        u64    bits 64..128
//! opp_bb_lo:        u64    bits 0..64 of the opponent bitboard
//! opp_bb_hi:        u64    bits 64..128
//! pushed_off_black: u8     opp marbles black has pushed off
//! pushed_off_white: u8     opp marbles white has pushed off
//! turn:             u8     0 = Black, 1 = White
//! ply:              u16    half-move counter at this position
//! move_played:      u16    flat 0..2562 move idx that was applied next
//! z:                f32    final game outcome from this POV (-1, 0, +1)
//! q:                f32    visit-weighted root Q at this position, this POV
//! child_move_idxs:  list<u16>   move indices for each legal child of root
//! child_visits:     list<u32>   visit counts, parallel to child_move_idxs
//! ```
//!
//! Records are accumulated in builders and flushed as Arrow record
//! batches every `BATCH_ROWS` rows. The parquet writer applies snappy
//! compression and dictionary encoding to the columns where it helps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Builder, ListBuilder, RecordBatch, UInt16Builder, UInt32Builder,
    UInt64Builder, UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use abalone_game::Side;

use crate::{GameOutcome, TrajectoryEntry};

/// Number of rows accumulated before flushing a record batch to the
/// parquet writer. Smaller = lower memory; larger = better compression.
/// 4096 is a good middle ground for our row size (~50 bytes after
/// compression).
const BATCH_ROWS: u64 = 4096;

pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("own_bb_lo", DataType::UInt64, false),
        Field::new("own_bb_hi", DataType::UInt64, false),
        Field::new("opp_bb_lo", DataType::UInt64, false),
        Field::new("opp_bb_hi", DataType::UInt64, false),
        Field::new("pushed_off_black", DataType::UInt8, false),
        Field::new("pushed_off_white", DataType::UInt8, false),
        Field::new("turn", DataType::UInt8, false),
        Field::new("ply", DataType::UInt16, false),
        Field::new("move_played", DataType::UInt16, false),
        Field::new("z", DataType::Float32, false),
        Field::new("q", DataType::Float32, false),
        // Inner items are marked nullable to match arrow-rs's
        // `ListBuilder` default. We never actually emit nulls, but the
        // declared nullability has to agree with the builder's output.
        Field::new(
            "child_move_idxs",
            DataType::List(Arc::new(Field::new("item", DataType::UInt16, true))),
            false,
        ),
        Field::new(
            "child_visits",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            false,
        ),
    ]))
}

pub struct ShardWriter {
    writer: ArrowWriter<std::fs::File>,
    schema: Arc<Schema>,
    /// Final destination path. We write to `<final>.tmp` and rename in
    /// `finish()` so the trainer never sees a partial-but-named file
    /// (which would fail `pq.read_table` and spam the log).
    final_path: PathBuf,
    tmp_path: PathBuf,
    own_bb_lo: UInt64Builder,
    own_bb_hi: UInt64Builder,
    opp_bb_lo: UInt64Builder,
    opp_bb_hi: UInt64Builder,
    pushed_off_black: UInt8Builder,
    pushed_off_white: UInt8Builder,
    turn: UInt8Builder,
    ply: UInt16Builder,
    move_played: UInt16Builder,
    z: Float32Builder,
    q: Float32Builder,
    child_move_idxs: ListBuilder<UInt16Builder>,
    child_visits: ListBuilder<UInt32Builder>,
    rows_in_batch: u64,
    pub games_written: u32,
    pub entries_written: u64,
}

impl ShardWriter {
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self, parquet::errors::ParquetError> {
        let final_path = path.as_ref().to_path_buf();
        let mut tmp_path = final_path.clone();
        tmp_path.set_extension("parquet.tmp");
        let file = std::fs::File::create(&tmp_path)?;
        let schema = schema();
        // zstd at a low level: faster than snappy on writes but compresses
        // bitboards substantially better. Level 3 is the standard "fast" tier.
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
            .set_dictionary_enabled(true)
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self {
            writer,
            schema,
            final_path,
            tmp_path,
            own_bb_lo: UInt64Builder::new(),
            own_bb_hi: UInt64Builder::new(),
            opp_bb_lo: UInt64Builder::new(),
            opp_bb_hi: UInt64Builder::new(),
            pushed_off_black: UInt8Builder::new(),
            pushed_off_white: UInt8Builder::new(),
            turn: UInt8Builder::new(),
            ply: UInt16Builder::new(),
            move_played: UInt16Builder::new(),
            z: Float32Builder::new(),
            q: Float32Builder::new(),
            child_move_idxs: ListBuilder::new(UInt16Builder::new()),
            child_visits: ListBuilder::new(UInt32Builder::new()),
            rows_in_batch: 0,
            games_written: 0,
            entries_written: 0,
        })
    }

    pub fn write_game(
        &mut self,
        outcome: &GameOutcome,
    ) -> Result<(), parquet::errors::ParquetError> {
        for entry in &outcome.trajectory {
            let z = outcome.z_for(entry);
            self.append_entry(entry, z);
            self.rows_in_batch += 1;
            self.entries_written += 1;
            if self.rows_in_batch >= BATCH_ROWS {
                self.flush_batch()?;
            }
        }
        self.games_written += 1;
        Ok(())
    }

    fn append_entry(&mut self, e: &TrajectoryEntry, z: f32) {
        let g = &e.state;
        let own = g.board.bb(g.turn);
        let opp = g.board.bb(g.turn.other());

        self.own_bb_lo.append_value(own as u64);
        self.own_bb_hi.append_value((own >> 64) as u64);
        self.opp_bb_lo.append_value(opp as u64);
        self.opp_bb_hi.append_value((opp >> 64) as u64);
        self.pushed_off_black
            .append_value(g.board.pushed_off[Side::Black.idx()]);
        self.pushed_off_white
            .append_value(g.board.pushed_off[Side::White.idx()]);
        self.turn.append_value(match g.turn {
            Side::Black => 0,
            Side::White => 1,
        });
        self.ply.append_value(g.ply as u16);
        self.move_played.append_value(e.move_played);
        self.z.append_value(z);
        self.q.append_value(e.q);

        // Two parallel List columns for the child distribution.
        for &(mv, _) in &e.child_visits {
            self.child_move_idxs.values().append_value(mv);
        }
        self.child_move_idxs.append(true);
        for &(_, v) in &e.child_visits {
            self.child_visits.values().append_value(v);
        }
        self.child_visits.append(true);
    }

    fn flush_batch(&mut self) -> Result<(), parquet::errors::ParquetError> {
        if self.rows_in_batch == 0 {
            return Ok(());
        }
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(self.own_bb_lo.finish()),
            Arc::new(self.own_bb_hi.finish()),
            Arc::new(self.opp_bb_lo.finish()),
            Arc::new(self.opp_bb_hi.finish()),
            Arc::new(self.pushed_off_black.finish()),
            Arc::new(self.pushed_off_white.finish()),
            Arc::new(self.turn.finish()),
            Arc::new(self.ply.finish()),
            Arc::new(self.move_played.finish()),
            Arc::new(self.z.finish()),
            Arc::new(self.q.finish()),
            Arc::new(self.child_move_idxs.finish()),
            Arc::new(self.child_visits.finish()),
        ];
        let batch = RecordBatch::try_new(self.schema.clone(), arrays)
            .map_err(|e| parquet::errors::ParquetError::ArrowError(e.to_string()))?;
        self.writer.write(&batch)?;
        self.rows_in_batch = 0;
        Ok(())
    }

    pub fn finish(mut self) -> Result<u64, parquet::errors::ParquetError> {
        self.flush_batch()?;
        let _meta = self.writer.close()?;
        // Atomic rename — the trainer's poll loop only ever sees the
        // .parquet name once the footer has been written.
        std::fs::rename(&self.tmp_path, &self.final_path)
            .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))?;
        Ok(self.entries_written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{play_game, SelfPlayConfig};
    use abalone_mcts::heuristic;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn write_then_read_back_real_game() {
        let cfg = SelfPlayConfig {
            simulations: 12,
            c_puct: 1.4,
            temperature_plies: 4,
            temperature: 1.0,
            dirichlet_alpha: 0.3,
            dirichlet_eps: 0.0,
        };
        let mut rng = SmallRng::seed_from_u64(0);
        let outcome = play_game(&cfg, &mut rng, heuristic);

        let path = std::env::temp_dir().join("abalone-test-shard.parquet");
        let mut writer = ShardWriter::create(&path).expect("create");
        writer.write_game(&outcome).expect("write");
        let n_entries = writer.finish().expect("finish");

        assert_eq!(n_entries as usize, outcome.trajectory.len());

        // Read it back via the parquet reader and verify schema + row count.
        let file = std::fs::File::open(&path).expect("open");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("builder");
        assert_eq!(
            builder.schema().fields().len(),
            schema().fields().len(),
            "schema field count mismatch on round-trip"
        );
        let reader = builder.build().expect("reader");
        let total_rows: usize = reader
            .map(|b| b.expect("batch").num_rows())
            .sum();
        assert_eq!(total_rows, outcome.trajectory.len());

        std::fs::remove_file(&path).ok();
    }
}
