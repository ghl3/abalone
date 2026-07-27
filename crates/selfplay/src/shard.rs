//! Self-play shards as Apache Parquet files — **schema v2**, normative
//! definition in [ARCHITECTURE §5.4](../../../docs/ARCHITECTURE.md). One row per
//! trajectory position.
//!
//! # Why parquet
//!
//! - **Zero-copy numpy reads.** `pyarrow.parquet.read_table` decodes
//!   directly into Arrow buffers; columnar primitives surface as numpy
//!   arrays without per-row Python object allocation.
//! - **Compression on low-entropy data.** Bitboards repeat heavily
//!   within a game's trajectory; zstd + dictionary encoding shrinks
//!   shards 2–5× vs raw bytes.
//! - **Schema evolution is documented.** Adding a column later doesn't
//!   require versioning logic on our side — pyarrow + arrow-rs
//!   handle missing columns natively.
//!
//! # Schema
//!
//! ```text
//! game_id          u32        source game within (run, generation)
//! seed             u64        RNG seed — the game is reproducible from this
//! opening          u8         0 = Standard, 1 = BelgianDaisy
//! handicap_black   u8         marbles Black conceded at curriculum seeding
//! handicap_white   u8         marbles White conceded at curriculum seeding
//!
//! own_bb_lo/hi     u64        side-to-move relative bitboards
//! opp_bb_lo/hi     u64
//! black_losses     u8         marbles BLACK has lost
//! white_losses     u8         marbles WHITE has lost
//! turn             u8         0 = Black, 1 = White
//! ply              u16
//! max_plies        u16        this game's cap — the ply plane's denominator
//!
//! move_played      u16        flat move index applied next
//! is_full_search   bool       true iff this position ran the FULL simulation
//!                             count; only these carry a policy target
//!
//! z                i8         outcome from this POV: +1 win, 0 draw, −1 loss
//! score_diff       i8         final capture differential from this POV, [−6, 6]
//! q                f32        MCTS root value from this POV (diagnostics / UI)
//!
//! child_move_idxs  list<u16>  the search result — the policy target
//! child_visits     list<u32>  parallel to child_move_idxs
//!
//! cap_map_idx      list<u16>  sparse capture map: channel*81 + cell, 0..162
//! cap_map_val      list<f32>  parallel discounted weights, clamped to [0, 1]
//! ```
//!
//! **`black_losses` / `white_losses` are named for LOSSES.** The v1 columns were
//! `pushed_off_black`, which held "marbles pushed off *by* Black" but reads
//! naturally as the opposite. That ambiguity produced a real plane-swap training
//! bug. [`abalone_game::Board::lost`] does the flip; never index `pushed_off`
//! here.
//!
//! Records are accumulated in builders and flushed as Arrow record
//! batches every `BATCH_ROWS` rows. The parquet writer applies zstd
//! compression and dictionary encoding to the columns where it helps.
//!
//! **Writes are atomic.** We write to `<name>.parquet.tmp` and rename in
//! [`ShardWriter::finish`], so the trainer — which polls the shard directory —
//! never sees a `.parquet` file without a footer.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float32Builder, Int8Builder, ListBuilder, RecordBatch,
    UInt16Builder, UInt32Builder, UInt64Builder, UInt8Builder,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use abalone_game::{Opening, Side};

use crate::{GameOutcome, TrajectoryEntry};

/// Number of rows accumulated before flushing a record batch to the
/// parquet writer. Smaller = lower memory; larger = better compression.
/// 4096 is a good middle ground for our row size (~50 bytes after
/// compression).
const BATCH_ROWS: u64 = 4096;

/// Schema version, written into the parquet key-value metadata so a reader can
/// fail loudly rather than silently mis-map columns.
pub const SCHEMA_VERSION: &str = "2";

/// Numeric code for `opening`. Stable — it is a shard column.
fn opening_code(o: Opening) -> u8 {
    match o {
        Opening::Standard => 0,
        Opening::BelgianDaisy => 1,
    }
}

/// A `list<item>` field with a nullable inner item. Inner nullability has to
/// match arrow-rs's `ListBuilder` default even though we never emit nulls.
fn list_field(name: &str, item: DataType) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", item, true))),
        false,
    )
}

pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("game_id", DataType::UInt32, false),
        Field::new("seed", DataType::UInt64, false),
        Field::new("opening", DataType::UInt8, false),
        Field::new("handicap_black", DataType::UInt8, false),
        Field::new("handicap_white", DataType::UInt8, false),
        Field::new("own_bb_lo", DataType::UInt64, false),
        Field::new("own_bb_hi", DataType::UInt64, false),
        Field::new("opp_bb_lo", DataType::UInt64, false),
        Field::new("opp_bb_hi", DataType::UInt64, false),
        Field::new("black_losses", DataType::UInt8, false),
        Field::new("white_losses", DataType::UInt8, false),
        Field::new("turn", DataType::UInt8, false),
        Field::new("ply", DataType::UInt16, false),
        Field::new("max_plies", DataType::UInt16, false),
        Field::new("move_played", DataType::UInt16, false),
        Field::new("is_full_search", DataType::Boolean, false),
        Field::new("z", DataType::Int8, false),
        Field::new("score_diff", DataType::Int8, false),
        Field::new("q", DataType::Float32, false),
        list_field("child_move_idxs", DataType::UInt16),
        list_field("child_visits", DataType::UInt32),
        list_field("cap_map_idx", DataType::UInt16),
        list_field("cap_map_val", DataType::Float32),
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
    game_id: UInt32Builder,
    seed: UInt64Builder,
    opening: UInt8Builder,
    handicap_black: UInt8Builder,
    handicap_white: UInt8Builder,
    own_bb_lo: UInt64Builder,
    own_bb_hi: UInt64Builder,
    opp_bb_lo: UInt64Builder,
    opp_bb_hi: UInt64Builder,
    black_losses: UInt8Builder,
    white_losses: UInt8Builder,
    turn: UInt8Builder,
    ply: UInt16Builder,
    max_plies: UInt16Builder,
    move_played: UInt16Builder,
    is_full_search: BooleanBuilder,
    z: Int8Builder,
    score_diff: Int8Builder,
    q: Float32Builder,
    child_move_idxs: ListBuilder<UInt16Builder>,
    child_visits: ListBuilder<UInt32Builder>,
    cap_map_idx: ListBuilder<UInt16Builder>,
    cap_map_val: ListBuilder<Float32Builder>,
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
            .set_key_value_metadata(Some(vec![parquet::file::metadata::KeyValue::new(
                "abalone_shard_schema_version".to_string(),
                SCHEMA_VERSION.to_string(),
            )]))
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self {
            writer,
            schema,
            final_path,
            tmp_path,
            game_id: UInt32Builder::new(),
            seed: UInt64Builder::new(),
            opening: UInt8Builder::new(),
            handicap_black: UInt8Builder::new(),
            handicap_white: UInt8Builder::new(),
            own_bb_lo: UInt64Builder::new(),
            own_bb_hi: UInt64Builder::new(),
            opp_bb_lo: UInt64Builder::new(),
            opp_bb_hi: UInt64Builder::new(),
            black_losses: UInt8Builder::new(),
            white_losses: UInt8Builder::new(),
            turn: UInt8Builder::new(),
            ply: UInt16Builder::new(),
            max_plies: UInt16Builder::new(),
            move_played: UInt16Builder::new(),
            is_full_search: BooleanBuilder::new(),
            z: Int8Builder::new(),
            score_diff: Int8Builder::new(),
            q: Float32Builder::new(),
            child_move_idxs: ListBuilder::new(UInt16Builder::new()),
            child_visits: ListBuilder::new(UInt32Builder::new()),
            cap_map_idx: ListBuilder::new(UInt16Builder::new()),
            cap_map_val: ListBuilder::new(Float32Builder::new()),
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
            self.append_entry(outcome, entry);
            self.rows_in_batch += 1;
            self.entries_written += 1;
            if self.rows_in_batch >= BATCH_ROWS {
                self.flush_batch()?;
            }
        }
        self.games_written += 1;
        Ok(())
    }

    fn append_entry(&mut self, o: &GameOutcome, e: &TrajectoryEntry) {
        let g = &e.state;
        let own = g.board.bb(g.turn);
        let opp = g.board.bb(g.turn.other());

        self.game_id.append_value(o.game_id);
        self.seed.append_value(o.seed);
        self.opening.append_value(opening_code(o.opening));
        self.handicap_black.append_value(o.handicap_black);
        self.handicap_white.append_value(o.handicap_white);

        self.own_bb_lo.append_value(own as u64);
        self.own_bb_hi.append_value((own >> 64) as u64);
        self.opp_bb_lo.append_value(opp as u64);
        self.opp_bb_hi.append_value((opp >> 64) as u64);
        // `lost(side)` = marbles that side has had pushed off. Never index
        // `pushed_off` here: it holds the opposite of what its name suggests.
        self.black_losses.append_value(g.board.lost(Side::Black));
        self.white_losses.append_value(g.board.lost(Side::White));
        self.turn.append_value(match g.turn {
            Side::Black => 0,
            Side::White => 1,
        });
        self.ply.append_value(g.ply as u16);
        self.max_plies.append_value(g.max_plies as u16);

        self.move_played.append_value(e.move_played);
        self.is_full_search.append_value(e.is_full_search);

        self.z.append_value(o.z_class_for(e));
        self.score_diff.append_value(o.score_diff_for(e));
        self.q.append_value(e.q);

        // Parallel List columns for the child distribution...
        for &(mv, _) in &e.child_visits {
            self.child_move_idxs.values().append_value(mv);
        }
        self.child_move_idxs.append(true);
        for &(_, v) in &e.child_visits {
            self.child_visits.values().append_value(v);
        }
        self.child_visits.append(true);

        // ...and for the sparse capture map.
        for &(idx, _) in &e.capture_map {
            self.cap_map_idx.values().append_value(idx);
        }
        self.cap_map_idx.append(true);
        for &(_, w) in &e.capture_map {
            self.cap_map_val.values().append_value(w);
        }
        self.cap_map_val.append(true);
    }

    fn flush_batch(&mut self) -> Result<(), parquet::errors::ParquetError> {
        if self.rows_in_batch == 0 {
            return Ok(());
        }
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(self.game_id.finish()),
            Arc::new(self.seed.finish()),
            Arc::new(self.opening.finish()),
            Arc::new(self.handicap_black.finish()),
            Arc::new(self.handicap_white.finish()),
            Arc::new(self.own_bb_lo.finish()),
            Arc::new(self.own_bb_hi.finish()),
            Arc::new(self.opp_bb_lo.finish()),
            Arc::new(self.opp_bb_hi.finish()),
            Arc::new(self.black_losses.finish()),
            Arc::new(self.white_losses.finish()),
            Arc::new(self.turn.finish()),
            Arc::new(self.ply.finish()),
            Arc::new(self.max_plies.finish()),
            Arc::new(self.move_played.finish()),
            Arc::new(self.is_full_search.finish()),
            Arc::new(self.z.finish()),
            Arc::new(self.score_diff.finish()),
            Arc::new(self.q.finish()),
            Arc::new(self.child_move_idxs.finish()),
            Arc::new(self.child_visits.finish()),
            Arc::new(self.cap_map_idx.finish()),
            Arc::new(self.cap_map_val.finish()),
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
    use abalone_game::Game;
    use abalone_mcts::{heuristic, LeafEval};
    use arrow::array::{
        Array, BooleanArray, Float32Array, Int8Array, ListArray, UInt16Array, UInt32Array,
        UInt64Array, UInt8Array,
    };
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn heuristic_batch(games: &[Game]) -> Vec<LeafEval> {
        let mut rng = SmallRng::seed_from_u64(0);
        games.iter().map(|g| heuristic(g, &mut rng)).collect()
    }

    fn col<'a, T: 'static>(b: &'a RecordBatch, name: &str) -> &'a T {
        b.column_by_name(name)
            .unwrap_or_else(|| panic!("missing column {name}"))
            .as_any()
            .downcast_ref::<T>()
            .unwrap_or_else(|| panic!("wrong arrow type for {name}"))
    }

    fn list_u16(a: &ListArray, row: usize) -> Vec<u16> {
        let v = a.value(row);
        v.as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[test]
    fn schema_matches_architecture_54() {
        let s = schema();
        let names: Vec<&str> = s.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "game_id",
                "seed",
                "opening",
                "handicap_black",
                "handicap_white",
                "own_bb_lo",
                "own_bb_hi",
                "opp_bb_lo",
                "opp_bb_hi",
                "black_losses",
                "white_losses",
                "turn",
                "ply",
                "max_plies",
                "move_played",
                "is_full_search",
                "z",
                "score_diff",
                "q",
                "child_move_idxs",
                "child_visits",
                "cap_map_idx",
                "cap_map_val",
            ]
        );
    }

    #[test]
    fn write_then_read_back_real_game() {
        let cfg = SelfPlayConfig {
            sims_fast: 8,
            sims_full: 20,
            full_search_rate: 0.4,
            batch_size: 4,
            temperature_plies: 4,
            dirichlet_eps: 0.0,
            opening: abalone_game::Opening::BelgianDaisy,
            handicap_rate: 1.0,
            random_opening_plies: 1,
            max_plies: 40,
            ..Default::default()
        };
        let outcome = play_game(&cfg, 17, 0xabc, heuristic_batch);
        assert!(!outcome.trajectory.is_empty());

        let path = std::env::temp_dir().join("abalone-test-shard-v2.parquet");
        let mut writer = ShardWriter::create(&path).expect("create");
        writer.write_game(&outcome).expect("write");
        assert!(
            path.with_extension("parquet.tmp").exists(),
            "writes go to the .tmp path until finish()"
        );
        let n_entries = writer.finish().expect("finish");
        assert_eq!(n_entries as usize, outcome.trajectory.len());
        assert!(
            !path.with_extension("parquet.tmp").exists(),
            "finish() renames the .tmp away"
        );

        let file = std::fs::File::open(&path).expect("open");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("builder");
        let version = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .and_then(|kv| {
                kv.iter()
                    .find(|k| k.key == "abalone_shard_schema_version")
                    .and_then(|k| k.value.clone())
            });
        assert_eq!(version.as_deref(), Some(SCHEMA_VERSION));
        assert_eq!(builder.schema().fields().len(), schema().fields().len());
        let batches: Vec<RecordBatch> = builder
            .build()
            .expect("reader")
            .map(|b| b.expect("batch"))
            .collect();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, outcome.trajectory.len());

        // Every column, every row, against the trajectory it came from.
        let mut row = 0usize;
        for b in &batches {
            for i in 0..b.num_rows() {
                let e = &outcome.trajectory[row];
                let g = &e.state;
                assert_eq!(col::<UInt32Array>(b, "game_id").value(i), outcome.game_id);
                assert_eq!(col::<UInt64Array>(b, "seed").value(i), outcome.seed);
                assert_eq!(
                    col::<UInt8Array>(b, "opening").value(i),
                    opening_code(outcome.opening)
                );
                assert_eq!(
                    col::<UInt8Array>(b, "handicap_black").value(i),
                    outcome.handicap_black
                );
                assert_eq!(
                    col::<UInt8Array>(b, "handicap_white").value(i),
                    outcome.handicap_white
                );

                let own = g.board.bb(g.turn);
                let opp = g.board.bb(g.turn.other());
                assert_eq!(col::<UInt64Array>(b, "own_bb_lo").value(i), own as u64);
                assert_eq!(
                    col::<UInt64Array>(b, "own_bb_hi").value(i),
                    (own >> 64) as u64
                );
                assert_eq!(col::<UInt64Array>(b, "opp_bb_lo").value(i), opp as u64);
                assert_eq!(
                    col::<UInt64Array>(b, "opp_bb_hi").value(i),
                    (opp >> 64) as u64
                );
                assert_eq!(
                    col::<UInt8Array>(b, "black_losses").value(i),
                    g.board.lost(Side::Black)
                );
                assert_eq!(
                    col::<UInt8Array>(b, "white_losses").value(i),
                    g.board.lost(Side::White)
                );
                assert_eq!(
                    col::<UInt8Array>(b, "turn").value(i),
                    if g.turn == Side::Black { 0 } else { 1 }
                );
                assert_eq!(col::<UInt16Array>(b, "ply").value(i), g.ply as u16);
                assert_eq!(
                    col::<UInt16Array>(b, "max_plies").value(i),
                    g.max_plies as u16
                );
                assert_eq!(
                    col::<UInt16Array>(b, "move_played").value(i),
                    e.move_played
                );
                assert_eq!(
                    col::<BooleanArray>(b, "is_full_search").value(i),
                    e.is_full_search
                );
                assert_eq!(col::<Int8Array>(b, "z").value(i), outcome.z_class_for(e));
                assert_eq!(
                    col::<Int8Array>(b, "score_diff").value(i),
                    outcome.score_diff_for(e)
                );
                assert_eq!(col::<Float32Array>(b, "q").value(i), e.q);

                let idxs = list_u16(col::<ListArray>(b, "child_move_idxs"), i);
                let visits = col::<ListArray>(b, "child_visits").value(i);
                let visits = visits.as_any().downcast_ref::<UInt32Array>().unwrap();
                assert_eq!(idxs.len(), e.child_visits.len());
                for (k, &(mv, v)) in e.child_visits.iter().enumerate() {
                    assert_eq!(idxs[k], mv);
                    assert_eq!(visits.value(k), v);
                }

                let cap_idx = list_u16(col::<ListArray>(b, "cap_map_idx"), i);
                let cap_val = col::<ListArray>(b, "cap_map_val").value(i);
                let cap_val = cap_val.as_any().downcast_ref::<Float32Array>().unwrap();
                assert_eq!(cap_idx.len(), e.capture_map.len());
                for (k, &(idx, w)) in e.capture_map.iter().enumerate() {
                    assert_eq!(cap_idx[k], idx);
                    assert_eq!(cap_val.value(k), w);
                }
                row += 1;
            }
        }

        // The fixture has to actually exercise the interesting columns.
        assert!(
            outcome.trajectory.iter().any(|e| e.is_full_search),
            "fixture must contain at least one full-search position"
        );
        assert!(
            outcome.trajectory.iter().any(|e| !e.capture_map.is_empty()),
            "fixture must contain at least one capture-map target"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_game_writes_a_valid_shard() {
        let outcome = GameOutcome {
            game_id: 0,
            seed: 0,
            opening: abalone_game::Opening::Standard,
            handicap_black: 0,
            handicap_white: 0,
            trajectory: Vec::new(),
            final_state: Game::new_standard(),
            captures: Vec::new(),
        };
        let path = std::env::temp_dir().join("abalone-test-shard-empty.parquet");
        let mut w = ShardWriter::create(&path).expect("create");
        w.write_game(&outcome).expect("write");
        assert_eq!(w.finish().expect("finish"), 0);
        let file = std::fs::File::open(&path).expect("open");
        let n: usize = ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("builder")
            .build()
            .expect("reader")
            .map(|b| b.expect("batch").num_rows())
            .sum();
        assert_eq!(n, 0);
        std::fs::remove_file(&path).ok();
    }
}
