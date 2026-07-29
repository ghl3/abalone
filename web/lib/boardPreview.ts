"use client";

import {
  cellCenter,
  DIR_PIXEL,
  DIR_SHIFTS,
  isValid,
  type MovingState,
} from "@/components/HexBoard";

type WasmModule = typeof import("abalone-wasm");

/** Build a snapped MovingState from a move idx — used for hover preview
 *  on an analysis row. The own destinations are the source cells shifted
 *  by the move's direction (NOT computed from the diff: for an inline
 *  push only the front-most landing cell shows up as a diff change, so
 *  diff-derived dest sets would be too short). Opponent diffs are
 *  pulled from `move_preview()` as before. */
export function buildHoverPreview(
  game: InstanceType<WasmModule["WasmGame"]>,
  wasm: WasmModule,
  cells: Int8Array,
  turnSide: 0 | 1,
  moveIdx: number
): MovingState {
  const oppSide = (turnSide === 0 ? 1 : 0) as 0 | 1;
  const srcRaw = Array.from(game.move_source_cells(moveIdx));
  const ownFromCells = srcRaw.filter((c) => c !== 0xff);

  const dirIdx = wasm.move_motion_dir(moveIdx);
  const dirShift = DIR_SHIFTS[dirIdx];
  const ownToCells = ownFromCells.map((c) => c + dirShift);
  const ownPositions = ownToCells.map(cellCenter);

  const post = game.move_preview(moveIdx);
  const oppFromCells: number[] = [];
  const oppToCells: number[] = [];
  let preOppCount = 0,
    postOppCount = 0;
  for (let c = 0; c < 81; c++) {
    const wasOpp = cells[c] === oppSide;
    const isOpp = post[c] === oppSide;
    if (wasOpp) preOppCount++;
    if (isOpp) postOppCount++;
    if (wasOpp && !isOpp) oppFromCells.push(c);
    if (!wasOpp && isOpp) oppToCells.push(c);
  }

  const oppToPositions = oppToCells.map((c) => ({
    ...cellCenter(c),
    offBoard: false,
  }));

  if (preOppCount > postOppCount && oppFromCells.length > 0) {
    let front = oppFromCells[0];
    while (
      isValid(front + dirShift) &&
      cells[front + dirShift] === oppSide
    ) {
      front += dirShift;
    }
    const fc = cellCenter(front);
    oppToPositions.push({
      x: fc.x + DIR_PIXEL[dirIdx].dx,
      y: fc.y + DIR_PIXEL[dirIdx].dy,
      offBoard: true,
    });
  }

  return {
    ownerColor: turnSide,
    oppColor: oppSide,
    ownFromCells,
    ownPositions,
    snapped: true,
    ownToCells,
    oppFromCells,
    oppToPositions,
    preview: true,
  };
}
