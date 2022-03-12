// https://cardgamedatabase.fandom.com/wiki/Abalone_(board_game)#Move_notation
//
// A popular notation: An inline move can be denoted by the movement of the
// trailing marble. Broadside moves can be denoted by the initial positions
// of the two extremities of the row followed by the final position of the
// first one (thus, with this notation, each broadside move has two notations
// possible, which could be avoided).
//
// https://project.dke.maastrichtuniversity.nl/games/files/msc/pcreport.pdf
//
// A move can also be represented using this notation. A simple notation can
// be used for inline moves. Only the field occupied by the last marble to move
// is noted followed by the field the marble is moved to.
//
// To notate a broadside move one has to refer to three fields. First of all one
// mentions the first and the last field of marbles in a row that are moved. The
// third field that is noted indicates the new field for the marble mentioned first.

use crate::positions::{Direction, Position};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Color {
    Black,
    White,
}

// The first position is the starting position of the trailing marble (or
// possibly the only marble).  The second position is the ending point of that
// same marble.  The movement of all other marbles is implied.
#[derive(Debug, Clone, Copy)]
pub struct RowMove {
    pub starting_position: Position,
    pub ending_position: Position,
}

// The first position is one end point of a series of marbles.  The
// second position is the other end point of that series of marbles.
// The third position is the final position of the first end point.
#[derive(Debug, Clone, Copy)]
pub struct BroadsideMove {
    pub group_start: Position,
    pub group_end: Position,
    pub group_start_final_position: Position,
}

// A PieceMove is the simplest representation of an abstract move.  It exists outside of a
// board or game.
#[derive(Debug, Clone, Copy)]
pub enum PieceMove {
    RowMove(RowMove),
    BroadsideMove(BroadsideMove),
}

/*
#[derive(Debug, Clone, Copy)]
pub struct PieceGroup {
    pub start: Position,
    pub end: Position,
    pub num_marbles: usize,
}

 */

// The specific interpretation of an abstract PieceMove, given a game.  It provides additional
// metadata that can be derived from a PieceMove given a game/board.
#[derive(Debug, Clone, Copy)]
pub struct InterpretedMove {
    pub piece_move: PieceMove,
    pub starting_position: Position,
    pub color: Color,
    pub direction: Direction,
    pub num_same_color: usize,
    pub num_opposite_color: usize,
}
