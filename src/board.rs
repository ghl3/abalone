// Representation of an Abalone board.

use crate::pieces::{Circle, Color, Direction, PieceMove};
use crate::positions::{Diagonal, Position};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Color {
    Black,
    White,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Circle {
    Empty,
    Filled(Color),
}

// The board is defined as a series of south-east to north-west diagonals,
// by convention, whose lengths start at 5, go up to 9, and then back
// down to 5.
//
//      I O O O O O
//     H O O O O O O
//    G + + O O O + +
//   F + + + + + + + +
//  E + + + + + + + + +
//   D + + + + + + + + 9
//    C + + @ @ @ + + 8
//     B @ @ @ @ @ @ 7
//      A @ @ @ @ @ 6
//         1 2 3 4 5

#[derive(Debug, PartialEq, Eq)]
pub struct Board {
    one: [Circle; 5],
    two: [Circle; 6],
    three: [Circle; 7],
    four: [Circle; 8],
    five: [Circle; 9],
    six: [Circle; 8],
    seven: [Circle; 7],
    eight: [Circle; 6],
    nine: [Circle; 5],
}

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

#[derive(Debug, Clone, Copy)]
pub enum Movement {
    // The first position is the starting position of the trailing marble (or
    // possibly the only marble).  The second position is the ending point of that
    // same marble.  The movement of all other marbles is implied.
    RowMove(Position, Position),

    // The first position is one end point of a series of marbles.  The
    // second position is the other end point of that series of marbles.
    // The third position is the final position of the first end point.
    BroadsideMove(Position, Position, Position),
}

#[derive(Debug, Clone, Copy)]
enum Direction {
    NorthEast,
    NorthWest,
    East,
    SouthEast,
    SouthWest,
    West,
}

pub struct PieceGroup {
    start: Position,
    end: Position,
    num_marbles: i8,
}

pub struct InterpretedMove {
    pub movement: Movement,
    pub color: Color,
    pub pieces: PieceGroup,
    pub direction: Direction,
}

#[derive(Debug, Clone, Copy)]
pub enum MoveResult {
    Invalid,
    Valid(InterpretedMove),
}

// Implementation block, all `Point` associated functions & methods go in here
impl Board {
    pub fn empty_board() -> Board {
        Board {
            one: [Circle::Empty; 5],
            two: [Circle::Empty; 6],
            three: [Circle::Empty; 7],
            four: [Circle::Empty; 8],
            five: [Circle::Empty; 9],
            six: [Circle::Empty; 8],
            seven: [Circle::Empty; 7],
            eight: [Circle::Empty; 6],
            nine: [Circle::Empty; 5],
        }
    }

    //
    //      I O O O O O
    //     H O O O O O O
    //    G + + O O O + +
    //   F + + + + + + + +
    //  E + + + + + + + + +
    //   D + + + + + + + + 9
    //    C + + @ @ @ + + 8
    //     B @ @ @ @ @ @ 7
    //      A @ @ @ @ @ 6
    //         1 2 3 4 5
    pub fn starting_board() -> Board {
        Board {
            one: [
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
            ],
            two: [
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
            ],
            three: [
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
            ],
            four: [
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::Black),
            ],
            five: [
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
            ],
            six: [
                Circle::Filled(Color::Black),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
            ],
            seven: [
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
            ],
            eight: [
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
            ],
            nine: [
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
            ],
        }
    }

    pub fn diagonal(&self, diagonal: Diagonal) -> &[Circle] {
        match row {
            Diagonal::ONE => &self.one,
            Diagonal::TWO => &self.two,
            Diagonal::THREE => &self.three,
            Diagonal::FOUR => &self.four,
            Diagonal::FIVE => &self.five,
            Diagonal::SIX => &self.six,
            Diagonal::SEVEN => &self.seven,
            Diagonal::EIGHT => &self.eight,
            Diagonal::NINE => &self.nine,
        }
    }

    pub fn circle(&self, position: Position) -> Circle {
        let (diagonal, row) = position.get_diagonal_row();
        self.diagonal(diagonal)[row.index()]
    }

    fn interpret_move(&self, movement: Movement) -> Option<InterpretedMove> {
        match movement {
            Movement::RowMove(starting_position, ending_position) => {
                self.interpret_row_move(starting_position, ending_position)
            }
            Movement::BroadsideMove(group_start, group_end, new_position) => {
                self.interpret_broadside_move(group_start, group_end, new_position)
            }
        }
    }

    fn interpret_row_move(
        &self,
        starting_position: Position,
        ending_position: Position,
    ) -> Option<InterpretedMove> {
        // First, we ensure there is a circle

        match self.circle(starting_position) {
            Circle::Empty => Option::None,

            Option::Filled(color) => {
                // TODO: Convert this to an InterpretedMove
                Option::None
            }
        }
    }

    fn interpret_broadside_move(
        &self,
        group_start: Position,
        group_end: Position,
        new_position: Position,
    ) -> Option<InterpretedMove> {
        Option::None
    }

    pub fn apply_move(&mut self, piece_move: PieceMove) -> MoveResult {
        MoveResult::Invalid
    }
}

#[cfg(test)]
mod tests {
    use crate::board::{Board, Circle, Color, Column, Position, Row};

    #[test]
    fn get_circle() {
        assert_eq!(
            Board::starting_board().circle(Position {
                row: Row::ONE,
                column: Column::ONE
            }),
            Some(Circle::Filled(Color::White))
        );
        assert_eq!(
            Board::starting_board().circle(Position {
                row: Row::FIVE,
                column: Column::ONE
            }),
            Some(Circle::Empty)
        );
        assert_eq!(
            Board::starting_board().circle(Position {
                row: Row::NINE,
                column: Column::ONE
            }),
            Some(Circle::Filled(Color::Black))
        );
        assert_eq!(
            Board::starting_board().circle(Position {
                row: Row::ONE,
                column: Column::NINE
            }),
            None
        );
    }
}
