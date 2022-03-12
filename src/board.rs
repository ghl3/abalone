// Representation of an Abalone board.
// Marble: A physical piece.  Has a specific color
// Circle: A position on the board.  Defined by a Row and a Diagonal (or just a Position)

use crate::piece_move::{BroadsideMove, Color, InterpretedMove, PieceGroup, PieceMove, RowMove};
use crate::positions::{Diagonal, Direction, Position};

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

// Or, arranged horizontally:
//
//     I         O O O O O
//     H       O O O O O O
//     G     + + O O O + +
//     F   + + + + + + + +
//     E + + + + + + + + +
//     D + + + + + + + +
//     C + + @ @ @ + +
//     B @ @ @ @ @ @
//     A @ @ @ @ @
//       1 2 3 4 5 6 7 8 9

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

#[derive(Debug, Clone, Copy)]
pub enum MoveResult {
    Invalid,
    Valid(InterpretedMove),
}

// Implementation block, all `Point` associated functions & methods go in here
impl Board {
    pub fn circle(&self, position: Position) -> Circle {
        let (diagonal, row) = position.get_diagonal_row();
        let index = diagonal.get_index_of_row(row);
        self.diagonal(diagonal)[index]
    }

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
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
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
        match diagonal {
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

    fn interpret_move(&self, piece_move: PieceMove) -> Option<InterpretedMove> {
        match piece_move {
            PieceMove::RowMove(row_move) => self.interpret_row_move(row_move),
            PieceMove::BroadsideMove(broadside_move) => {
                self.interpret_broadside_move(broadside_move)
            }
        }
    }

    fn can_move(&self, position: Position, direction: Direction, distance: usize) -> bool {
        // Validate

        false
    }

    fn interpret_row_move(&self, row_move: RowMove) -> Option<InterpretedMove> {
        // First, get the marble corresponding
        let Circle::Filled(color) = self.circle(row_move.starting_position) else {
            return Option::None;
        };

        let Option::Some((direction, distance))  =
           row_move.starting_position.get_direction_and_distance(row_move.ending_position) else {
            return Option::None;
        };

        if distance != 1 {
            return Option::None;
        }

        // Ensure all marbles in the move have the same color and that the move is valid

        Option::Some(InterpretedMove {
            piece_move: PieceMove::RowMove(row_move),
            color: color,
            pieces: PieceGroup {
                start: row_move.starting_position,
                end: Position::A1,
                num_marbles: 0,
            },
            direction: direction,
        })
    }

    fn interpret_broadside_move(&self, broadside_move: BroadsideMove) -> Option<InterpretedMove> {
        Option::None
    }

    pub fn apply_move(&mut self, piece_move: PieceMove) -> MoveResult {
        MoveResult::Invalid
    }
}

#[cfg(test)]
mod tests {
    use crate::board::{Board, Circle, Color, Position};

    #[test]
    fn get_circle() {
        assert_eq!(
            Board::starting_board().circle(Position::A1),
            Circle::Filled(Color::White)
        );
        assert_eq!(Board::starting_board().circle(Position::D1), Circle::Empty);
        assert_eq!(
            Board::starting_board().circle(Position::H6),
            Circle::Filled(Color::Black)
        );
        assert_eq!(Board::starting_board().circle(Position::C2), Circle::Empty);
        assert_eq!(
            Board::starting_board().circle(Position::C3),
            Circle::Filled(Color::White)
        );
        assert_eq!(
            Board::starting_board().circle(Position::C5),
            Circle::Filled(Color::White)
        );
        assert_eq!(Board::starting_board().circle(Position::C6), Circle::Empty);

        assert_eq!(Board::starting_board().circle(Position::G4), Circle::Empty);

        assert_eq!(
            Board::starting_board().circle(Position::G5),
            Circle::Filled(Color::Black)
        );

        assert_eq!(
            Board::starting_board().circle(Position::I5),
            Circle::Filled(Color::Black)
        );
    }
}
