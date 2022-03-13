// Representation of an Abalone board.
// Marble: A physical piece.  Has a specific color
// Circle: A position on the board.  Defined by a Row and a Diagonal (or just a Position)

use crate::piece_move::{
    BroadsideMove, Color, InterpretedMove, PieceMove, RowMove,
};
use crate::positions::{Diagonal, Direction, Position};
use std::thread::current;

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

#[derive(Debug, PartialEq, Eq)]
struct RowMoveResult {
    color: Color,
    starting_position: Position,
    direction: Direction,
    num_same_color: usize,
    num_opposite_color: usize,
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

    fn can_move(
        &self,
        position: Position,
        direction: Direction,
        color: Color,
    ) -> Option<RowMoveResult> {
        // Starting at the position, we walk along the direction to see if a marble of a given
        // color can move in that direction.

        let mut current_position = position;

        // We walk until we find the end
        let mut still_same_color = true;
        let mut num_same_colors = 1;
        let mut num_opposite_colors = 0;

        // We walk until we encounter:
        // - A free space
        // - The end of the board
        // - A change in color
        loop {
            let Some(next_position) = current_position.neighbor(direction) else {
                // We encountered the end of the board
                return None;
            };

            // Update the current position
            current_position = next_position;

            let current_circle = self.circle(current_position);

            match current_circle {
                // We found an empty space.  We can return true
                Circle::Empty => {
                    return Some(RowMoveResult {
                        color: color,
                        starting_position: position,
                        direction: direction,
                        num_same_color: num_same_colors,
                        num_opposite_color: num_opposite_colors,
                    });
                }

                Circle::Filled(current_color) => {
                    if still_same_color && current_color == color {
                        // We encountered the original color again.
                        num_same_colors += 1;
                    }

                    if still_same_color && current_color != color {
                        // We swapped colors

                        num_opposite_colors += 1;
                        still_same_color = false;
                    }

                    if !still_same_color && current_color == color {
                        // We encountered the original color after the opposite color.
                        // We cannot push through this.
                        return None;
                    }

                    if num_same_colors > 3 {
                        return None;
                    }

                    if num_opposite_colors > 3 {
                        return None;
                    }
                }
            }
        }
    }

    fn interpret_row_move(&self, row_move: RowMove) -> Option<InterpretedMove> {
        // First, get the marble corresponding
        let Circle::Filled(color) = self.circle(row_move.starting_position) else {
            return Option::None;
        };

        let Option::Some((direction, distance)) =
            row_move.starting_position.get_direction_and_distance(row_move.ending_position) else {
            return Option::None;
        };

        if distance != 1 {
            return Option::None;
        }

        // Ensure all marbles in the move have the same color and that the move is valid
        let Some(row_move_result) = self.can_move(row_move.starting_position, direction, color) else {
            return Option::None;
        };

        Option::Some(InterpretedMove {
            piece_move: PieceMove::RowMove(row_move),
            starting_position: row_move.starting_position,
            color: color,
            direction: direction,
            num_same_color: row_move_result.num_same_color,
            num_opposite_color: row_move_result.num_opposite_color,
        })
    }

    fn squares_empty(
        &self,
        starting_position: Position,
        direction: Direction,
        distance: usize,
    ) -> bool {
        let mut num_steps = 1;
        let mut current_position = starting_position;

        loop {
            if self.circle(current_position) != Circle::Empty {
                return false;
            }

            if num_steps == distance {
                return true;
            }

            let Some(next_position) = current_position.neighbor(direction) else {
                // We have gone off the board, so we return false
                return false;
            };

            current_position = next_position;
            num_steps += 1;
        }
    }

    // TODO: Merge with 'squares_empty'
    fn squares_filled_with_color(
        &self,
        starting_position: Position,
        direction: Direction,
        distance: usize,
        color: Color,
    ) -> bool {
        let mut num_steps = 1;
        let mut current_position = starting_position;

        loop {
            if self.circle(current_position) != Circle::Filled(color) {
                return false;
            }

            if num_steps == distance {
                return true;
            }

            let Some(next_position) = current_position.neighbor(direction) else {
                // We have gone off the board, so we return false
                return false;
            };

            current_position = next_position;
            num_steps += 1;
        }
    }

    fn interpret_broadside_move(
        &self,
        broadside_move: BroadsideMove,
    ) -> Option<InterpretedMove> {
        // We need to confirm the following:
        // - All pieces in the starting position are of the same color
        // - There are no pieces in the terminal position

        // First, get the marble corresponding to the starting position
        let Circle::Filled(color) = self.circle(broadside_move.group_start) else {
            return Option::None;
        };

        // Then, get the distance and direction to the final position.
        // This determines the direction of the shift
        let Option::Some((_, shift_distance)) =
            broadside_move.group_start.get_direction_and_distance(
                broadside_move.group_start_final_position) else {
            return Option::None;
        };

        // The distance must only be 1 since shifts can only be one space
        if shift_distance != 1 {
            return Option::None;
        }

        //
        let Option::Some((group_direction, group_distance)) =
            broadside_move.group_start.get_direction_and_distance(
                broadside_move.group_start_final_position) else {
            return Option::None;
        };

        // The group can only be 3 elements (1 element + 2 distance = 3)
        if group_distance > 2 {
            return Option::None;
        }

        // Then, we check that the starting circles all have the same color.
        if !self.squares_filled_with_color(
            broadside_move.group_start,
            group_direction,
            group_distance,
            color,
        ) {
            return Option::None;
        }

        // Now, we check that all the target circles are empty
        if !self.squares_empty(
            broadside_move.group_start_final_position,
            group_direction,
            group_distance,
        ) {
            return Option::None;
        }

        Some(InterpretedMove {
            piece_move: PieceMove::BroadsideMove(broadside_move),
            starting_position: broadside_move.group_start,
            color,
            direction: group_direction,
            num_same_color: group_distance,
            num_opposite_color: 0,
        })
    }

    pub fn apply_move(&mut self, piece_move: PieceMove) -> MoveResult {
        MoveResult::Invalid
    }
}

#[cfg(test)]
mod tests {
    use crate::board::{Board, Circle, Color, Position, RowMoveResult};
    use crate::piece_move::{
        BroadsideMove, InterpretedMove, PieceMove, RowMove,
    };
    use crate::positions::Direction;

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

    #[test]
    fn test_can_move() {
        let board = Board::starting_board();

        assert_eq!(
            board.can_move(Position::A5, Direction::NorthWest, Color::White),
            Some(RowMoveResult {
                color: Color::White,
                starting_position: Position::A5,
                direction: Direction::NorthWest,
                num_same_color: 3,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::B5, Direction::NorthWest, Color::White),
            Some(RowMoveResult {
                color: Color::White,
                starting_position: Position::B5,
                direction: Direction::NorthWest,
                num_same_color: 2,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::C5, Direction::NorthWest, Color::White),
            Some(RowMoveResult {
                color: Color::White,
                starting_position: Position::C5,
                direction: Direction::NorthWest,
                num_same_color: 1,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::A4, Direction::NorthEast, Color::White),
            Some(RowMoveResult {
                color: Color::White,
                starting_position: Position::A4,
                direction: Direction::NorthEast,
                num_same_color: 2,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::B4, Direction::NorthEast, Color::White),
            Some(RowMoveResult {
                color: Color::White,
                starting_position: Position::B4,
                direction: Direction::NorthEast,
                num_same_color: 2,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::C3, Direction::NorthEast, Color::White),
            Some(RowMoveResult {
                color: Color::White,
                starting_position: Position::C3,
                direction: Direction::NorthEast,
                num_same_color: 1,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::C3, Direction::NorthWest, Color::White),
            Some(RowMoveResult {
                color: Color::White,
                starting_position: Position::C3,
                direction: Direction::NorthWest,
                num_same_color: 1,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::B6, Direction::East, Color::White),
            None
        );

        assert_eq!(
            board.can_move(Position::B1, Direction::West, Color::White),
            None
        );

        assert_eq!(
            board.can_move(Position::I5, Direction::SouthEast, Color::Black),
            Some(RowMoveResult {
                color: Color::Black,
                starting_position: Position::I5,
                direction: Direction::SouthEast,
                num_same_color: 3,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::I6, Direction::SouthEast, Color::Black),
            Some(RowMoveResult {
                color: Color::Black,
                starting_position: Position::I6,
                direction: Direction::SouthEast,
                num_same_color: 3,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::H5, Direction::SouthEast, Color::Black),
            Some(RowMoveResult {
                color: Color::Black,
                starting_position: Position::H5,
                direction: Direction::SouthEast,
                num_same_color: 2,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::F5, Direction::SouthEast, Color::Black),
            Some(RowMoveResult {
                color: Color::Black,
                starting_position: Position::F5,
                direction: Direction::SouthEast,
                num_same_color: 1,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::H9, Direction::SouthWest, Color::Black),
            Some(RowMoveResult {
                color: Color::Black,
                starting_position: Position::H9,
                direction: Direction::SouthWest,
                num_same_color: 1,
                num_opposite_color: 0,
            })
        );

        assert_eq!(
            board.can_move(Position::H9, Direction::East, Color::Black),
            None
        );

        assert_eq!(
            board.can_move(Position::I5, Direction::West, Color::Black),
            None
        );
    }

    #[test]
    fn test_squares_empty() {
        let board = Board::starting_board();

        assert!(board.squares_empty(Position::C1, Direction::NorthWest, 3));
        assert!(board.squares_empty(Position::C2, Direction::NorthWest, 1));

        assert!(!board.squares_empty(Position::A1, Direction::NorthWest, 1));
        assert!(!board.squares_empty(Position::E7, Direction::NorthWest, 3));
    }

    #[test]
    fn test_interpret_row_move() {
        let board = Board::starting_board();
        assert_eq!(
            board.interpret_row_move(RowMove {
                starting_position: Position::A5,
                ending_position: Position::B5,
            }),
            Some(InterpretedMove {
                piece_move: PieceMove::RowMove(RowMove {
                    starting_position: Position::A5,
                    ending_position: Position::B5,
                }),
                starting_position: Position::A5,
                color: Color::White,
                direction: Direction::NorthWest,
                num_same_color: 3,
                num_opposite_color: 0,
            })
        );
    }

    #[test]
    fn test_interpret_broadside_move() {
        let board = Board::starting_board();
        assert_eq!(
            board.interpret_broadside_move(BroadsideMove {
                group_start: Position::C3,
                group_end: Position::C5,
                group_start_final_position: Position::D4
            }),
            Some(InterpretedMove {
                piece_move: PieceMove::BroadsideMove(BroadsideMove {
                    group_start: Position::C3,
                    group_end: Position::C5,
                    group_start_final_position: Position::D4,
                }),
                starting_position: Position::C3,
                color: Color::White,
                direction: Direction::NorthEast,
                num_same_color: 1,
                num_opposite_color: 0
            })
        );
    }
}
