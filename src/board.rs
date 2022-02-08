// Representation of an Abalone board.

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

#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum Row {
    ONE,
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,
}

#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum Column {
    ONE,
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,
}

// Number of spaces:   61
//     X X X X X       5
//    X X X X X X      6
//   X X X X X X X     7
//  X X X X X X X X    8
// X X X X X X X X X   9
//  X X X X X X X X    8
//   X X X X X X X     7
//    X X X X X X      6
//     X X X X X       5
pub struct Board {
    row1: [Circle; 5],
    row2: [Circle; 6],
    row3: [Circle; 7],
    row4: [Circle; 8],
    row5: [Circle; 9],
    row6: [Circle; 8],
    row7: [Circle; 7],
    row8: [Circle; 6],
    row9: [Circle; 5],
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    NorthEast,
    NorthWest,
    East,
    SouthEast,
    SouthWest,
    West,
}

#[derive(Debug, Clone, Copy)]
pub struct Position {
    row: Row,
    column: Column,
}

#[derive(Debug, Clone, Copy)]
pub enum Span {
    // A single circle
    Single(Color, Position),
    // Two connected circles, specified by the
    // position of one circle and the direction to the connected circle
    Double(Color, Position, Direction),
    // Three connected circles, specified by the
    // position of one circle and the direction to the two connected circle
    Triple(Color, Position, Direction),
}

#[derive(Debug, Clone, Copy)]
pub struct PieceMove {
    // The position of the connected circles
    span: Span,
    // The movement direction
    direction: Direction,
}

#[derive(Debug, Clone, Copy)]
pub enum MoveResult {
    Invalid,
    Valid,
}

// Implementation block, all `Point` associated functions & methods go in here
impl Board {
    pub fn column_index(col: Column) -> usize {
        match col {
            Column::ONE => 1,
            Column::TWO => 2,
            Column::THREE => 3,
            Column::FOUR => 4,
            Column::FIVE => 5,
            Column::SIX => 6,
            Column::SEVEN => 7,
            Column::EIGHT => 8,
            Column::NINE => 9,
        }
    }

    pub fn get_highest_column(row: Row) -> Column {
        match row {
            Row::ONE => Column::FIVE,
            Row::TWO => Column::SIX,
            Row::THREE => Column::SEVEN,
            Row::FOUR => Column::EIGHT,
            Row::FIVE => Column::NINE,
            Row::SIX => Column::EIGHT,
            Row::SEVEN => Column::SEVEN,
            Row::EIGHT => Column::SIX,
            Row::NINE => Column::FIVE,
        }
    }

    pub fn empty_board() -> Board {
        Board {
            row1: [Circle::Empty; 5],
            row2: [Circle::Empty; 6],
            row3: [Circle::Empty; 7],
            row4: [Circle::Empty; 8],
            row5: [Circle::Empty; 9],
            row6: [Circle::Empty; 8],
            row7: [Circle::Empty; 7],
            row8: [Circle::Empty; 6],
            row9: [Circle::Empty; 5],
        }
    }

    pub fn starting_board() -> Board {
        Board {
            row1: [
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
            ],
            row2: [
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
            ],
            row3: [
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
            ],
            row4: [
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
            ],
            row5: [
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
            ],
            row6: [
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Filled(Color::White),
                Circle::Empty,
                Circle::Empty,
                Circle::Empty,
            ],
            row7: [
                Circle::Empty,
                Circle::Empty,
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Empty,
                Circle::Empty,
            ],
            row8: [
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
            ],
            row9: [
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
                Circle::Filled(Color::Black),
            ],
        }
    }

    pub fn row(&self, row: Row) -> &[Circle] {
        match row {
            Row::ONE => &self.row1,
            Row::TWO => &self.row2,
            Row::THREE => &self.row3,
            Row::FOUR => &self.row4,
            Row::FIVE => &self.row5,
            Row::SIX => &self.row6,
            Row::SEVEN => &self.row8,
            Row::EIGHT => &self.row8,
            Row::NINE => &self.row9,
        }
    }

    pub fn circle(&self, Position { row, column }: Position) -> Option<Circle> {
        if column > Board::get_highest_column(row) {
            Option::None
        } else {
            Option::Some(self.row(row)[Board::column_index(column)])
        }
    }

    pub fn get_new_position(position: Position, direction: Direction, num: i8) -> Option<Position> {
        Option::None
    }

    pub fn span_exists(&self, span: Span) -> bool {
        match span {
            Span::Single(color, position) => self.circle(position) == Some(Circle::Filled(color)),
            Span::Double(color, position, direction) => {
                match Board::get_new_position(position, direction, 1) {
                    None => false,
                    Some(new_position) => self.circle(new_position) == Some(Circle::Filled(color)),
                }
            }
            Span::Triple(color, position, direction) => {
                match (
                    Board::get_new_position(position, direction, 1),
                    Board::get_new_position(position, direction, 2),
                ) {
                    (Some(first_position), Some(second_position)) => {
                        self.circle(first_position) == Some(Circle::Filled(color))
                            && self.circle(second_position) == Some(Circle::Filled(color))
                    }
                    _ => false,
                }
            }
        }
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
