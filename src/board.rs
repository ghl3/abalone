// Representation of an Abalone board.

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Circle {
    EMPTY,
    BLACK,
    WHITE,
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
            row1: [Circle::EMPTY; 5],
            row2: [Circle::EMPTY; 6],
            row3: [Circle::EMPTY; 7],
            row4: [Circle::EMPTY; 8],
            row5: [Circle::EMPTY; 9],
            row6: [Circle::EMPTY; 8],
            row7: [Circle::EMPTY; 7],
            row8: [Circle::EMPTY; 6],
            row9: [Circle::EMPTY; 5],
        }
    }

    pub fn starting_board() -> Board {
        Board {
            row1: [
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
            ],
            row2: [
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
            ],
            row3: [
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::EMPTY,
                Circle::EMPTY,
            ],
            row4: [
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::EMPTY,
            ],
            row5: [
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::EMPTY,
            ],
            row6: [
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::WHITE,
                Circle::WHITE,
                Circle::WHITE,
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::EMPTY,
            ],
            row7: [
                Circle::EMPTY,
                Circle::EMPTY,
                Circle::BLACK,
                Circle::BLACK,
                Circle::BLACK,
                Circle::EMPTY,
                Circle::EMPTY,
            ],
            row8: [
                Circle::BLACK,
                Circle::BLACK,
                Circle::BLACK,
                Circle::BLACK,
                Circle::BLACK,
                Circle::BLACK,
            ],
            row9: [
                Circle::BLACK,
                Circle::BLACK,
                Circle::BLACK,
                Circle::BLACK,
                Circle::BLACK,
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

    pub fn circle(&self, row: Row, column: Column) -> Option<Circle> {
        if column > Board::get_highest_column(row) {
            Option::None
        } else {
            Option::Some(self.row(row)[Board::column_index(column)])
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::{Board, Circle, Column, Row};

    #[test]
    fn get_circle() {
        assert_eq!(
            Board::starting_board().circle(Row::ONE, Column::ONE),
            Some(Circle::WHITE)
        );
        assert_eq!(
            Board::starting_board().circle(Row::FIVE, Column::ONE),
            Some(Circle::EMPTY)
        );
        assert_eq!(
            Board::starting_board().circle(Row::NINE, Column::ONE),
            Some(Circle::BLACK)
        );
        assert_eq!(Board::starting_board().circle(Row::ONE, Column::NINE), None);
    }
}
