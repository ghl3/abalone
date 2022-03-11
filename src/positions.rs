// The notation for recording moves gives the letters A-I to the horizontal lines, and the numbers 1-9 to northwest-southeast diagonals.
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

#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum Row {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
}

impl Row {
    pub fn index(&self) -> usize {
        match &self {
            Row::A => 0,
            Row::B => 1,
            Row::C => 2,
            Row::D => 3,
            Row::E => 4,
            Row::F => 5,
            Row::G => 6,
            Row::H => 7,
            Row::I => 8,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum Diagonal {
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

impl Diagonal {
    pub fn index(&self) -> usize {
        match &self {
            Diagonal::ONE => 0,
            Diagonal::TWO => 1,
            Diagonal::THREE => 2,
            Diagonal::FOUR => 3,
            Diagonal::FIVE => 4,
            Diagonal::SIX => 5,
            Diagonal::SEVEN => 6,
            Diagonal::EIGHT => 7,
            Diagonal::NINE => 8,
        }
    }

    pub fn get_highest_row(&self) -> Row {
        match self {
            Diagonal::ONE => Row::E,
            Diagonal::TWO => Row::F,
            Diagonal::THREE => Row::G,
            Diagonal::FOUR => Row::H,
            Diagonal::FIVE => Row::I,
            Diagonal::SIX => Row::I,
            Diagonal::SEVEN => Row::I,
            Diagonal::EIGHT => Row::I,
            Diagonal::NINE => Row::I,
        }
    }

    pub fn get_front_buffer(&self) -> usize {
        match self {
            Diagonal::ONE => 0,
            Diagonal::TWO => 0,
            Diagonal::THREE => 0,
            Diagonal::FOUR => 0,
            Diagonal::FIVE => 0,
            Diagonal::SIX => 1,
            Diagonal::SEVEN => 2,
            Diagonal::EIGHT => 3,
            Diagonal::NINE => 4,
        }
    }

    pub fn get_back_buffer(&self) -> usize {
        match self {
            Diagonal::ONE => 4,
            Diagonal::TWO => 3,
            Diagonal::THREE => 2,
            Diagonal::FOUR => 1,
            Diagonal::FIVE => 0,
            Diagonal::SIX => 0,
            Diagonal::SEVEN => 0,
            Diagonal::EIGHT => 0,
            Diagonal::NINE => 0,
        }
    }

    pub fn get_index_of_row(&self, row: Row) -> usize {
        return row.index() - self.get_front_buffer();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    NorthEast,
    NorthWest,
    East,
    SouthEast,
    SouthWest,
    West,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Position {
    A1,
    A2,
    A3,
    A4,
    A5,

    B1,
    B2,
    B3,
    B4,
    B5,
    B6,

    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,

    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    D8,

    E1,
    E2,
    E3,
    E4,
    E5,
    E6,
    E7,
    E8,
    E9,

    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,

    G3,
    G4,
    G5,
    G6,
    G7,
    G8,
    G9,

    H4,
    H5,
    H6,
    H7,
    H8,
    H9,

    I5,
    I6,
    I7,
    I8,
    I9,
}

impl Position {
    pub fn get_diagonal_row(&self) -> (Diagonal, Row) {
        match self {
            Position::A1 => (Diagonal::ONE, Row::A),
            Position::A2 => (Diagonal::TWO, Row::A),
            Position::A3 => (Diagonal::THREE, Row::A),
            Position::A4 => (Diagonal::FOUR, Row::A),
            Position::A5 => (Diagonal::FIVE, Row::A),
            Position::B1 => (Diagonal::ONE, Row::B),
            Position::B2 => (Diagonal::TWO, Row::B),
            Position::B3 => (Diagonal::THREE, Row::B),
            Position::B4 => (Diagonal::FOUR, Row::B),
            Position::B5 => (Diagonal::FIVE, Row::B),
            Position::B6 => (Diagonal::SIX, Row::B),
            Position::C1 => (Diagonal::ONE, Row::C),
            Position::C2 => (Diagonal::TWO, Row::C),
            Position::C3 => (Diagonal::THREE, Row::C),
            Position::C4 => (Diagonal::FOUR, Row::C),
            Position::C5 => (Diagonal::FIVE, Row::C),
            Position::C6 => (Diagonal::SIX, Row::C),
            Position::C7 => (Diagonal::SEVEN, Row::C),
            Position::D1 => (Diagonal::ONE, Row::D),
            Position::D2 => (Diagonal::TWO, Row::D),
            Position::D3 => (Diagonal::THREE, Row::D),
            Position::D4 => (Diagonal::FOUR, Row::D),
            Position::D5 => (Diagonal::FIVE, Row::D),
            Position::D6 => (Diagonal::SIX, Row::D),
            Position::D7 => (Diagonal::SEVEN, Row::D),
            Position::D8 => (Diagonal::EIGHT, Row::D),
            Position::E1 => (Diagonal::ONE, Row::E),
            Position::E2 => (Diagonal::TWO, Row::E),
            Position::E3 => (Diagonal::THREE, Row::E),
            Position::E4 => (Diagonal::FOUR, Row::E),
            Position::E5 => (Diagonal::FIVE, Row::E),
            Position::E6 => (Diagonal::SIX, Row::E),
            Position::E7 => (Diagonal::SEVEN, Row::E),
            Position::E8 => (Diagonal::EIGHT, Row::E),
            Position::E9 => (Diagonal::NINE, Row::E),
            Position::F2 => (Diagonal::TWO, Row::F),
            Position::F3 => (Diagonal::THREE, Row::F),
            Position::F4 => (Diagonal::FOUR, Row::F),
            Position::F5 => (Diagonal::FIVE, Row::F),
            Position::F6 => (Diagonal::SIX, Row::F),
            Position::F7 => (Diagonal::SEVEN, Row::F),
            Position::F8 => (Diagonal::EIGHT, Row::F),
            Position::F9 => (Diagonal::NINE, Row::F),
            Position::G3 => (Diagonal::THREE, Row::G),
            Position::G4 => (Diagonal::FOUR, Row::G),
            Position::G5 => (Diagonal::FIVE, Row::G),
            Position::G6 => (Diagonal::SIX, Row::G),
            Position::G7 => (Diagonal::SEVEN, Row::G),
            Position::G8 => (Diagonal::EIGHT, Row::G),
            Position::G9 => (Diagonal::NINE, Row::G),
            Position::H4 => (Diagonal::FOUR, Row::H),
            Position::H5 => (Diagonal::FIVE, Row::H),
            Position::H6 => (Diagonal::SIX, Row::H),
            Position::H7 => (Diagonal::SEVEN, Row::H),
            Position::H8 => (Diagonal::EIGHT, Row::H),
            Position::H9 => (Diagonal::NINE, Row::H),
            Position::I5 => (Diagonal::FIVE, Row::I),
            Position::I6 => (Diagonal::SIX, Row::I),
            Position::I7 => (Diagonal::SEVEN, Row::I),
            Position::I8 => (Diagonal::EIGHT, Row::I),
            Position::I9 => (Diagonal::NINE, Row::I),
        }
    }

    fn row(&self) -> Row {
        let (_, row) = self.get_diagonal_row();
        row
    }

    fn diagonal(&self) -> Diagonal {
        let (diagonal, _) = self.get_diagonal_row();
        diagonal
    }

    fn has_same_row(&self, other: Position) -> bool {
        let (_, row) = self.get_diagonal_row();
        let (_, other_row) = other.get_diagonal_row();

        row == other_row
    }

    fn has_same_diagonal(&self, other: Position) -> bool {
        let (diagonal, row) = self.get_diagonal_row();
        let (other_diagonal, other_row) = other.get_diagonal_row();

        if diagonal == other_diagonal {
            return true;
        }

        let (diagonal_index, row_index) = (diagonal.index(), row.index());
        let (other_diagonal_index, other_row_index) = (other_diagonal.index(), other_row.index());

        if (other_diagonal_index > diagonal_index) {
            return other_row_index > row_index
                && other_diagonal_index - diagonal_index == other_row_index - row_index;
        } else {
            return row_index > other_row_index
                && diagonal_index - other_diagonal_index == row_index - other_row_index;
        }

        //    return other_diagonal_index - diagonal_index == other_row_index - row_index;
    }

    // Given a starting and an ending point, returns the direction and number of circles
    // needed to traverse to go from the start to the end point.  If there is no straight line
    // between the two, returns an empty optional.
    pub fn get_direction_and_distance(&self, other: Position) -> Option<(Direction, usize)> {
        if other == *self {
            return Option::None;
        } else if self.has_same_row(other) {
            if self.diagonal() > other.diagonal() {
                let delta: usize = self.diagonal().index() - other.diagonal().index();
                Option::Some((Direction::West, delta))
            } else {
                let delta: usize = other.diagonal().index() - self.diagonal().index();
                Option::Some((Direction::East, delta))
            }
        } else if self.has_same_diagonal(other) && self.diagonal() == other.diagonal() {
            // They are on the same North-West to South-East Diagonal
            if other.row() > self.row() {
                let delta: usize = other.row().index() - self.row().index();
                Option::Some((Direction::NorthWest, delta))
            } else {
                let delta: usize = self.row().index() - other.row().index();
                Option::Some((Direction::SouthEast, delta))
            }
        } else if self.has_same_diagonal(other) {
            // They are on the same North-West to South-East Diagonal
            if other.row() > self.row() {
                let delta: usize = other.row().index() - self.row().index();
                Option::Some((Direction::NorthEast, delta))
            } else {
                let delta: usize = self.row().index() - other.row().index();
                Option::Some((Direction::SouthWest, delta))
            }
        } else {
            Option::None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::positions::{Direction, Position};

    #[test]
    fn test_same_row() {
        assert!(Position::A1.has_same_row(Position::A2));
        assert!(!Position::A1.has_same_row(Position::B2));
        assert!(Position::H4.has_same_row(Position::H8));
    }

    #[test]
    fn test_same_diagonal() {
        assert!(Position::A1.has_same_diagonal(Position::B2));
        assert!(Position::A3.has_same_diagonal(Position::B3));
        assert!(Position::A1.has_same_diagonal(Position::B2));
        assert!(Position::A1.has_same_diagonal(Position::I9));
        assert!(Position::E1.has_same_diagonal(Position::I5));

        assert!(!Position::A1.has_same_diagonal(Position::A2));
        assert!(!Position::B2.has_same_diagonal(Position::G4));
    }

    #[test]
    fn test_get_direction_and_distance() {
        assert_eq!(
            Position::A1.get_direction_and_distance(Position::A2),
            Some((Direction::East, 1))
        );

        assert_eq!(
            Position::A1.get_direction_and_distance(Position::A5),
            Some((Direction::East, 4))
        );

        assert_eq!(
            Position::F2.get_direction_and_distance(Position::F9),
            Some((Direction::East, 7))
        );

        assert_eq!(
            Position::E6.get_direction_and_distance(Position::E2),
            Some((Direction::West, 4))
        );

        assert_eq!(Position::F2.get_direction_and_distance(Position::G9), None);

        assert_eq!(
            Position::A1.get_direction_and_distance(Position::B2),
            Some((Direction::NorthEast, 1))
        );

        assert_eq!(
            Position::B4.get_direction_and_distance(Position::G4),
            Some((Direction::NorthWest, 5))
        );

        assert_eq!(
            Position::H6.get_direction_and_distance(Position::E6),
            Some((Direction::SouthEast, 3))
        );

        assert_eq!(
            Position::B2.get_direction_and_distance(Position::A1),
            Some((Direction::SouthWest, 1))
        );
    }
}
