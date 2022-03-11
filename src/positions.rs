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

#[derive(Debug, Clone, Copy)]
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
}
