use crate::board::{Board, Column, Row};

pub enum Turn {
    Black,
    White,
}

pub enum State {
    InProgress,
    BlackWins,
    WhiteWins,
}

pub struct Game {
    board: Board,
    turn: Turn,
    num_black_points: i8,
    num_white_points: i8,
    state: State,
}
