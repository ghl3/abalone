use crate::board::{Board, Column, MoveResult, PieceMove, Row};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Turn {
    Black,
    White,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GameState {
    InProgress,
    BlackWins,
    WhiteWins,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Game {
    board: Board,
    turn: Turn,
    num_white_points: i8,
    num_black_points: i8,
    state: GameState,
}

impl Game {
    pub fn new_game() -> Game {
        Game {
            board: Board::empty_board(),
            turn: Turn::White,
            num_white_points: 0,
            num_black_points: 0,
            state: GameState::InProgress,
        }
    }

    pub fn apply_move(&mut self, piece_move: PieceMove) -> MoveResult {
        if piece_move.pieces.color() != self.turn {
            MoveResult::Invalid
        }

        match self.board.apply_move(piece_move) {
            MoveResult::Valid => {
                self.turn = (if turn == Turn::White {
                    Turn::Black
                } else {
                    Turn::White
                });
                MoveResult::Valid
            }
            _ => MoveResult::Invalid,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::{Board, Color, PieceGroup, PieceMove};
    use crate::game::{Game, GameState};

    #[test]
    fn empty_board() {
        let game = Game::new_game();
        assert_eq!(game.state, GameState::InProgress)

        //game.apply_move(PieceMove{pieces:
        //PieceGroup::Single(Color::White,
        )
    }
}
