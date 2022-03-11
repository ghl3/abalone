use crate::board::{Board, MoveResult};
use crate::piece_move::{Color, PieceMove};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GameState {
    InProgress,
    BlackWins,
    WhiteWins,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Game {
    board: Board,
    turn: Color,
    num_white_points: i8,
    num_black_points: i8,
    state: GameState,
}

impl Game {
    pub fn new_game() -> Game {
        Game {
            board: Board::empty_board(),
            turn: Color::White,
            num_white_points: 0,
            num_black_points: 0,
            state: GameState::InProgress,
        }
    }

    pub fn apply_move(&mut self, piece_move: PieceMove) -> MoveResult {
        let MoveResult::Valid(interpreted_move) = self.board.apply_move(piece_move) else {
            return MoveResult::Invalid;
        };

        if interpreted_move.color != self.turn {
            return MoveResult::Invalid;
        };

        self.turn = if self.turn == Color::Black {
            Color::White
        } else {
            Color::Black
        };

        MoveResult::Valid(interpreted_move)
    }
}

#[cfg(test)]
mod tests {
    use crate::game::{Game, GameState};

    #[test]
    fn empty_board() {
        let game = Game::new_game();
        assert_eq!(game.state, GameState::InProgress);

        //game.apply_move(PieceMove{pieces:
        //PieceGroup::Single(Color::White,
    }
}
