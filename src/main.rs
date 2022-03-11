#![feature(let_else)]

mod board;
mod game;
mod piece_move;
mod positions;

fn main() {
    let _b = board::Board::starting_board();
    println!("Hello, world!");
}
