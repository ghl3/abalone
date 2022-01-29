// Representation of an Abalone board
// Optimized for speed.

//struct Row<const ROWSIZE: usize> {
//    circles: [u8; ROWSIZE]
//}

type Circle = u8;

// Number of spaces: 61
//     X X X X X
//    X X X X X X
//   X X X X X X X
//  X X X X X X X X
// X X X X X X X X X
//  X X X X X X X X
//   X X X X X X X
//    X X X X X X
//     X X X X X
pub struct Board {
    circles: [Circle; 61],
}

// Implementation block, all `Point` associated functions & methods go in here
impl Board {
    pub fn new() -> Board {
        Board { circles: [0; 61] }
    }

    pub fn row(&self) -> &[Circle] {
        &self.circles[0..10]
    }
}
