struct MyRange {
    start: u64,
    end: u64,
}

impl MyRange {
    fn new(start: u64, end: u64) -> MyRange {
        MyRange {
            start: start,
            end: end,
        }
    }
}

impl Iterator for MyRange {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.start == self.end {
            None
        } else {
            let result = Some(self.start);
            self.start += 1;
            result
        }
    }
}

fn main() {
    let sum: u64 = MyRange::new(0, 1_000_000).sum();
    println!("{}", sum)
}

/*
struct PositionAndDirection {
    position: Position,
    direction: Direction,
}

impl Iterator for PositionAndDirection {
    type Item = Position;

    fn next(&mut self) -> Option<Position> {


        match self.direction {
            Direction::NorthEast => {}
            Direction::NorthWest => {}
            Direction::East => {



            }
            Direction::SouthEast => {}
            Direction::SouthWest => {}
            Direction::West => {}
        }


    }
}
*/
