#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    let mut reader = StdinReader::new(stdin.lock());
    let n = reader.u();
    let a = reader.uv(n);

    let mut index_map = vec![0; n + 1];
    for i in 0..n {
        index_map[a[i]] = i + 1;
    }

    let mut set = BTreeSet::new();
    set.insert(0);
    set.insert(n + 1);

    let mut ans = 0;
    for i in 1..=n {
        let index = index_map[i];
        let l = set.range(..index).last().unwrap();
        let r = set.range((index + 1)..).next().unwrap();
        ans += (index - l) * (r - index) * i;
        set.insert(index);
    }
    println!("{}", ans);
}

#[allow(unused_imports)]
use itertools::Itertools;
#[allow(unused_imports)]
use std::{cmp::*, collections::*, io::*, num::*, str::*};
#[allow(unused_imports)]
use stdin_reader::StdinReader;

#[allow(dead_code)]
mod stdin_reader {
    use std::{fmt::Debug, io::*, str::*};

    pub struct StdinReader<R: BufRead> {
        reader: R,
        buf: Vec<u8>,
        // Should never be empty
        pos: usize, // Should never be out of bounds as long as the input ends with '\n'
    }

    impl<R: BufRead> StdinReader<R> {
        pub fn new(reader: R) -> StdinReader<R> {
            let (buf, pos) = (Vec::new(), 0);
            StdinReader { reader, buf, pos }
        }
        pub fn n<T: FromStr>(&mut self) -> T
        where
            T::Err: Debug,
        {
            if self.buf.is_empty() {
                self._read_next_line();
            }
            let mut start = None;
            while self.pos != self.buf.len() {
                match (self.buf[self.pos], start.is_some()) {
                    (b' ', true) | (b'\n', true) => break,
                    (_, true) | (b' ', false) => self.pos += 1,
                    (b'\n', false) => self._read_next_line(),
                    (_, false) => start = Some(self.pos),
                }
            }
            let target = &self.buf[start.unwrap()..self.pos];
            from_utf8(target).unwrap().parse().unwrap()
        }

        fn _read_next_line(&mut self) {
            self.pos = 0;
            self.buf.clear();
            if self.reader.read_until(b'\n', &mut self.buf).unwrap() == 0 {
                panic!("Reached EOF");
            }
        }

        pub fn str(&mut self) -> String {
            self.n()
        }
        pub fn s(&mut self) -> Vec<char> {
            self.n::<String>().chars().collect()
        }
        pub fn i(&mut self) -> i64 {
            self.n()
        }
        pub fn i2(&mut self) -> (i64, i64) {
            (self.n(), self.n())
        }
        pub fn i3(&mut self) -> (i64, i64, i64) {
            (self.n(), self.n(), self.n())
        }
        pub fn u(&mut self) -> usize {
            self.n()
        }
        pub fn u2(&mut self) -> (usize, usize) {
            (self.n(), self.n())
        }
        pub fn u3(&mut self) -> (usize, usize, usize) {
            (self.n(), self.n(), self.n())
        }
        pub fn u4(&mut self) -> (usize, usize, usize, usize) {
            (self.n(), self.n(), self.n(), self.n())
        }
        pub fn u5(&mut self) -> (usize, usize, usize, usize, usize) {
            (self.n(), self.n(), self.n(), self.n(), self.n())
        }
        pub fn u6(&mut self) -> (usize, usize, usize, usize, usize, usize) {
            (self.n(), self.n(), self.n(), self.n(), self.n(), self.n())
        }
        pub fn f(&mut self) -> f64 {
            self.n()
        }
        pub fn f2(&mut self) -> (f64, f64) {
            (self.n(), self.n())
        }
        pub fn c(&mut self) -> char {
            self.n::<String>().pop().unwrap()
        }
        pub fn iv(&mut self, n: usize) -> Vec<i64> {
            (0..n).map(|_| self.i()).collect()
        }
        pub fn iv2(&mut self, n: usize) -> Vec<(i64, i64)> {
            (0..n).map(|_| self.i2()).collect()
        }
        pub fn iv3(&mut self, n: usize) -> Vec<(i64, i64, i64)> {
            (0..n).map(|_| self.i3()).collect()
        }
        pub fn uv(&mut self, n: usize) -> Vec<usize> {
            (0..n).map(|_| self.u()).collect()
        }
        pub fn uv2(&mut self, n: usize) -> Vec<(usize, usize)> {
            (0..n).map(|_| self.u2()).collect()
        }
        pub fn uv3(&mut self, n: usize) -> Vec<(usize, usize, usize)> {
            (0..n).map(|_| self.u3()).collect()
        }
        pub fn uv4(&mut self, n: usize) -> Vec<(usize, usize, usize, usize)> {
            (0..n).map(|_| self.u4()).collect()
        }
        pub fn fv(&mut self, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.f()).collect()
        }
        pub fn cmap(&mut self, h: usize) -> Vec<Vec<char>> {
            (0..h).map(|_| self.s()).collect()
        }
    }
}
