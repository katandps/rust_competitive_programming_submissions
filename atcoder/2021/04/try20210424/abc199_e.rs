#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    solve(Reader::new(stdin.lock()));
}

pub fn solve<R: BufRead>(mut reader: Reader<R>) {
    let (n, m) = reader.u2();

    let xyz = reader.uv3(m);

    // limit[x][y] = z;
    let mut limit = vec![vec![n; n + 1]; n + 1];

    for (x, y, z) in xyz {
        limit[x][y] = min(limit[x][y], z);
    }

    let mut dp = vec![vec![0usize; 1usize << n]; n + 1];
    dp[0][0] = 1;

    for i in 0..n {
        // 次の状態
        's: for new_s in 0..1usize << n {
            if new_s.count_ones() != i as u32 + 1 {
                continue;
            }
            for y in 0..n {
                // y以下の数をすべて含んだ数
                let p = (1 << y) - 1;
                if limit[i + 1][y] < (new_s & p).count_ones() as usize {
                    // dbg!(new_s, p);
                    // dbg!(limit[i + 1][y], (new_s & p).count_ones() as usize);
                    continue 's;
                };
            }
            // 次にとるやつ
            for p in 0..n {
                if ((1 << p) & new_s) == 0 {
                    continue;
                }
                dp[i + 1][new_s] += dp[i][new_s ^ (1 << p)];
            }
        }
    }
    // dbg!(&dp);
    println!("{}", dp[n][(1usize << n) - 1]);
}

pub use reader::*;
#[allow(unused_imports)]
use {
    itertools::Itertools,
    num::Integer,
    std::{cmp::*, collections::*, io::*, num::*, str::*},
};

#[allow(dead_code)]
#[rustfmt::skip]
pub mod reader {
    #[allow(unused_imports)]
    use itertools::Itertools;
    use std::{fmt::Debug, io::*, str::*};

    pub struct Reader<R: BufRead> {
        reader: R,
        buf: Vec<u8>,
        pos: usize,
    }  macro_rules! prim_method { ($name:ident: $T: ty) => { pub fn $name(&mut self) -> $T { self.n::<$T>() } }; ($name:ident) => { prim_method!($name: $name); }; } macro_rules! prim_methods { ($name:ident: $T:ty; $($rest:tt)*) => { prim_method!($name:$T); prim_methods!($($rest)*); }; ($name:ident; $($rest:tt)*) => { prim_method!($name); prim_methods!($($rest)*); }; () => () }  macro_rules! replace_expr { ($_t:tt $sub:expr) => { $sub }; } macro_rules! tuple_method { ($name: ident: ($($T:ident),+)) => { pub fn $name(&mut self) -> ($($T),+) { ($(replace_expr!($T self.n())),+) } } } macro_rules! tuple_methods { ($name:ident: ($($T:ident),+); $($rest:tt)*) => { tuple_method!($name:($($T),+)); tuple_methods!($($rest)*); }; () => () } macro_rules! vec_method { ($name: ident: ($($T:ty),+)) => { pub fn $name(&mut self, n: usize) -> Vec<($($T),+)> { (0..n).map(|_|($(replace_expr!($T self.n())),+)).collect_vec() } }; ($name: ident: $T:ty) => { pub fn $name(&mut self, n: usize) -> Vec<$T> { (0..n).map(|_|self.n()).collect_vec() } }; } macro_rules! vec_methods { ($name:ident: ($($T:ty),+); $($rest:tt)*) => { vec_method!($name:($($T),+)); vec_methods!($($rest)*); }; ($name:ident: $T:ty; $($rest:tt)*) => { vec_method!($name:$T); vec_methods!($($rest)*); }; () => () } impl<R: BufRead> Reader<R> {
    pub fn new(reader: R) -> Reader<R> {
        let (buf, pos) = (Vec::new(), 0);
        Reader { reader, buf, pos }
    } prim_methods! { u: usize; i: i64; f: f64; str: String; c: char; string: String; u8; u16; u32; u64; u128; usize; i8; i16; i32; i64; i128; isize; f32; f64; char; } tuple_methods! { u2: (usize, usize); u3: (usize, usize, usize); u4: (usize, usize, usize, usize); i2: (i64, i64); i3: (i64, i64, i64); i4: (i64, i64, i64, i64); cuu: (char, usize, usize); } vec_methods! { uv: usize; uv2: (usize, usize); uv3: (usize, usize, usize); iv: i64; iv2: (i64, i64); iv3: (i64, i64, i64); vq: (char, usize, usize); }  pub fn n<T: FromStr>(&mut self) -> T where T::Err: Debug, { self.n_op().unwrap() }
    pub fn n_op<T: FromStr>(&mut self) -> Option<T> where T::Err: Debug, {
        if self.buf.is_empty() { self._read_next_line(); }
        let mut start = None;
        while self.pos != self.buf.len() {
            match (self.buf[self.pos], start.is_some()) {
                (b' ', true) | (b'\n', true) => break,
                (_, true) | (b' ', false) => self.pos += 1,
                (b'\n', false) => self._read_next_line(),
                (_, false) => start = Some(self.pos),
            }
        }
        start.map(|s| from_utf8(&self.buf[s..self.pos]).unwrap().parse().unwrap())
    }
    fn _read_next_line(&mut self) {
        self.pos = 0;
        self.buf.clear();
        self.reader.read_until(b'\n', &mut self.buf).unwrap();
    }
    pub fn s(&mut self) -> Vec<char> { self.n::<String>().chars().collect() }
    pub fn char_map(&mut self, h: usize) -> Vec<Vec<char>> { (0..h).map(|_| self.s()).collect() }
    pub fn bool_map(&mut self, h: usize, ng: char) -> Vec<Vec<bool>> { self.char_map(h).iter().map(|v| v.iter().map(|&c| c != ng).collect()).collect() }
    pub fn matrix(&mut self, h: usize, w: usize) -> Vec<Vec<i64>> { (0..h).map(|_| self.iv(w)).collect() }
}
}