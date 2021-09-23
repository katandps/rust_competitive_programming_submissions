use crate::binary_indexed_tree_2d::BinaryIndexedTree2;
pub use reader::*;
#[allow(unused_imports)]
use {
    itertools::Itertools,
    num::Integer,
    proconio::fastout,
    std::convert::TryInto,
    std::{cmp::*, collections::*, io::*, num::*, str::*},
};

#[allow(unused_macros)]
macro_rules! chmin {($base:expr, $($cmps:expr),+ $(,)*) => {{let cmp_min = min!($($cmps),+);if $base > cmp_min {$base = cmp_min;true} else {false}}};}
#[allow(unused_macros)]
macro_rules! chmax {($base:expr, $($cmps:expr),+ $(,)*) => {{let cmp_max = max!($($cmps),+);if $base < cmp_max {$base = cmp_max;true} else {false}}};}
#[allow(unused_macros)]
macro_rules! min {
    ($a:expr $(,)*) => {{$a}};
    ($a:expr, $b:expr $(,)*) => {{if $a > $b {$b} else {$a}}};
    ($a:expr, $($rest:expr),+ $(,)*) => {{let b = min!($($rest),+);if $a > b {b} else {$a}}};
}
#[allow(unused_macros)]
macro_rules! max {
    ($a:expr $(,)*) => {{$a}};
    ($a:expr, $b:expr $(,)*) => {{if $a > $b {$a} else {$b}}};
    ($a:expr, $($rest:expr),+ $(,)*) => {{let b = max!($($rest),+);if $a > b {$a} else {b}}};
}

#[allow(dead_code)]
#[rustfmt::skip]
pub mod reader { #[allow(unused_imports)] use itertools::Itertools; use std::{fmt::Debug, io::*, str::*};  pub struct Reader<R: BufRead> { reader: R, buf: Vec<u8>, pos: usize, }  macro_rules! prim_method { ($name:ident: $T: ty) => { pub fn $name(&mut self) -> $T { self.n::<$T>() } }; ($name:ident) => { prim_method!($name: $name); } } macro_rules! prim_methods { ($name:ident: $T:ty; $($rest:tt)*) => { prim_method!($name:$T); prim_methods!($($rest)*); }; ($name:ident; $($rest:tt)*) => { prim_method!($name); prim_methods!($($rest)*); }; () => () }  macro_rules! replace_expr { ($_t:tt $sub:expr) => { $sub }; } macro_rules! tuple_method { ($name: ident: ($($T:ident),+)) => { pub fn $name(&mut self) -> ($($T),+) { ($(replace_expr!($T self.n())),+) } } } macro_rules! tuple_methods { ($name:ident: ($($T:ident),+); $($rest:tt)*) => { tuple_method!($name:($($T),+)); tuple_methods!($($rest)*); }; () => () } macro_rules! vec_method { ($name: ident: ($($T:ty),+)) => { pub fn $name(&mut self, n: usize) -> Vec<($($T),+)> { (0..n).map(|_|($(replace_expr!($T self.n())),+)).collect_vec() } }; ($name: ident: $T:ty) => { pub fn $name(&mut self, n: usize) -> Vec<$T> { (0..n).map(|_|self.n()).collect_vec() } }; } macro_rules! vec_methods { ($name:ident: ($($T:ty),+); $($rest:tt)*) => { vec_method!($name:($($T),+)); vec_methods!($($rest)*); }; ($name:ident: $T:ty; $($rest:tt)*) => { vec_method!($name:$T); vec_methods!($($rest)*); }; () => () } impl<R: BufRead> Reader<R> { pub fn new(reader: R) -> Reader<R> { let (buf, pos) = (Vec::new(), 0); Reader { reader, buf, pos } } prim_methods! { u: usize; i: i64; f: f64; str: String; c: char; string: String; u8; u16; u32; u64; u128; usize; i8; i16; i32; i64; i128; isize; f32; f64; char; } tuple_methods! { u2: (usize, usize); u3: (usize, usize, usize); u4: (usize, usize, usize, usize); i2: (i64, i64); i3: (i64, i64, i64); i4: (i64, i64, i64, i64); cuu: (char, usize, usize); } vec_methods! { uv: usize; uv2: (usize, usize); uv3: (usize, usize, usize); iv: i64; iv2: (i64, i64); iv3: (i64, i64, i64); vq: (char, usize, usize); }  pub fn n<T: FromStr>(&mut self) -> T where T::Err: Debug, { self.n_op().unwrap() }  pub fn n_op<T: FromStr>(&mut self) -> Option<T> where T::Err: Debug, { if self.buf.is_empty() { self._read_next_line(); } let mut start = None; while self.pos != self.buf.len() { match (self.buf[self.pos], start.is_some()) { (b' ', true) | (b'\n', true) => break, (_, true) | (b' ', false) => self.pos += 1, (b'\n', false) => self._read_next_line(), (_, false) => start = Some(self.pos), } } start.map(|s| from_utf8(&self.buf[s..self.pos]).unwrap().parse().unwrap()) }  fn _read_next_line(&mut self) { self.pos = 0; self.buf.clear(); self.reader.read_until(b'\n', &mut self.buf).unwrap(); } pub fn s(&mut self) -> Vec<char> { self.n::<String>().chars().collect() } pub fn digits(&mut self) -> Vec<i64> { self.n::<String>() .chars() .map(|c| (c as u8 - b'0') as i64) .collect() } pub fn char_map(&mut self, h: usize) -> Vec<Vec<char>> { (0..h).map(|_| self.s()).collect() } pub fn bool_map(&mut self, h: usize, ng: char) -> Vec<Vec<bool>> { self.char_map(h) .iter() .map(|v| v.iter().map(|&c| c != ng).collect()) .collect() } pub fn matrix(&mut self, h: usize, w: usize) -> Vec<Vec<i64>> { (0..h).map(|_| self.iv(w)).collect() } } }

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    solve(Reader::new(stdin.lock()));
}

#[fastout]
pub fn solve<R: BufRead>(mut reader: Reader<R>) {
    let n = reader.u();
    let mut bit2 = BinaryIndexedTree2::new(10, 10);
    for _ in 0..n {
        let (x1, y1, x2, y2) = reader.u4();
        bit2.add(y2, x2, 1);
        bit2.add(y1, x2, -1);
        bit2.add(y2, x1, -1);
        bit2.add(y1, x1, 1);
    }

    let mut ans = vec![0; n + 1];
    for y in 0..10 {
        for x in 0..10 {
            let k = bit2.sum(y, x);
            assert!(k >= 0);
            ans[k as usize] += 1;
        }
    }
    dbg!(bit2);
    for i in 1..=n {
        println!("{}", ans[i]);
    }
}

/// verified by https://atcoder.jp/contests/typical90/tasks/typical90_ab

pub mod binary_indexed_tree_2d {
    pub struct BinaryIndexedTree2 {
        h: usize,
        w: usize,
        bit: Vec<Vec<VALUE>>,
    }

    type VALUE = i64;

    impl BinaryIndexedTree2 {
        pub fn new(h: usize, w: usize) -> BinaryIndexedTree2 {
            let (h, w) = (h + 1, w + 1);
            let bit = vec![vec![0; w]; h];
            BinaryIndexedTree2 { h, w, bit }
        }

        pub fn add(&mut self, y: usize, x: usize, v: VALUE) {
            let mut idx = x as i32 + 1;
            while idx < self.w as i32 {
                let mut idy = y as i32 + 1;
                while idy < self.h as i32 {
                    self.bit[idy as usize][idx as usize] += v;
                    idy += idy & -idy;
                }
                idx += idx & -idx;
            }
        }

        /// sum of 0 <= y <= h & 0 <= x <= w
        pub fn sum(&self, y: usize, x: usize) -> VALUE {
            let mut ret = 0;
            let mut idx = x as i32 + 1;
            while idx > 0 {
                let mut idy = y as i32 + 1;
                while idy > 0 {
                    ret += self.bit[idy as usize][idx as usize];
                    idy -= idy & -idy;
                }
                idx -= idx & -idx;
            }
            ret
        }

        pub fn sum_ab(&self, (y1, x1): (usize, usize), (y2, x2): (usize, usize)) -> VALUE {
            self.sum(y2, x2) - self.sum(y2, x1) - self.sum(y1, x2) + self.sum(y1, x1)
        }
    }

    impl std::fmt::Debug for BinaryIndexedTree2 {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut buf = String::new();
            buf += "\n";
            for y in 0..self.h - 1 {
                for x in 0..self.w - 1 {
                    if x > 0 {
                        buf += " ";
                    }
                    buf += self.sum(y, x).to_string().as_str();
                }
                buf += "\n";
            }
            write!(f, "{}", buf)
        }
    }
}
