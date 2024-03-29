pub use reader::*;
#[allow(unused_imports)]
use {
    itertools::Itertools,
    num::Integer,
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
pub mod reader { #[allow(unused_imports)]use itertools::Itertools;use std::{fmt::Debug, io::*, str::*};pub struct Reader<R: BufRead> {reader: R,buf: Vec<u8>,pos: usize,}  macro_rules! prim_method { ($name:ident: $T: ty) => { pub fn $name(&mut self) -> $T { self.n::<$T>() } }; ($name:ident) => { prim_method!($name: $name); }; } macro_rules! prim_methods { ($name:ident: $T:ty; $($rest:tt)*) => { prim_method!($name:$T); prim_methods!($($rest)*); }; ($name:ident; $($rest:tt)*) => { prim_method!($name); prim_methods!($($rest)*); }; () => () }  macro_rules! replace_expr { ($_t:tt $sub:expr) => { $sub }; } macro_rules! tuple_method { ($name: ident: ($($T:ident),+)) => { pub fn $name(&mut self) -> ($($T),+) { ($(replace_expr!($T self.n())),+) } } } macro_rules! tuple_methods { ($name:ident: ($($T:ident),+); $($rest:tt)*) => { tuple_method!($name:($($T),+)); tuple_methods!($($rest)*); }; () => () } macro_rules! vec_method { ($name: ident: ($($T:ty),+)) => { pub fn $name(&mut self, n: usize) -> Vec<($($T),+)> { (0..n).map(|_|($(replace_expr!($T self.n())),+)).collect_vec() } }; ($name: ident: $T:ty) => { pub fn $name(&mut self, n: usize) -> Vec<$T> { (0..n).map(|_|self.n()).collect_vec() } }; } macro_rules! vec_methods { ($name:ident: ($($T:ty),+); $($rest:tt)*) => { vec_method!($name:($($T),+)); vec_methods!($($rest)*); }; ($name:ident: $T:ty; $($rest:tt)*) => { vec_method!($name:$T); vec_methods!($($rest)*); }; () => () } impl<R: BufRead> Reader<R> {pub fn new(reader: R) -> Reader<R> {let (buf, pos) = (Vec::new(), 0);Reader { reader, buf, pos }} prim_methods! { u: usize; i: i64; f: f64; str: String; c: char; string: String; u8; u16; u32; u64; u128; usize; i8; i16; i32; i64; i128; isize; f32; f64; char; } tuple_methods! { u2: (usize, usize); u3: (usize, usize, usize); u4: (usize, usize, usize, usize); i2: (i64, i64); i3: (i64, i64, i64); i4: (i64, i64, i64, i64); cuu: (char, usize, usize); } vec_methods! { uv: usize; uv2: (usize, usize); uv3: (usize, usize, usize); iv: i64; iv2: (i64, i64); iv3: (i64, i64, i64); vq: (char, usize, usize); }  pub fn n<T: FromStr>(&mut self) -> T where T::Err: Debug, { self.n_op().unwrap() }pub fn n_op<T: FromStr>(&mut self) -> Option<T> where T::Err: Debug, {if self.buf.is_empty() { self._read_next_line(); }let mut start = None;while self.pos != self.buf.len() {match (self.buf[self.pos], start.is_some()) {(b' ', true) | (b'\n', true) => break,(_, true) | (b' ', false) => self.pos += 1,(b'\n', false) => self._read_next_line(),(_, false) => start = Some(self.pos),}}start.map(|s| from_utf8(&self.buf[s..self.pos]).unwrap().parse().unwrap())}fn _read_next_line(&mut self) {self.pos = 0;self.buf.clear();self.reader.read_until(b'\n', &mut self.buf).unwrap();}pub fn s(&mut self) -> Vec<char> { self.n::<String>().chars().collect() }pub fn char_map(&mut self, h: usize) -> Vec<Vec<char>> { (0..h).map(|_| self.s()).collect() }pub fn bool_map(&mut self, h: usize, ng: char) -> Vec<Vec<bool>> { self.char_map(h).iter().map(|v| v.iter().map(|&c| c != ng).collect()).collect() }pub fn matrix(&mut self, h: usize, w: usize) -> Vec<Vec<i64>> { (0..h).map(|_| self.iv(w)).collect() }}}

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    solve(Reader::new(stdin.lock()));
}

#[allow(unused_imports)]
use greatest_common_divisor::*;

#[allow(dead_code)]
mod greatest_common_divisor {
    use std::mem::swap;

    pub fn gcd(mut a: usize, mut b: usize) -> usize {
        if a < b {
            swap(&mut b, &mut a);
        }
        while b != 0 {
            a = a % b;
            swap(&mut a, &mut b);
        }
        a
    }
}

pub fn solve<R: BufRead>(mut reader: Reader<R>) {
    let n = reader.u();
    let mut ab = reader.uv2(n);

    let mut all_gcd = gcd(ab[0].0, ab[0].1);
    for i in 1..n {
        all_gcd = gcd(all_gcd, ab[i].0);
        all_gcd = gcd(all_gcd, ab[i].1);
    }

    for i in 0..n {
        ab[i] = (ab[i].0 / all_gcd, ab[i].1 / all_gcd);
    }

    let mut set = BTreeSet::new();
    for i in 0..n {
        set_c(ab[i].0, &mut set);
        set_c(ab[i].1, &mut set);
    }

    let mut ans = 1usize;
    'case: for &p in &set {
        let mut forced = Vec::new();
        let mut can_select = Vec::new();
        for &(a, b) in &ab {
            if a % p != 0 && b % p != 0 {
                continue 'case;
            }
            if a % p != 0 {
                forced.push(a);
            } else if b % p != 0 {
                forced.push(b);
            } else {
                can_select.push((a / p, b / p));
            }
        }
        if !forced.is_empty() {
            let mut forced_gcd = forced[0];
            for i in 1..forced.len() {
                forced_gcd = gcd(forced_gcd, forced[i]);
            }
            can_select.push((forced_gcd, forced_gcd));
        }

        let mut max2 = 1;
        'case2: for &p2 in &set {
            for &(a, b) in &can_select {
                if a % p2 != 0 && b % p2 != 0 {
                    continue 'case2;
                }
            }
            max2 = max(max2, p2);
        }
        ans = max(ans, p * max2);
    }
    println!("{}", ans * all_gcd);
}

fn set_c(p: usize, set: &mut BTreeSet<usize>) {
    let mut l = 1;
    while l * l <= p {
        if p % l == 0 {
            set.insert(l);
            set.insert(p / l);
        }
        l += 1;
    }
}
