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
    let (n, w) = reader.u2();

    let vw = reader.uv2(n);
    let mut v_max = 0;
    let mut w_max = 0;
    for i in 0..n {
        chmax!(v_max, vw[i].0);
        chmax!(w_max, vw[i].1);
    }
    if n <= 30 {
        let l = min(15, n);
        let mut res1 = BTreeMap::new();
        for p in 0..1 << l {
            let mut v = 0;
            let mut w = 0;
            for i in 0..l {
                if ((p >> i) & 1) != 0 {
                    v += vw[i].0;
                    w += vw[i].1;
                }
            }
            chmax!(*res1.entry(w).or_insert(0), v);
        }
        let mut m = 0;
        let res1: BTreeMap<_, _> = res1
            .into_iter()
            .map(|(w, v)| {
                chmax!(m, v);
                (w, max(v, m))
            })
            .collect();

        if n > 15 {
            let l = n - 15;
            let mut res2 = Vec::new();
            for p in 0..1 << l {
                let mut v_sum = 0;
                let mut w_sum = 0;
                for i in 0..l {
                    if ((p >> i) & 1) != 0 {
                        v_sum += vw[i + 15].0;
                        w_sum += vw[i + 15].1;
                    }
                }
                res2.push((v_sum, w_sum));
            }
            let mut ans = 0;
            for &(v_sum, w_sum) in &res2 {
                if w_sum > w {
                    continue;
                }
                let t = res1.range(..=(w - w_sum)).last().unwrap();
                let _b = chmax!(ans, v_sum + t.1);
            }
            println!("{}", ans);
        } else {
            let mut ans = 0;
            for (w_sum, v_sum) in res1 {
                if w_sum <= w {
                    chmax!(ans, v_sum);
                }
            }
            println!("{}", ans);
        }
    } else if v_max <= 1000 {
        const INF: usize = 1 << 60;
        let mut dp = vec![INF; 200010];
        dp[0] = 0;
        for (v, w) in vw {
            for src in (0..200001 - v).rev() {
                let dst = src + v;
                chmin!(dp[dst], dp[src] + w);
            }
        }
        let mut ans = 0;
        for i in 0..=200000 {
            if dp[i] <= w {
                chmax!(ans, i);
            }
        }
        println!("{}", ans);
    } else if w_max <= 1000 {
        let mut dp = vec![0; 200010];
        for (v, w) in vw {
            for src in (0..200001 - w).rev() {
                let dst = src + w;
                chmax!(dp[dst], dp[src] + v);
            }
        }
        let mut ans = 0;
        for i in 0..=w {
            chmax!(ans, dp[i]);
        }
        println!("{}", ans);
    }
}
