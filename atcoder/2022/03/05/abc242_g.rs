pub fn to_lr<R: RangeBounds<usize>>(range: &R, length: usize) -> (usize, usize) {
    use Bound::{Excluded, Included, Unbounded};
    let l = match range.start_bound() {
        Unbounded => 0,
        Included(&s) => s,
        Excluded(&s) => s + 1,
    };
    let r = match range.end_bound() {
        Unbounded => length,
        Included(&e) => e + 1,
        Excluded(&e) => e,
    };
    assert!(l <= r && r <= length);
    (l, r)
}
pub use std::{
    cmp::{max, min, Ordering, Reverse},
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
    convert::Infallible,
    convert::{TryFrom, TryInto},
    fmt::{Debug, Display, Formatter},
    io::{stdin, stdout, BufRead, BufWriter, Read, Write},
    iter::{Product, Sum},
    marker::PhantomData,
    mem::swap,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Bound,
        Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Range,
        RangeBounds, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
    str::{from_utf8, FromStr},
};
#[allow(unused_macros)]
macro_rules ! chmin {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_min = min ! ($ ($ cmps ) ,+ ) ; if $ base > cmp_min {$ base = cmp_min ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
#[allow(unused_macros)]
macro_rules ! min {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ b } else {$ a } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = min ! ($ ($ rest ) ,+ ) ; if $ a > b {b } else {$ a } } } ; }
pub struct Reader<F> {
    init: F,
    buf: VecDeque<String>,
}
impl<R: BufRead, F: FnMut() -> R> Iterator for Reader<F> {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            let reader = (self.init)();
            for l in reader.lines().flatten() {
                self.buf
                    .append(&mut l.split_whitespace().map(ToString::to_string).collect());
            }
        }
        self.buf.pop_front()
    }
}
impl<R: BufRead, F: FnMut() -> R> Reader<F> {
    pub fn new(init: F) -> Self {
        let buf = VecDeque::new();
        Reader { init, buf }
    }
    pub fn v<T: FromStr>(&mut self) -> T {
        let s = self.next().expect("Insufficient input.");
        s.parse().ok().expect("Failed to parse.")
    }
    pub fn v2<T1: FromStr, T2: FromStr>(&mut self) -> (T1, T2) {
        (self.v(), self.v())
    }
    pub fn v3<T1: FromStr, T2: FromStr, T3: FromStr>(&mut self) -> (T1, T2, T3) {
        (self.v(), self.v(), self.v())
    }
    pub fn v4<T1: FromStr, T2: FromStr, T3: FromStr, T4: FromStr>(&mut self) -> (T1, T2, T3, T4) {
        (self.v(), self.v(), self.v(), self.v())
    }
    pub fn v5<T1: FromStr, T2: FromStr, T3: FromStr, T4: FromStr, T5: FromStr>(
        &mut self,
    ) -> (T1, T2, T3, T4, T5) {
        (self.v(), self.v(), self.v(), self.v(), self.v())
    }
    pub fn vec<T: FromStr>(&mut self, length: usize) -> Vec<T> {
        (0..length).map(|_| self.v()).collect()
    }
    pub fn vec2<T1: FromStr, T2: FromStr>(&mut self, length: usize) -> Vec<(T1, T2)> {
        (0..length).map(|_| self.v2()).collect()
    }
    pub fn vec3<T1: FromStr, T2: FromStr, T3: FromStr>(
        &mut self,
        length: usize,
    ) -> Vec<(T1, T2, T3)> {
        (0..length).map(|_| self.v3()).collect()
    }
    pub fn vec4<T1: FromStr, T2: FromStr, T3: FromStr, T4: FromStr>(
        &mut self,
        length: usize,
    ) -> Vec<(T1, T2, T3, T4)> {
        (0..length).map(|_| self.v4()).collect()
    }
    pub fn chars(&mut self) -> Vec<char> {
        self.v::<String>().chars().collect()
    }
    pub fn digits(&mut self) -> Vec<i64> {
        self.v::<String>()
            .chars()
            .map(|c| (c as u8 - b'0') as i64)
            .collect()
    }
    pub fn char_map(&mut self, h: usize) -> Vec<Vec<char>> {
        (0..h).map(|_| self.chars()).collect()
    }
    pub fn bool_map(&mut self, h: usize, ng: char) -> Vec<Vec<bool>> {
        self.char_map(h)
            .iter()
            .map(|v| v.iter().map(|&c| c != ng).collect())
            .collect()
    }
    pub fn matrix<T: FromStr>(&mut self, h: usize, w: usize) -> Vec<Vec<T>> {
        (0..h).map(|_| self.vec(w)).collect()
    }
}
pub struct Writer<W: Write> {
    writer: BufWriter<W>,
}
impl<W: Write> Writer<W> {
    pub fn new(write: W) -> Self {
        Self {
            writer: BufWriter::new(write),
        }
    }
    pub fn ln<S: Display>(&mut self, s: S) {
        writeln!(self.writer, "{}", s).expect("Failed to write.")
    }
    pub fn out<S: Display>(&mut self, s: S) {
        write!(self.writer, "{}", s).expect("Failed to write.")
    }
    pub fn join<S: Display>(&mut self, v: &[S], separator: &str) {
        v.iter().fold("", |sep, arg| {
            write!(self.writer, "{}{}", sep, arg).expect("Failed to write.");
            separator
        });
        writeln!(self.writer).expect("Failed to write.");
    }
    pub fn flush(&mut self) {
        let _ = self.writer.flush();
    }
}
pub fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(|| stdin.lock()), Writer::new(stdout.lock()));
}
pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let n: usize = reader.v();
    let a = reader.vec::<usize>(n);
    let q: usize = reader.v();
    let lr = reader.vec2::<usize, usize>(q);

    let mut mo = Mo::<MoKind>::new(n, 1024, a);
    let mut len = vec![0; q];
    for i in 0..q {
        let (l, r) = lr[i];
        len[i] = r - l + 1;
        mo.add_query(l - 1, r);
    }
    mo.build();
    let ans = mo.run();
    for i in 0..q {
        let a = ans[i].unwrap();
        let ans = (len[i] - a as usize) / 2;
        writer.ln(ans);
    }
}

pub trait MoState {
    type Elem;
    type Answer;
    fn new() -> Self;
    fn get_ans(&self) -> Self::Answer;
    fn add_right(&mut self, e: &Self::Elem);
    fn add_left(&mut self, e: &Self::Elem);
    fn delete_right(&mut self, e: &Self::Elem);
    fn delete_left(&mut self, e: &Self::Elem);
}

pub struct Mo<M: MoState> {
    n: usize,
    q: usize,
    bsz: usize,
    qs: Vec<Vec<(usize, usize, usize)>>,
    v: Vec<M::Elem>,
}

impl<M: MoState> Mo<M> {
    pub fn new<I>(n: usize, bucket_sz: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = M::Elem>,
    {
        Self {
            n,
            q: 0,
            bsz: bucket_sz,
            qs: vec![Vec::new(); (n + bucket_sz - 1) / bucket_sz],
            v: iter.into_iter().collect(),
        }
    }

    pub fn add_query(&mut self, l: usize, r: usize) -> usize {
        let i = self.q;
        self.qs[l / self.bsz].push((r, l, i));
        self.q += 1;
        i
    }

    pub fn build(&mut self) {
        for s in 0..self.qs.len() {
            self.qs[s].sort();
        }
    }

    pub fn run(&self) -> Vec<Option<M::Answer>> {
        let mut ans: Vec<_> = (0..self.q).map(|_| None).collect();
        for s in 0..self.qs.len() {
            let mut state = M::new();
            let mut l = s * self.bsz;
            let mut r = l;
            for &(qr, ql, qi) in self.qs[s].iter() {
                while r < qr {
                    state.add_right(&self.v[r]);
                    r += 1;
                }
                while l > ql {
                    l -= 1;
                    state.add_left(&self.v[l]);
                }
                while r > qr {
                    r -= 1;
                    state.delete_right(&self.v[r]);
                }
                while l < ql {
                    state.delete_left(&self.v[l]);
                    l += 1;
                }
                ans[qi] = Some(state.get_ans());
            }
        }
        ans
    }
}

struct MoKind {
    cnt: Vec<i32>,
    ans: i32,
}

impl MoState for MoKind {
    type Elem = usize;
    type Answer = i32;
    fn new() -> Self {
        Self {
            cnt: vec![0; 505050],
            ans: 0,
        }
    }
    fn get_ans(&self) -> Self::Answer {
        self.ans
    }
    fn add_right(&mut self, e: &Self::Elem) {
        if self.cnt[*e] == 0 {
            self.cnt[*e] += 1;
            self.ans += 1;
        } else {
            self.cnt[*e] -= 1;
            self.ans -= 1;
        }
    }
    fn add_left(&mut self, e: &Self::Elem) {
        if self.cnt[*e] == 0 {
            self.cnt[*e] += 1;
            self.ans += 1;
        } else {
            self.cnt[*e] -= 1;
            self.ans -= 1;
        }
    }
    fn delete_right(&mut self, e: &Self::Elem) {
        if self.cnt[*e] == 0 {
            self.cnt[*e] += 1;
            self.ans += 1;
        } else {
            self.cnt[*e] -= 1;
            self.ans -= 1;
        }
    }
    fn delete_left(&mut self, e: &Self::Elem) {
        if self.cnt[*e] == 0 {
            self.cnt[*e] += 1;
            self.ans += 1;
        } else {
            self.cnt[*e] -= 1;
            self.ans -= 1;
        }
    }
}
