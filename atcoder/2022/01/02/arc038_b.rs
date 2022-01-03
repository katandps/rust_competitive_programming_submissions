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

#[derive(Copy, Clone, Eq, Default)]
pub struct Edge<W> {
    pub src: usize,
    pub dst: usize,
    pub weight: W,
}
impl<W: Display> Debug for Edge<W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {} : {}", self.src, self.dst, self.weight)
    }
}
impl<W> Edge<W> {
    pub fn new(src: usize, dst: usize, weight: W) -> Self {
        Edge { src, dst, weight }
    }
}
impl<W: PartialEq> PartialEq for Edge<W> {
    fn eq(&self, other: &Self) -> bool {
        self.weight.eq(&other.weight)
    }
}
impl<W: PartialOrd> PartialOrd for Edge<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.weight.partial_cmp(&other.weight)
    }
}
impl<W: PartialOrd + Eq> Ord for Edge<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight.partial_cmp(&other.weight).expect("Found NAN")
    }
}
pub trait GraphTrait {
    type Weight;
    fn size(&self) -> usize;
    fn edges(&self, src: usize) -> Vec<Edge<Self::Weight>>;
    fn rev_edges(&self, dst: usize) -> Vec<Edge<Self::Weight>>;
    fn indegree(&self) -> Vec<i32> {
        (0..self.size())
            .map(|dst| self.rev_edges(dst).len() as i32)
            .collect()
    }
    fn outdegree(&self) -> Vec<i32> {
        (0..self.size())
            .map(|src| self.edges(src).len() as i32)
            .collect()
    }
}

#[derive(Debug)]
pub struct Grid<W> {
    pub h: usize,
    pub w: usize,
    pub size: usize,
    pub map: Vec<W>,
}
impl<W: Clone> GraphTrait for Grid<W> {
    type Weight = W;
    fn size(&self) -> usize {
        self.size
    }
    fn edges(&self, src: usize) -> Vec<Edge<W>> {
        let mut ret = Vec::new();
        if self.x(src) + 1 < self.w {
            ret.push(Edge::new(src, src + 1, self.map[src + 1].clone()));
        }
        if self.y(src) + 1 < self.h {
            ret.push(Edge::new(src, src + self.w, self.map[src + self.w].clone()));
        }
        if self.x(src) + 1 < self.w && self.y(src) + 1 < self.h {
            ret.push(Edge::new(
                src,
                src + self.w + 1,
                self.map[src + self.w + 1].clone(),
            ))
        }
        ret
    }
    fn rev_edges(&self, dst: usize) -> Vec<Edge<W>> {
        let mut ret = Vec::new();

        if self.x(dst) > 0 {
            ret.push(Edge::new(dst, dst - 1, self.map[dst].clone()));
        }
        if self.y(dst) > 0 {
            ret.push(Edge::new(dst, dst - self.w, self.map[dst].clone()));
        }
        ret
    }
}
impl<W: Clone> Grid<W> {
    pub fn new(h: usize, w: usize, input: Vec<Vec<W>>) -> Grid<W> {
        let mut map = Vec::new();
        for r in input {
            for c in r {
                map.push(c);
            }
        }
        let max = h * w;
        Grid {
            h,
            w,
            size: max,
            map,
        }
    }
    pub fn key(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }
    pub fn xy(&self, k: usize) -> (usize, usize) {
        (self.x(k), self.y(k))
    }
    pub fn x(&self, k: usize) -> usize {
        k % self.w
    }
    pub fn y(&self, k: usize) -> usize {
        k / self.w
    }
    pub fn get(&self, key: usize) -> &W {
        &self.map[key]
    }
    pub fn set(&mut self, key: usize, value: W) {
        self.map[key] = value;
    }
}

pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let (h, w): (usize, usize) = reader.v2();
    let s = reader.bool_map(h, '#');
    let grid = Grid::new(h, w, s);

    let mut winner = vec![None; h * w];
    writer.ln(if dfs(0, &grid, &mut winner) {
        "First"
    } else {
        "Second"
    });
}

fn dfs(cur: usize, grid: &Grid<bool>, winner: &mut Vec<Option<bool>>) -> bool {
    if let Some(b) = winner[cur] {
        return b;
    }

    let mut win = false;
    for edge in grid.edges(cur) {
        if !edge.weight {
            continue;
        }
        if !dfs(edge.dst, grid, winner) {
            win = true;
        }
    }
    winner[cur] = Some(win);
    return win;
}
