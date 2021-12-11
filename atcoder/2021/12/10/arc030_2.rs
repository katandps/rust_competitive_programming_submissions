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
pub struct Reader<R> {
    reader: R,
    buf: VecDeque<String>,
}
impl<R: Read> Iterator for Reader<R> {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            let mut buf = Vec::new();
            self.reader.read_to_end(&mut buf).unwrap();
            let s = from_utf8(&buf).expect("Not UTF-8 format input.");
            self.buf = s.split_whitespace().map(ToString::to_string).collect();
        }
        self.buf.pop_front()
    }
}
impl<R: Read> Reader<R> {
    pub fn new(reader: R) -> Reader<R> {
        let buf = VecDeque::new();
        Reader { reader, buf }
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
}
pub fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
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
impl<W: Clone> GraphTrait for Graph<W> {
    type Weight = W;
    fn size(&self) -> usize {
        self.n
    }
    fn edges(&self, src: usize) -> Vec<Edge<W>> {
        self.edges[src].clone()
    }
    fn rev_edges(&self, src: usize) -> Vec<Edge<W>> {
        self.rev_edges[src].clone()
    }
}
pub struct Graph<W> {
    pub n: usize,
    pub edges: Vec<Vec<Edge<W>>>,
    pub rev_edges: Vec<Vec<Edge<W>>>,
}
impl<W: Clone> Clone for Graph<W> {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            edges: self.edges.clone(),
            rev_edges: self.rev_edges.clone(),
        }
    }
}
impl From<&Vec<Vec<i64>>> for Graph<i64> {
    fn from(w: &Vec<Vec<i64>>) -> Self {
        let n = w.len();
        let mut ret = Self::new(n);
        for i in 0..n {
            assert_eq!(n, w[i].len());
            for j in i + 1..n {
                if w[i][j] == -1 {
                    continue;
                }
                ret.add_edge(i, j, w[i as usize][j as usize]);
                ret.add_edge(j, i, w[j as usize][i as usize]);
            }
        }
        ret
    }
}
impl<W: Clone> Graph<W> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: vec![Vec::new(); n],
            rev_edges: vec![Vec::new(); n],
        }
    }
    pub fn add_edge(&mut self, a: usize, b: usize, w: W) {
        self.edges[a].push(Edge::new(a, b, w.clone()));
        self.edges[b].push(Edge::new(b, a, w.clone()));
        self.rev_edges[a].push(Edge::new(a, b, w.clone()));
        self.rev_edges[b].push(Edge::new(b, a, w));
    }
    pub fn add_arc(&mut self, a: usize, b: usize, w: W) {
        self.edges[a].push(Edge::new(a, b, w.clone()));
        self.rev_edges[b].push(Edge::new(b, a, w));
    }
}
pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let n: usize = reader.v();
    let x: usize = reader.v();
    let h = reader.vec::<usize>(n);
    let ab = reader.vec2::<usize, usize>(n - 1);

    let mut graph = Graph::new(n);
    for (a, b) in ab {
        graph.add_edge(a - 1, b - 1, 1);
    }

    writer.ln(dfs(x - 1, x - 1, &graph, &h).0);
}

fn dfs(cur: usize, par: usize, graph: &Graph<usize>, h: &Vec<usize>) -> (usize, bool) {
    let mut ret = 0;
    let mut gem = h[cur] > 0;
    for e in graph.edges(cur) {
        if e.dst == par {
            continue;
        }
        let (w, g) = dfs(e.dst, e.src, graph, h);
        if g {
            ret += w + 2;
            gem |= g;
        }
    }
    (ret, gem)
}
