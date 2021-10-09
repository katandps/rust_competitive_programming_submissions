/// general import
pub use std::{
    cmp::{max, min, Ordering, Reverse},
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
    convert::Infallible,
    convert::{TryFrom, TryInto},
    fmt::{Debug, Display, Formatter},
    io::{stdin, stdout, BufRead, BufWriter, Write},
    iter::{Product, Sum},
    marker::PhantomData,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Bound,
        Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Not, RangeBounds, Rem, RemAssign,
        Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
    str::{from_utf8, FromStr},
};

/// min-max macros
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

/// stdin reader
pub struct Reader<R: BufRead> {
    reader: R,
    buf: Vec<u8>,
    pos: usize,
}

macro_rules! prim_method {
    ($name:ident: $T: ty) => {
        pub fn $name(&mut self) -> $T {
            self.n::<$T>()
        }
    };
    ($name:ident) => {
        prim_method!($name: $name);
    }
}
macro_rules! prim_methods {
    ($name:ident: $T:ty; $($rest:tt)*) => {
        prim_method!($name:$T);
        prim_methods!($($rest)*);
    };
    ($name:ident; $($rest:tt)*) => {
        prim_method!($name);
        prim_methods!($($rest)*);
    };
    () => ()
}

macro_rules! replace_expr {
    ($_t:tt $sub:expr) => {
        $sub
    };
}
macro_rules! tuple_method {
    ($name: ident: ($($T:ident),+)) => {
        pub fn $name(&mut self) -> ($($T),+) {
            ($(replace_expr!($T self.n())),+)
        }
    }
}
macro_rules! tuple_methods {
    ($name:ident: ($($T:ident),+); $($rest:tt)*) => {
        tuple_method!($name:($($T),+));
        tuple_methods!($($rest)*);
    };
    () => ()
}
macro_rules! vec_method {
    ($name: ident: ($($T:ty),+)) => {
        pub fn $name(&mut self, n: usize) -> Vec<($($T),+)> {
            (0..n).map(|_|($(replace_expr!($T self.n())),+)).collect()
        }
    };
    ($name: ident: $T:ty) => {
        pub fn $name(&mut self, n: usize) -> Vec<$T> {
            (0..n).map(|_|self.n()).collect()
        }
    };
}
macro_rules! vec_methods {
    ($name:ident: ($($T:ty),+); $($rest:tt)*) => {
        vec_method!($name:($($T),+));
        vec_methods!($($rest)*);
    };
    ($name:ident: $T:ty; $($rest:tt)*) => {
        vec_method!($name:$T);
        vec_methods!($($rest)*);
    };
    () => ()
}
impl<R: BufRead> Reader<R> {
    pub fn new(reader: R) -> Reader<R> {
        let (buf, pos) = (Vec::new(), 0);
        Reader { reader, buf, pos }
    }
    prim_methods! {
        u: usize; i: i64; f: f64; str: String; c: char; string: String;
        u8; u16; u32; u64; u128; usize; i8; i16; i32; i64; i128; isize; f32; f64; char;
    }
    tuple_methods! {
        u2: (usize, usize); u3: (usize, usize, usize); u4: (usize, usize, usize, usize);
        i2: (i64, i64); i3: (i64, i64, i64); i4: (i64, i64, i64, i64);
        cuu: (char, usize, usize);
    }
    vec_methods! {
        uv: usize; uv2: (usize, usize); uv3: (usize, usize, usize);
        iv: i64; iv2: (i64, i64); iv3: (i64, i64, i64);
        vq: (char, usize, usize);
    }

    pub fn n<T: FromStr>(&mut self) -> T
        where
            T::Err: Debug,
    {
        self.n_op().unwrap()
    }

    pub fn n_op<T: FromStr>(&mut self) -> Option<T>
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
        start.map(|s| from_utf8(&self.buf[s..self.pos]).unwrap().parse().unwrap())
    }

    fn _read_next_line(&mut self) {
        self.pos = 0;
        self.buf.clear();
        self.reader.read_until(b'\n', &mut self.buf).unwrap();
    }
    pub fn s(&mut self) -> Vec<char> {
        self.n::<String>().chars().collect()
    }
    pub fn digits(&mut self) -> Vec<i64> {
        self.n::<String>()
            .chars()
            .map(|c| (c as u8 - b'0') as i64)
            .collect()
    }
    pub fn char_map(&mut self, h: usize) -> Vec<Vec<char>> {
        (0..h).map(|_| self.s()).collect()
    }
    /// charの2次元配列からboolのmapを作る ngで指定した壁のみfalseとなる
    pub fn bool_map(&mut self, h: usize, ng: char) -> Vec<Vec<bool>> {
        self.char_map(h)
            .iter()
            .map(|v| v.iter().map(|&c| c != ng).collect())
            .collect()
    }
    /// h*w行列を取得する
    pub fn matrix(&mut self, h: usize, w: usize) -> Vec<Vec<i64>> {
        (0..h).map(|_| self.iv(w)).collect()
    }
}

/// stdin writer
pub struct Writer<W: Write> {
    w: BufWriter<W>,
}
impl<W: Write> Writer<W> {
    pub fn new(writer: W) -> Writer<W> {
        Writer {
            w: BufWriter::new(writer),
        }
    }

    pub fn println<S: Display>(&mut self, s: &S) {
        writeln!(self.w, "{}", s).unwrap()
    }

    pub fn print<S: Display>(&mut self, s: &S) {
        write!(self.w, "{}", s).unwrap()
    }

    pub fn print_join<S: Display>(&mut self, v: &[S], separator: Option<&str>) {
        let sep = separator.unwrap_or_else(|| "\n");
        writeln!(
            self.w,
            "{}",
            v.iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .join(sep)
        )
            .unwrap()
    }
}

pub type Weight = i64;

pub const INF: Weight = 1 << 60;

#[derive(Copy, Clone)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub weight: Weight,
}

impl Debug for Edge {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {} : {}", self.src, self.dst, self.weight)
    }
}

impl Edge {
    pub fn default() -> Edge {
        let (src, dst, weight) = (0, 0, 0);
        Edge { src, dst, weight }
    }

    pub fn new(src: usize, dst: usize, weight: Weight) -> Edge {
        Edge { src, dst, weight }
    }
}

impl std::cmp::PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.weight.eq(&other.weight)
    }
}

impl std::cmp::Eq for Edge {}

impl std::cmp::PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.weight.partial_cmp(&other.weight)
    }
}

impl std::cmp::Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight.cmp(&other.weight)
    }
}

/// 辺の情報を使用してグラフの問題を解くためのライブラリ
#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<Vec<Edge>>,
    pub rev_edges: Vec<Vec<Edge>>,
}

/// 辺行列からグラフを生成する O(N^2)
impl From<&Vec<Vec<Weight>>> for Graph {
    fn from(w: &Vec<Vec<Weight>>) -> Self {
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

impl Graph {
    /// n: 頂点数
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: vec![Vec::new(); n],
            rev_edges: vec![Vec::new(); n],
        }
    }

    /// 相互に行き来できる辺をつける
    pub fn add_edge(&mut self, a: usize, b: usize, w: Weight) {
        self.edges[a].push(Edge::new(a, b, w));
        self.edges[b].push(Edge::new(b, a, w));
        self.rev_edges[a].push(Edge::new(a, b, w));
        self.rev_edges[b].push(Edge::new(b, a, w));
    }

    /// 1方向にのみ移動できる辺をつける
    pub fn add_arc(&mut self, a: usize, b: usize, w: Weight) {
        self.edges[a].push(Edge::new(a, b, w));
        self.rev_edges[b].push(Edge::new(b, a, w));
    }

    /// 各頂点の入次数を返す
    pub fn indegree(&self) -> Vec<i32> {
        (0..self.n)
            .map(|dst| self.rev_edges[dst].len() as i32)
            .collect()
    }

    /// 各頂点の出次数を返す
    pub fn outdegree(&self) -> Vec<i32> {
        (0..self.n)
            .map(|src| self.edges[src].len() as i32)
            .collect()
    }
}

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let (n, m) = reader.u2();
    let mut g = vec![HashMap::new(); n + 1];
    let mut v = Vec::new();
    for _ in 0..m {
        let (a, b) = reader.u2();
        let (a, b) = (a - 1, b - 1);
        let c = reader.c();
        g[a].entry(c).or_insert(Vec::new()).push(b);
        g[b].entry(c).or_insert(Vec::new()).push(a);
        v.push((a, b));
    }

    let mut g2 = Graph::new(n * n);
    for src_a in 0..n {
        for src_b in 0..n {
            for (c, from) in &g[src_a] {
                if let Some(to) = g[src_b].get(c) {
                    for dst_a in from {
                        for dst_b in to {
                            g2.add_edge(key(src_a, src_b, n), key(*dst_a, *dst_b, n), 1);
                        }
                    }
                }
            }
        }
    }
    let mut q = VecDeque::new();
    q.push_front(n - 1);
    let mut dist = vec![1i64 << 60; n * n];
    dist[n - 1] = 0;
    while let Some(src) = q.pop_front() {
        for e in &g2.edges[src] {
            if chmin!(dist[e.dst], dist[src] + e.weight) {
                q.push_back(e.dst);
            }
        }
    }
    let mut ans = 1 << 60;
    for i in 0..n {
        chmin!(ans, dist[key(i, i, n)] * 2);
    }
    for (a, b) in v {
        chmin!(ans, dist[key(a, b, n)] + dist[key(b, a, n)]);
    }
    if ans == 1 << 60 {
        writer.println(&-1)
    } else {
        writer.println(&ans);
    }
}

fn key(a: usize, b: usize, n: usize) -> usize {
    a * n + b
}
