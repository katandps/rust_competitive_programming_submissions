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
    mem::swap,
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

/// マグマ
/// 二項演算: $`M \circ M \to M`$
pub trait Magma {
    /// マグマを構成する集合$`M`$
    type M: Clone + Debug + PartialEq;
    /// マグマを構成する演算$`op`$
    fn op(x: &Self::M, y: &Self::M) -> Self::M;
}

/// 結合則
/// $`\forall a,\forall b, \forall c \in T, (a \circ b) \circ c = a \circ (b \circ c)`$
pub trait Associative {}

/// 半群
pub trait SemiGroup {}
impl<M: Magma + Associative> SemiGroup for M {}

/// 単位的
pub trait Unital: Magma {
    /// 単位元 identity element: $`e`$
    fn unit() -> Self::M;
}

/// モノイド
/// 結合則と、単位元を持つ
pub trait Monoid {
    type M: Clone + Debug + PartialEq;
    fn op(x: &Self::M, y: &Self::M) -> Self::M;

    fn unit() -> Self::M;

    /// $`x^n = x\circ\cdots\circ x`$
    fn pow(&self, x: Self::M, n: usize) -> Self::M;
}

impl<M: SemiGroup + Unital> Monoid for M {
    type M = M::M;
    fn op(x: &M::M, y: &M::M) -> M::M {
        M::op(x, y)
    }

    fn unit() -> Self::M {
        M::unit()
    }

    /// $`x^n = x\circ\cdots\circ x`$
    fn pow(&self, x: Self::M, mut n: usize) -> Self::M {
        let mut res = Self::unit();
        let mut base = x;
        while n > 0 {
            if n & 1 == 1 {
                res = Self::op(&res, &base);
            }
            base = Self::op(&base, &base);
            n >>= 1;
        }
        res
    }
}

/// 可逆的
/// $`\exists e \in T, \forall a \in T, \exists b,c \in T, b \circ a = a \circ c = e`$
pub trait Invertible: Magma {
    /// $`a`$ where $`a \circ x = e`$
    fn inv(&self, x: &Self::M) -> Self::M;
}

/// 群
pub trait Group {}
impl<M: Monoid + Invertible> Group for M {}

/// 作用付きモノイド
/// 値Mono、作用Funcはモノイドで、
pub trait MapMonoid: Debug {
    /// モノイドM
    type Mono: Monoid;
    type Func: Monoid;
    /// 値xと値yを併合する
    fn op(
        x: &<Self::Mono as Monoid>::M,
        y: &<Self::Mono as Monoid>::M,
    ) -> <Self::Mono as Monoid>::M {
        Self::Mono::op(&x, &y)
    }
    fn unit() -> <Self::Mono as Monoid>::M {
        Self::Mono::unit()
    }
    /// 作用fをvalueに作用させる
    fn apply(
        f: &<Self::Func as Monoid>::M,
        value: &<Self::Mono as Monoid>::M,
    ) -> <Self::Mono as Monoid>::M;
    /// 作用fの単位元
    fn identity_map() -> <Self::Func as Monoid>::M {
        Self::Func::unit()
    }
    /// 作用fと作用gを合成する
    fn compose(
        f: &<Self::Func as Monoid>::M,
        g: &<Self::Func as Monoid>::M,
    ) -> <Self::Func as Monoid>::M {
        Self::Func::op(f, g)
    }
}

/// 加算の単位元
pub trait Zero {
    fn zero() -> Self;
}

/// 乗算の単位元
pub trait One {
    fn one() -> Self;
}

/// 下に有界
pub trait BoundedBelow {
    fn min_value() -> Self;
}

/// 上に有界
pub trait BoundedAbove {
    fn max_value() -> Self;
}

/// 整数
pub trait Integral:
'static
+ Send
+ Sync
+ Copy
+ Ord
+ Not<Output = Self>
+ Add<Output = Self>
+ Sub<Output = Self>
+ Mul<Output = Self>
+ Div<Output = Self>
+ Rem<Output = Self>
+ AddAssign
+ SubAssign
+ MulAssign
+ DivAssign
+ RemAssign
+ Sum
+ Product
+ BitOr<Output = Self>
+ BitAnd<Output = Self>
+ BitXor<Output = Self>
+ BitOrAssign
+ BitAndAssign
+ BitXorAssign
+ Shl<Output = Self>
+ Shr<Output = Self>
+ ShlAssign
+ ShrAssign
+ Display
+ Debug
+ Zero
+ One
+ BoundedBelow
+ BoundedAbove
{
}

macro_rules! impl_integral {
    ($($ty:ty),*) => {
        $(
            impl Zero for $ty {
                fn zero() -> Self {
                    0
                }
            }

            impl One for $ty {
                fn one() -> Self {
                    1
                }
            }

            impl BoundedBelow for $ty {
                fn min_value() -> Self {
                    Self::min_value()
                }
            }

            impl BoundedAbove for $ty {
                fn max_value() -> Self {
                    Self::max_value()
                }
            }

            impl Integral for $ty {}
        )*
    };
}
impl_integral!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

#[derive(Copy, Clone, Eq, Ord, Default)]
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

pub trait GraphTrait {
    type Weight;
    fn size(&self) -> usize;
    fn edges(&self, src: usize) -> Vec<Edge<Self::Weight>>;
}

impl<W: Clone> GraphTrait for Graph<W> {
    type Weight = W;
    fn size(&self) -> usize {
        self.n
    }
    fn edges(&self, src: usize) -> Vec<Edge<W>> {
        self.edges[src].clone()
    }
}

/// 辺の情報を使用してグラフの問題を解くためのライブラリ
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

/// i64の辺行列からグラフを生成する O(N^2)
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
                ret.add_edge(i, j, w[i as usize][j as usize].clone());
                ret.add_edge(j, i, w[j as usize][i as usize].clone());
            }
        }
        ret
    }
}

impl<W> Graph<W> {
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

impl<W: Clone> Graph<W> {
    /// n: 頂点数
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: vec![Vec::new(); n],
            rev_edges: vec![Vec::new(); n],
        }
    }
    /// 相互に行き来できる辺をつける
    pub fn add_edge(&mut self, a: usize, b: usize, w: W) {
        self.edges[a].push(Edge::new(a, b, w.clone()));
        self.edges[b].push(Edge::new(b, a, w.clone()));
        self.rev_edges[a].push(Edge::new(a, b, w.clone()));
        self.rev_edges[b].push(Edge::new(b, a, w));
    }

    /// 1方向にのみ移動できる辺をつける
    pub fn add_arc(&mut self, a: usize, b: usize, w: W) {
        self.edges[a].push(Edge::new(a, b, w.clone()));
        self.rev_edges[b].push(Edge::new(b, a, w));
    }
}

pub struct Kruskal<W> {
    tree: Vec<Edge<W>>,
    sum: W,
}

impl<W: Zero + PartialOrd + Copy + Add<Output = W> + AddAssign, Graph: GraphTrait<Weight = W>>
From<&Graph> for Kruskal<W>
{
    fn from(graph: &Graph) -> Self {
        let mut edges = Vec::new();
        for i in 0..graph.size() {
            for e in graph.edges(i) {
                edges.push(e);
            }
        }
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut tree = Vec::new();
        let mut sum = W::zero();
        let mut uf = UnionFind::new(graph.size());
        for edge in edges {
            if uf.root(edge.src) != uf.root(edge.dst) {
                uf.unite(edge.src, edge.dst);
                sum += edge.weight;
                tree.push(edge);
            }
        }
        Self { tree, sum }
    }
}

impl<W: Copy> Kruskal<W> {
    pub fn tree(&self) -> &Vec<Edge<W>> {
        &self.tree
    }

    pub fn sum(&self) -> W {
        self.sum
    }
}

#[allow(dead_code)]
fn sqrt(a: i128) -> i128 {
    let mut ng = std::i64::MAX as i128 / 2;
    let mut ok = -1;
    while (ok - ng).abs() > 1 {
        let mid = (ok + ng) / 2;
        if mid * mid <= a {
            ok = mid;
        } else {
            ng = mid;
        }
    }
    ok
}
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> UnionFind {
        let parent = (0..n + 1).collect::<Vec<_>>();
        let rank = vec![0; n + 1];
        let size = vec![1; n + 1];
        UnionFind { parent, rank, size }
    }

    pub fn root(&mut self, x: usize) -> usize {
        if self.parent[x] == x {
            x
        } else {
            let p = self.parent[x];
            self.parent[x] = self.root(p);
            self.parent[x]
        }
    }

    pub fn rank(&self, x: usize) -> usize {
        self.rank[x]
    }

    pub fn size(&mut self, x: usize) -> usize {
        let root = self.root(x);
        self.size[root]
    }

    pub fn same(&mut self, x: usize, y: usize) -> bool {
        self.root(x) == self.root(y)
    }

    pub fn unite(&mut self, x: usize, y: usize) {
        let mut x = self.root(x);
        let mut y = self.root(y);
        if x == y {
            return;
        }
        if self.rank(x) < self.rank(y) {
            std::mem::swap(&mut x, &mut y);
        }
        if self.rank(x) == self.rank(y) {
            self.rank[x] += 1;
        }
        self.parent[x] = y;
        self.size[y] += self.size[x];
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
    let xyc = reader.iv3(n);
    let xyc2 = reader.iv3(m);
    let mut g = Graph::new(n + m);
    for i in 0..n {
        for j in i + 1..n {
            let (x1, y1, c1) = xyc[i];
            let (x2, y2, c2) = xyc[j];
            let d = (((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) as f64).sqrt();
            if c1 == c2 {
                g.add_edge(i, j, d);
            } else {
                g.add_edge(i, j, d * 10.0);
            }
        }
    }

    let mut ans: f64 = 1e18;
    for p in 0..1 << m {
        let mut g = g.clone();
        let v: Vec<_> = (0..m)
            .filter_map(|i| {
                if ((p >> i) & 1) == 1 {
                    Some(xyc2[i])
                } else {
                    None
                }
            })
            .collect();
        for i in 0..n {
            for j in 0..v.len() {
                let (x1, y1, c1) = xyc[i];
                let (x2, y2, c2) = v[j];
                let d = (((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) as f64).sqrt();
                if c1 == c2 {
                    g.add_edge(i, j + n, d);
                } else {
                    g.add_edge(i, j + n, d * 10.0);
                }
            }
        }
        for i in 0..v.len() {
            for j in i + 1..v.len() {
                let (x1, y1, c1) = v[i];
                let (x2, y2, c2) = v[j];
                let d = (((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) as f64).sqrt();

                if c1 == c2 {
                    g.add_edge(i + n, j + n, d);
                } else {
                    g.add_edge(i + n, j + n, d * 10.0);
                }
            }
        }
        let kruskal = Kruskal::from(&g);
        chmin!(ans, kruskal.sum);
    }
    writer.println(&(ans));
}

impl Zero for f64 {
    fn zero() -> f64 {
        0.0
    }
}
