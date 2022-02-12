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
    let ab = reader.vec2::<usize, usize>(n - 1);
    let q: usize = reader.v();
    let mut graph = Graph::new(n);
    for (a, b) in ab {
        graph.add_edge(a - 1, b - 1, ());
    }
    let lcm = LowestCommonAncestor::new(&graph, 0);
    for _ in 0..q {
        let k: usize = reader.v();
        let mut vs = reader.vec::<usize>(k);
        for i in 0..k {
            vs[i] -= 1;
        }
        let edges = lcm.auxiliary_tree(&mut vs);
        let mut ans = 0;
        for (a, b) in edges {
            ans += lcm.dist(a, b);
        }
        writer.ln(ans);
    }
}

pub trait GraphTrait {
    type Weight;
    fn size(&self) -> usize;
    fn edges(&self, src: usize) -> Vec<(usize, Self::Weight)>;
    fn rev_edges(&self, dst: usize) -> Vec<(usize, Self::Weight)>;
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

pub struct Graph<W> {
    pub n: usize,
    pub edges: Vec<Vec<(usize, W)>>,
    pub rev_edges: Vec<Vec<(usize, W)>>,
}
impl<W: Clone> GraphTrait for Graph<W> {
    type Weight = W;
    fn size(&self) -> usize {
        self.n
    }
    fn edges(&self, src: usize) -> Vec<(usize, W)> {
        self.edges[src].clone()
    }
    fn rev_edges(&self, dst: usize) -> Vec<(usize, W)> {
        self.rev_edges[dst].clone()
    }
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
impl<W: Clone> Graph<W> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: vec![Vec::new(); n],
            rev_edges: vec![Vec::new(); n],
        }
    }
    pub fn add_edge(&mut self, src: usize, dst: usize, w: W) {
        self.edges[src].push((dst, w.clone()));
        self.edges[dst].push((src, w.clone()));
        self.rev_edges[src].push((dst, w.clone()));
        self.rev_edges[dst].push((src, w));
    }
    pub fn add_arc(&mut self, src: usize, dst: usize, w: W) {
        self.edges[src].push((dst, w.clone()));
        self.rev_edges[dst].push((src, w));
    }
}
impl<W: Debug> Debug for Graph<W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{{").unwrap();
        for i in 0..self.n {
            writeln!(f, "  {{").unwrap();
            for j in 0..self.edges[i].len() {
                writeln!(f, "    {:?}", self.edges[i][j]).unwrap();
            }
            writeln!(f, "  }}").unwrap();
        }
        writeln!(f, "}}")
    }
}
pub struct LowestCommonAncestor {
    tour: EulerTour,
    depth: SparseTable<Minimization<IntWithIndex<usize>>>,
}
impl LowestCommonAncestor {
    pub fn new<G: GraphTrait>(g: &G, root: usize) -> Self {
        let tour = EulerTour::new(g, root);
        let depth = SparseTable::<Minimization<IntWithIndex<usize>>>::from(
            &tour
                .tour
                .iter()
                .map(|i| tour.depth[*i])
                .enumerate()
                .map(IntWithIndex::from)
                .collect::<Vec<_>>()[..],
        );
        Self { tour, depth }
    }
    pub fn query(&self, u: usize, v: usize) -> usize {
        let (mut l, mut r) = (self.tour.time_in[u], self.tour.time_out[v]);
        if l > r {
            swap(&mut l, &mut r)
        }
        self.tour.tour[self.depth.query(l..=r).index]
    }
    pub fn dist(&self, u: usize, v: usize) -> usize {
        let lca = self.query(u, v);
        self.tour.depth[u] + self.tour.depth[v] - 2 * self.tour.depth[lca]
    }
    pub fn on_path(&self, u: usize, v: usize, a: usize) -> bool {
        self.dist(u, a) + self.dist(a, v) == self.dist(u, v)
    }
    pub fn auxiliary_tree(&self, vs: &mut Vec<usize>) -> Vec<(usize, usize)> {
        vs.sort_by_key(|v| self.tour.time_in[*v]);
        let mut stack = vec![vs[0]];
        let mut edges = Vec::new();
        for i in 1..vs.len() {
            let lca = self.query(vs[i - 1], vs[i]);
            if lca != vs[i - 1] {
                let mut last = stack.pop().unwrap();
                while !stack.is_empty()
                    && self.tour.depth[lca] < self.tour.depth[stack[stack.len() - 1]]
                {
                    edges.push((stack[stack.len() - 1], last));
                    last = stack.pop().unwrap();
                }
                if stack.is_empty() || stack[stack.len() - 1] != lca {
                    stack.push(lca);
                    vs.push(lca);
                }
                edges.push((lca, last));
            }
            stack.push(vs[i]);
        }
        for i in 1..stack.len() {
            edges.push((stack[i - 1], stack[i]));
        }
        edges
    }
}

#[derive(Clone, Debug)]
pub struct EulerTour {
    pub time_in: Vec<usize>,
    pub time_out: Vec<usize>,
    pub depth: Vec<usize>,
    pub tour: Vec<usize>,
}
impl EulerTour {
    pub fn new<G: GraphTrait>(g: &G, root: usize) -> Self {
        let mut tour = EulerTour {
            time_in: vec![0; g.size()],
            time_out: vec![0; g.size()],
            depth: vec![0; g.size()],
            tour: Vec::new(),
        };
        tour.dfs(root, root, 0, g);
        tour
    }
    fn dfs<G: GraphTrait>(&mut self, cur: usize, par: usize, d: usize, g: &G) {
        self.depth[cur] = d;
        self.time_in[cur] = self.tour.len();
        self.tour.push(cur);
        for (dst, _) in g.edges(cur) {
            if dst == par {
                continue;
            }
            self.dfs(dst, cur, d + 1, g);
            self.tour.push(cur);
        }
        self.time_out[cur] = self.tour.len();
    }
}

#[derive(Clone)]
pub struct SparseTable<B: Band> {
    size: usize,
    table: Vec<Vec<B::M>>,
}
impl<B: Band> From<&[B::M]> for SparseTable<B> {
    fn from(v: &[B::M]) -> Self {
        let size = v.len();
        let l = v.len();
        let lg = 63 - l.leading_zeros();
        let mut table = vec![Vec::new(); lg as usize + 1];
        table[0] = v.to_vec();
        let mut k = 1;
        while 1 << k <= size {
            table[k] = (0..=size - (1 << k))
                .map(|i| B::op(&table[k - 1][i], &table[k - 1][i + (1 << (k - 1))]))
                .collect();
            k += 1;
        }
        Self { size, table }
    }
}
impl<B: Band> SparseTable<B> {
    pub fn query<R: RangeBounds<usize>>(&self, range: R) -> B::M {
        let (l, r) = to_lr(&range, self.size);
        let lg = 63 - (r - l).leading_zeros();
        B::op(
            &self.table[lg as usize][l],
            &self.table[lg as usize][r - (1 << lg)],
        )
    }
}
impl<B: Band> Debug for SparseTable<B>
where
    B::M: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        for i in 0..self.size {
            writeln!(f, "{:?}", self.query(i..=i))?;
        }
        Ok(())
    }
}

pub trait Magma {
    type M: Clone + PartialEq;
    fn op(x: &Self::M, y: &Self::M) -> Self::M;
}
pub trait Associative {}
pub trait Unital: Magma {
    fn unit() -> Self::M;
}
pub trait Commutative: Magma {}
pub trait Invertible: Magma {
    fn inv(x: &Self::M) -> Self::M;
}
pub trait Idempotent: Magma {}
pub trait SemiGroup: Magma + Associative {}
pub trait Monoid: Magma + Associative + Unital {
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
impl<M: Magma + Associative + Unital> Monoid for M {}
pub trait CommutativeMonoid: Magma + Associative + Unital + Commutative {}
impl<M: Magma + Associative + Unital + Commutative> CommutativeMonoid for M {}
pub trait Group: Magma + Associative + Unital + Invertible {}
impl<M: Magma + Associative + Unital + Invertible> Group for M {}
pub trait AbelianGroup: Magma + Associative + Unital + Commutative + Invertible {}
impl<M: Magma + Associative + Unital + Commutative + Invertible> AbelianGroup for M {}
pub trait Band: Magma + Associative + Idempotent {}
impl<M: Magma + Associative + Idempotent> Band for M {}
pub trait MapMonoid {
    type Mono: Monoid;
    type Func: Monoid;
    fn op(
        &self,
        x: &<Self::Mono as Magma>::M,
        y: &<Self::Mono as Magma>::M,
    ) -> <Self::Mono as Magma>::M {
        Self::Mono::op(x, y)
    }
    fn unit() -> <Self::Mono as Magma>::M {
        Self::Mono::unit()
    }
    fn apply(
        &self,
        f: &<Self::Func as Magma>::M,
        value: &<Self::Mono as Magma>::M,
    ) -> <Self::Mono as Magma>::M;
    fn identity_map() -> <Self::Func as Magma>::M {
        Self::Func::unit()
    }
    fn compose(
        &self,
        f: &<Self::Func as Magma>::M,
        g: &<Self::Func as Magma>::M,
    ) -> <Self::Func as Magma>::M {
        Self::Func::op(f, g)
    }
}
pub trait Zero {
    fn zero() -> Self;
}
pub trait One {
    fn one() -> Self;
}
pub trait BoundedBelow {
    fn min_value() -> Self;
}
pub trait BoundedAbove {
    fn max_value() -> Self;
}
# [rustfmt :: skip ] pub trait Integral : 'static + Send + Sync + Copy + Ord + Display + Debug + Add < Output = Self > + Sub < Output = Self > + Mul < Output = Self > + Div < Output = Self > + Rem < Output = Self > + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + Product + BitOr < Output = Self > + BitAnd < Output = Self > + BitXor < Output = Self > + Not < Output = Self > + Shl < Output = Self > + Shr < Output = Self > + BitOrAssign + BitAndAssign + BitXorAssign + ShlAssign + ShrAssign + Zero + One + BoundedBelow + BoundedAbove {}
macro_rules ! impl_integral {($ ($ ty : ty ) ,* ) => {$ (impl Zero for $ ty {fn zero () -> Self {0 } } impl One for $ ty {fn one () -> Self {1 } } impl BoundedBelow for $ ty {fn min_value () -> Self {Self :: min_value () } } impl BoundedAbove for $ ty {fn max_value () -> Self {Self :: max_value () } } impl Integral for $ ty {} ) * } ; }
impl_integral!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntWithIndex<I: Integral> {
    pub value: I,
    pub index: usize,
}
impl<I: Integral> PartialOrd for IntWithIndex<I> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        use Ordering::*;
        let r = match self.value.cmp(&rhs.value) {
            Greater => Greater,
            Less => Less,
            Equal => self.index.cmp(&rhs.index),
        };
        Some(r)
    }
}
impl<I: Integral> Ord for IntWithIndex<I> {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.partial_cmp(rhs).unwrap()
    }
}
impl<I: Integral> From<(usize, I)> for IntWithIndex<I> {
    fn from((index, value): (usize, I)) -> Self {
        IntWithIndex { value, index }
    }
}

#[derive(Clone, Debug)]
pub struct Minimization<S>(Infallible, PhantomData<fn() -> S>);
impl<S: Clone + PartialOrd> Magma for Minimization<S> {
    type M = S;
    fn op(x: &Self::M, y: &Self::M) -> Self::M {
        if x <= y {
            x.clone()
        } else {
            y.clone()
        }
    }
}
impl<S: BoundedAbove + Clone + PartialOrd> Unital for Minimization<S> {
    fn unit() -> Self::M {
        S::max_value()
    }
}
impl<S: Clone + PartialOrd> Associative for Minimization<S> {}
impl<S: Clone + PartialOrd> Commutative for Minimization<S> {}
impl<S: Clone + PartialOrd> Idempotent for Minimization<S> {}
