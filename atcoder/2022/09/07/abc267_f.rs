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
    pub fn bits(&mut self, i: i64, len: usize) {
        (0..len).for_each(|b| write!(self.writer, "{}", i >> b & 1).expect("Failed to write."));
        writeln!(self.writer).expect("Failed to write.")
    }
    pub fn flush(&mut self) {
        let _ = self.writer.flush();
    }
}
pub struct Reader<F> {
    init: F,
    buf: VecDeque<String>,
}
impl<R: BufRead, F: FnMut() -> R> Iterator for Reader<F> {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            let mut reader = (self.init)();
            let mut l = String::new();
            reader.read_line(&mut l).unwrap();
            self.buf
                .append(&mut l.split_whitespace().map(ToString::to_string).collect());
        }
        self.buf.pop_front()
    }
}
impl<R: BufRead, F: FnMut() -> R> Reader<F> {
    pub fn new(init: F) -> Self {
        let buf = VecDeque::new();
        Reader { init, buf }
    }
    pub fn v<T: FS>(&mut self) -> T {
        let s = self.next().expect("Insufficient input.");
        s.parse().ok().expect("Failed to parse.")
    }
    pub fn v2<T1: FS, T2: FS>(&mut self) -> (T1, T2) {
        (self.v(), self.v())
    }
    pub fn v3<T1: FS, T2: FS, T3: FS>(&mut self) -> (T1, T2, T3) {
        (self.v(), self.v(), self.v())
    }
    pub fn v4<T1: FS, T2: FS, T3: FS, T4: FS>(&mut self) -> (T1, T2, T3, T4) {
        (self.v(), self.v(), self.v(), self.v())
    }
    pub fn v5<T1: FS, T2: FS, T3: FS, T4: FS, T5: FS>(&mut self) -> (T1, T2, T3, T4, T5) {
        (self.v(), self.v(), self.v(), self.v(), self.v())
    }
    pub fn vec<T: FS>(&mut self, length: usize) -> Vec<T> {
        (0..length).map(|_| self.v()).collect()
    }
    pub fn vec2<T1: FS, T2: FS>(&mut self, length: usize) -> Vec<(T1, T2)> {
        (0..length).map(|_| self.v2()).collect()
    }
    pub fn vec3<T1: FS, T2: FS, T3: FS>(&mut self, length: usize) -> Vec<(T1, T2, T3)> {
        (0..length).map(|_| self.v3()).collect()
    }
    pub fn vec4<T1: FS, T2: FS, T3: FS, T4: FS>(&mut self, length: usize) -> Vec<(T1, T2, T3, T4)> {
        (0..length).map(|_| self.v4()).collect()
    }
    pub fn chars(&mut self) -> Vec<char> {
        self.v::<String>().chars().collect()
    }
    fn split(&mut self, zero: u8) -> Vec<usize> {
        self.v::<String>()
            .chars()
            .map(|c| (c as u8 - zero) as usize)
            .collect()
    }
    pub fn digits(&mut self) -> Vec<usize> {
        self.split(b'0')
    }
    pub fn lowercase(&mut self) -> Vec<usize> {
        self.split(b'a')
    }
    pub fn uppercase(&mut self) -> Vec<usize> {
        self.split(b'A')
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
    pub fn matrix<T: FS>(&mut self, h: usize, w: usize) -> Vec<Vec<T>> {
        (0..h).map(|_| self.vec(w)).collect()
    }
}
pub fn to_lr<T, R: RangeBounds<T>>(range: &R, length: T) -> (T, T)
where
    T: Copy + One + Zero + Add<Output = T> + PartialOrd,
{
    use Bound::{Excluded, Included, Unbounded};
    let l = match range.start_bound() {
        Unbounded => T::zero(),
        Included(&s) => s,
        Excluded(&s) => s + T::one(),
    };
    let r = match range.end_bound() {
        Unbounded => length,
        Included(&e) => e + T::one(),
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
    iter::{repeat, Product, Sum},
    marker::PhantomData,
    mem::swap,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Bound,
        Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Range,
        RangeBounds, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
    str::{from_utf8, FromStr as FS},
};
#[allow(unused_macros)]
macro_rules ! chmin {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_min = min ! ($ ($ cmps ) ,+ ) ; if $ base > cmp_min {$ base = cmp_min ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! min {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ b } else {$ a } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = min ! ($ ($ rest ) ,+ ) ; if $ a > b {b } else {$ a } } } ; }
#[allow(unused_macros)]
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
pub trait Magma {
    type M: Clone + PartialEq + Debug;
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
pub fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(|| stdin.lock()), Writer::new(stdout.lock()));
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
    pub edges: Vec<(usize, usize, W)>,
    pub index: Vec<Vec<usize>>,
    pub rev_index: Vec<Vec<usize>>,
}
impl<W: Clone> GraphTrait for Graph<W> {
    type Weight = W;
    fn size(&self) -> usize {
        self.n
    }
    fn edges(&self, src: usize) -> Vec<(usize, W)> {
        self.index[src]
            .iter()
            .map(|i| {
                let (_src, dst, w) = &self.edges[*i];
                (*dst, w.clone())
            })
            .collect()
    }
    fn rev_edges(&self, dst: usize) -> Vec<(usize, W)> {
        self.rev_index[dst]
            .iter()
            .map(|i| {
                let (src, _dst, w) = &self.edges[*i];
                (*src, w.clone())
            })
            .collect()
    }
}
impl<W: Clone> Clone for Graph<W> {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            edges: self.edges.clone(),
            index: self.index.clone(),
            rev_index: self.rev_index.clone(),
        }
    }
}
impl<W: Clone> Graph<W> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            index: vec![Vec::new(); n],
            rev_index: vec![Vec::new(); n],
        }
    }
    pub fn add_edge(&mut self, src: usize, dst: usize, w: W) -> (usize, usize) {
        let i = self.edges.len();
        self.edges.push((src, dst, w.clone()));
        self.index[src].push(i);
        self.rev_index[dst].push(i);
        let j = self.edges.len();
        self.edges.push((dst, src, w));
        self.index[dst].push(j);
        self.rev_index[src].push(j);
        (i, j)
    }
    pub fn add_arc(&mut self, src: usize, dst: usize, w: W) -> usize {
        let i = self.edges.len();
        self.edges.push((src, dst, w));
        self.index[src].push(i);
        self.rev_index[dst].push(i);
        i
    }
    pub fn all_edges(&self) -> Vec<(usize, usize, W)> {
        self.edges.clone()
    }
}
impl<W: Debug> Debug for Graph<W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "n: {}, m: {}", self.n, self.edges.len()).unwrap();
        for (src, dst, w) in &self.edges {
            writeln!(f, "({} -> {}): {:?}", src, dst, w).unwrap();
        }
        Ok(())
    }
}
pub fn rank<G: GraphTrait>(g: &G) -> Vec<i64> {
    let mut rank = vec![None; g.size()];
    for i in 0..g.size() {
        if rank[i].is_none() {
            rank[i] = Some(0);
            rank_dfs(i, i, g, &mut rank);
        }
    }
    rank.into_iter().flatten().collect()
}
fn rank_dfs<G: GraphTrait>(cur: usize, par: usize, g: &G, rank: &mut Vec<Option<i64>>) {
    for (dst, _weight) in g.edges(cur) {
        if dst == par {
            continue;
        }
        rank[dst] = rank[cur].map(|k| k + 1);
        rank_dfs(dst, cur, g, rank);
    }
}

pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let n: usize = reader.v();
    let ab = reader.vec2::<usize, usize>(n - 1);
    let q = reader.v::<usize>();
    let uk = reader.vec2::<usize, usize>(q);
    let mut graph = Graph::new(n);
    for (a, b) in ab {
        graph.add_edge(a - 1, b - 1, 1);
    }
    let r = rank(&graph);
    let mut h = 0;
    let mut l = 0;
    for i in 0..n {
        if chmax!(h, r[i]) {
            l = i;
        }
    }
    let mut rank = vec![None; graph.size()];
    rank[l] = Some(0);
    rank_dfs(l, l, &graph, &mut rank);
    let mut h = 0;
    let mut r = 0;
    for i in 0..n {
        let k = rank[i].unwrap();
        if chmax!(h, k) {
            r = i;
        }
    }
    let mut lparent = vec![None; n];
    parent_dfs(l, l, &graph, &mut lparent);
    let mut rparent = vec![None; n];
    parent_dfs(r, r, &graph, &mut rparent);
    // dbg!(lparent, rparent);
    let ld = doubling(&lparent[..], 20);
    let rd = doubling(&rparent[..], 20);
    // dbg!(&ld);
    for (u, k) in uk {
        let u = u - 1;
        let mut cur = Some(u);
        for t in 0..20 {
            if let Some(c) = cur {
                if k >> t & 1 == 1 {
                    // dbg!(t, c);
                    cur = ld[t + 1][c];
                }
            } else {
                break;
            }
        }
        if let Some(c) = cur {
            writer.ln(c + 1);
            continue;
        }
        let mut cur = Some(u);
        for t in 0..20 {
            if let Some(c) = cur {
                if k >> t & 1 == 1 {
                    cur = rd[t + 1][c];
                }
            } else {
                break;
            }
        }
        if let Some(c) = cur {
            writer.ln(c + 1);
        } else {
            writer.ln(-1);
        }
    }
    // dbg!(ld, rd);
}

fn parent_dfs<G: GraphTrait>(src: usize, par: usize, graph: &G, parent: &mut Vec<Option<usize>>) {
    for (dst, _weight) in graph.edges(src) {
        if dst == par {
            continue;
        }
        if parent[dst].is_none() {
            parent[dst] = Some(src);
            parent_dfs(dst, src, graph, parent)
        }
    }
}

fn doubling(src: &[Option<usize>], len: usize) -> Vec<Vec<Option<usize>>> {
    let mut ret = vec![Vec::new(); len + 1];
    ret[1] = src.to_vec();
    for t in 1..len {
        for i in 0..src.len() {
            let next = if let Some(to) = ret[t][i] {
                ret[t][to]
            } else {
                None
            };
            ret[t + 1].push(next);
        }
    }
    ret
}
