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
    let q: usize = reader.v();
    let x: usize = reader.v();
    let p = reader.vec::<usize>(n);
    let clr = reader.vec3::<usize, usize, usize>(q);
    let mut segtree_large = LazySegmentTree::<AddSum>::from((AddSum, n));
    let mut segtree_small = LazySegmentTree::<AddSum>::from((AddSum, n));
    for i in 0..n {
        if p[i] < x {
            segtree_small.update_at(i, Some(1));
        } else if p[i] > x {
            segtree_large.update_at(i, Some(1));
        }
    }
    for (c, l, r) in clr {
        let (l, r) = (l - 1, r);

        let large = segtree_large.prod(l..r).value as usize;
        let small = segtree_small.prod(l..r).value as usize;
        segtree_large.update_range(l..r, Some(0));
        segtree_small.update_range(l..r, Some(0));

        // dbg!(large, small);
        if c == 1 {
            segtree_large.update_range(r - large..r, Some(1));
            segtree_small.update_range(l..l + small, Some(1));
        } else {
            segtree_large.update_range(l..l + large, Some(1));
            segtree_small.update_range(r - small..r, Some(1));
        }
    }
    for i in 0..n {
        if segtree_large.get(i).value == 0 && segtree_small.get(i).value == 0 {
            return writer.ln(i + 1);
        }
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

#[derive(Clone)]
pub struct LazySegmentTree<M: MapMonoid> {
    m: M,
    n: usize,
    log: usize,
    node: Vec<<M::Mono as Magma>::M>,
    lazy: Vec<<M::Func as Magma>::M>,
}
impl<M: MapMonoid> From<(M, usize)> for LazySegmentTree<M> {
    fn from((m, length): (M, usize)) -> Self {
        let n = (length + 1).next_power_of_two();
        let log = n.trailing_zeros() as usize;
        let node = vec![M::unit(); 2 * n];
        let lazy = vec![M::identity_map(); n];
        let mut tree = Self {
            m,
            n,
            log,
            node,
            lazy,
        };
        (1..n).rev().for_each(|i| tree.calc(i));
        tree
    }
}
impl<M: MapMonoid> From<(M, &Vec<<M::Mono as Magma>::M>)> for LazySegmentTree<M> {
    fn from((m, v): (M, &Vec<<M::Mono as Magma>::M>)) -> Self {
        let mut segtree = Self::from((m, v.len() + 1));
        segtree.node[segtree.n..segtree.n + v.len() - 1].clone_from_slice(v);
        (0..segtree.n - 1).rev().for_each(|i| segtree.calc(i));
        segtree
    }
}
impl<M: MapMonoid> LazySegmentTree<M> {
    pub fn update_at(&mut self, mut i: usize, f: <M::Func as Magma>::M) {
        assert!(i < self.n);
        i += self.n;
        (1..=self.log).rev().for_each(|j| self.propagate(i >> j));
        self.node[i] = self.m.apply(&f, &self.node[i]);
        (1..=self.log).for_each(|j| self.calc(i >> j));
    }
    pub fn update_range<R: RangeBounds<usize>>(&mut self, range: R, f: <M::Func as Magma>::M) {
        let (mut l, mut r) = to_lr(&range, self.n);
        if l == r {
            return;
        }
        l += self.n;
        r += self.n;
        for i in (1..=self.log).rev() {
            if ((l >> i) << i) != l {
                self.propagate(l >> i);
            }
            if ((r >> i) << i) != r {
                self.propagate((r - 1) >> i);
            }
        }
        {
            let l2 = l;
            let r2 = r;
            while l < r {
                if l & 1 != 0 {
                    self.eval(l, f.clone());
                    l += 1;
                }
                if r & 1 != 0 {
                    r -= 1;
                    self.eval(r, f.clone());
                }
                l >>= 1;
                r >>= 1;
            }
            l = l2;
            r = r2;
        }
        for i in 1..=self.log {
            if ((l >> i) << i) != l {
                self.calc(l >> i);
            }
            if ((r >> i) << i) != r {
                self.calc((r - 1) >> i);
            }
        }
    }
    pub fn get(&mut self, mut i: usize) -> <M::Mono as Magma>::M {
        assert!(i < self.n);
        i += self.n;
        for j in (1..=self.log).rev() {
            self.propagate(i >> j);
        }
        self.node[i].clone()
    }
    pub fn prod<R: RangeBounds<usize>>(&mut self, range: R) -> <M::Mono as Magma>::M {
        let (mut l, mut r) = to_lr(&range, self.n);
        if l == r {
            return M::unit();
        }
        l += self.n;
        r += self.n;
        for i in (1..=self.log).rev() {
            if ((l >> i) << i) != l {
                self.propagate(l >> i);
            }
            if ((r >> i) << i) != r {
                self.propagate(r >> i);
            }
        }
        let mut sml = M::unit();
        let mut smr = M::unit();
        while l < r {
            if l & 1 != 0 {
                sml = self.m.op(&sml, &self.node[l]);
                l += 1;
            }
            if r & 1 != 0 {
                r -= 1;
                smr = self.m.op(&self.node[r], &smr);
            }
            l >>= 1;
            r >>= 1;
        }
        self.m.op(&sml, &smr)
    }
    fn calc(&mut self, k: usize) {
        assert!(2 * k + 1 < self.node.len());
        self.node[k] = self.m.op(&self.node[2 * k], &self.node[2 * k + 1]);
    }
    fn eval(&mut self, k: usize, f: <M::Func as Magma>::M) {
        self.node[k] = self.m.apply(&f, &self.node[k]);
        if k < self.n {
            self.lazy[k] = self.m.compose(&self.lazy[k], &f);
        }
    }
    fn propagate(&mut self, k: usize) {
        self.eval(2 * k, self.lazy[k].clone());
        self.eval(2 * k + 1, self.lazy[k].clone());
        self.lazy[k] = M::identity_map();
    }
}

pub struct Addition<S>(Infallible, PhantomData<fn() -> S>);
impl<S: Clone + Add<Output = S> + PartialEq> Magma for Addition<S> {
    type M = S;
    fn op(x: &S, y: &S) -> S {
        x.clone() + y.clone()
    }
}
impl<S: Clone + Add<Output = S> + PartialEq> Associative for Addition<S> {}
impl<S: Clone + Add<Output = S> + PartialEq + Zero> Unital for Addition<S> {
    fn unit() -> S {
        S::zero()
    }
}
impl<S: Clone + Add<Output = S> + PartialEq> Commutative for Addition<S> {}
impl<S: Clone + Add<Output = S> + PartialEq + Neg<Output = S>> Invertible for Addition<S> {
    fn inv(x: &S) -> S {
        x.clone().neg()
    }
}

pub struct OverwriteOperation<S>(Infallible, PhantomData<fn() -> S>);
impl<S: Clone + PartialEq> Magma for OverwriteOperation<S> {
    type M = Option<S>;
    fn op(x: &Self::M, y: &Self::M) -> Self::M {
        match (x, y) {
            (_, Some(y)) => Some(y.clone()),
            (Some(x), _) => Some(x.clone()),
            _ => None,
        }
    }
}
impl<S: Clone + PartialEq> Unital for OverwriteOperation<S> {
    fn unit() -> Self::M {
        None
    }
}
impl<S: Clone + PartialEq> Associative for OverwriteOperation<S> {}
impl<S: Clone + PartialEq> Idempotent for OverwriteOperation<S> {}

pub struct AddSum;
impl MapMonoid for AddSum {
    type Mono = Addition<Segment<i64>>;
    type Func = OverwriteOperation<i64>;

    fn apply(
        &self,
        f: &<Self::Func as Magma>::M,
        value: &<Self::Mono as Magma>::M,
    ) -> <Self::Mono as Magma>::M {
        if let Some(write) = f {
            Segment {
                value: value.size * *write,
                size: value.size,
            }
        } else {
            value.clone()
        }
    }
}

#[derive(Clone, PartialEq, Ord, PartialOrd, Eq)]
pub struct Segment<M: Clone + PartialEq> {
    pub value: M,
    size: i64,
}
impl<M: Clone + PartialEq + Display> Debug for Segment<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "v: {}, size: {}", self.value, self.size)
    }
}
impl<M: Clone + PartialEq + Add<Output = M> + Zero> Add for Segment<M> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (value, size) = (self.value + rhs.value, self.size + rhs.size);
        Self { value, size }
    }
}
impl<M: Clone + PartialEq + Zero> Zero for Segment<M> {
    fn zero() -> Self {
        let (value, size) = (M::zero(), 1);
        Self { value, size }
    }
}
impl<M: Clone + PartialEq + Add<Output = M>> Add<M> for Segment<M> {
    type Output = Self;
    fn add(self, rhs: M) -> Self {
        let (value, size) = (self.value + rhs, self.size);
        Self { value, size }
    }
}
impl<M: Clone + PartialEq + Mul<Output = M>> Mul<M> for Segment<M> {
    type Output = Self;
    fn mul(self, rhs: M) -> Self {
        let (value, size) = (self.value * rhs, self.size);
        Self { value, size }
    }
}
