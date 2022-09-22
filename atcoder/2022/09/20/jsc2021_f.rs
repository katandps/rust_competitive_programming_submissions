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
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
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
#[derive(Clone, Default)]
pub struct DynamicSegmentTree<M: Monoid> {
    root: dynamic_segment_tree_impl::OptionalNode<M>,
}
mod dynamic_segment_tree_impl {
    use super::{swap, to_lr, Debug, DynamicSegmentTree, Monoid, RangeBounds};
    type IndexType = i64;
    impl<M: Monoid> DynamicSegmentTree<M> {
        const BIT_LEN: i32 = 62;
        const MAX: IndexType = 1 << Self::BIT_LEN;
        pub fn set(&mut self, i: IndexType, value: M::M) {
            self.root.set(i, Self::BIT_LEN - 1, value);
        }
        pub fn apply<F: Fn(M::M) -> M::M>(&mut self, i: IndexType, f: F) {
            self.root.apply(i, Self::BIT_LEN - 1, f)
        }
        pub fn get(&self, i: IndexType) -> M::M {
            self.root.get(i, Self::BIT_LEN - 1)
        }
        pub fn prod<R: RangeBounds<IndexType>>(&self, range: R) -> M::M {
            let (l, r) = to_lr(&range, Self::MAX);
            self.root.prod(l, r, 0, Self::MAX)
        }
    }
    #[derive(Clone, Debug, Default)]
    pub struct OptionalNode<M: Monoid>(Option<Node<M>>);
    #[derive(Clone, Debug, Default)]
    struct Node<M: Monoid> {
        value: M::M,
        l: Box<OptionalNode<M>>,
        r: Box<OptionalNode<M>>,
    }
    impl<M: Monoid> OptionalNode<M> {
        pub fn new(value: M::M) -> Self {
            Self(Some(Node {
                value,
                l: Box::new(Self(None)),
                r: Box::new(Self(None)),
            }))
        }
        pub fn set(&mut self, idx: IndexType, bit: i32, value: M::M) {
            match self.0.as_mut() {
                Some(node) if bit < 0 => node.value = value,
                Some(node) => {
                    node.child_mut(idx, bit).set(idx, bit - 1, value);
                    node.value = M::op(
                        &node.l.prod(0, 1 << 62, 0, 1 << bit),
                        &node.r.prod(0, 1 << 62, 0, 1 << bit),
                    )
                }
                None if bit < 0 => swap(self, &mut Self::new(value)),
                None => {
                    swap(self, &mut Self::new(value.clone()));
                    self.set(idx, bit, value);
                }
            }
        }
        pub fn apply<F: Fn(M::M) -> M::M>(&mut self, idx: IndexType, bit: i32, f: F) {
            match self.0.as_mut() {
                Some(node) if bit < 0 => node.value = f(node.value.clone()),
                Some(node) => {
                    node.child_mut(idx, bit).apply(idx, bit - 1, f);
                    node.value = M::op(
                        &node.l.prod(0, 1 << 62, 0, 1 << bit),
                        &node.r.prod(0, 1 << 62, 0, 1 << bit),
                    )
                }
                None if bit < 0 => swap(self, &mut Self::new(f(M::unit()))),
                None => {
                    swap(self, &mut Self::new(M::unit()));
                    self.apply(idx, bit, f);
                }
            }
        }
        pub fn get(&self, idx: IndexType, bit: i32) -> M::M {
            match &self.0 {
                Some(node) if bit < 0 => node.value.clone(),
                Some(node) => node.child(idx, bit).get(idx, bit - 1),
                None => M::unit(),
            }
        }
        pub fn prod(&self, l: IndexType, r: IndexType, lb: IndexType, ub: IndexType) -> M::M {
            match &self.0 {
                Some(node) if l <= lb && ub <= r => node.value.clone(),
                Some(node) if lb < r && l < ub => M::op(
                    &node.l.prod(l, r, lb, (lb + ub) >> 1),
                    &node.r.prod(l, r, (lb + ub) >> 1, ub),
                ),
                _ => M::unit(),
            }
        }
    }
    impl<M: Monoid> Node<M> {
        fn child_mut(&mut self, idx: IndexType, bit: i32) -> &mut OptionalNode<M> {
            match () {
                () if idx >> bit & 1 == 0 => self.l.as_mut(),
                _ => self.r.as_mut(),
            }
        }
        fn child(&self, idx: IndexType, bit: i32) -> &OptionalNode<M> {
            match () {
                () if idx >> bit & 1 == 0 => &self.l,
                _ => &self.r,
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Addition<S>(PhantomData<fn() -> S>);
mod addition_impl {
    use super::{
        Add, Addition, Associative, Commutative, Debug, Invertible, Magma, Neg, Unital, Zero,
    };
    impl<S: Clone + Debug + Add<Output = S> + PartialEq> Magma for Addition<S> {
        type M = S;
        fn op(x: &S, y: &S) -> S {
            x.clone() + y.clone()
        }
    }
    impl<S: Clone + Debug + Add<Output = S> + PartialEq> Associative for Addition<S> {}
    impl<S: Clone + Debug + Add<Output = S> + PartialEq + Zero> Unital for Addition<S> {
        fn unit() -> S {
            S::zero()
        }
    }
    impl<S: Clone + Debug + Add<Output = S> + PartialEq> Commutative for Addition<S> {}
    impl<S: Clone + Debug + Add<Output = S> + PartialEq + Neg<Output = S>> Invertible for Addition<S> {
        fn inv(x: &S) -> S {
            x.clone().neg()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
struct CountSum(i64, i64);
impl Add for CountSum {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        CountSum(self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl Zero for CountSum {
    fn zero() -> Self {
        CountSum(0, 0)
    }
}
impl Neg for CountSum {
    type Output = Self;
    fn neg(self) -> Self {
        CountSum(-self.0, -self.1)
    }
}

pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let (n, m, q) = reader.v3::<usize, usize, usize>();
    let txy = reader.vec3::<usize, usize, i64>(q);
    let mut a = vec![0; n];
    let mut b = vec![0; m];
    let mut a_seg = DynamicSegmentTree::<Addition<CountSum>>::default();
    let mut b_seg = DynamicSegmentTree::<Addition<CountSum>>::default();
    a_seg.set(0, CountSum(n as i64, 0));
    b_seg.set(0, CountSum(m as i64, 0));
    let mut sum = 0;
    for (t, x, y) in txy {
        let (v, cur, other) = if t == 1 {
            (&mut a, &mut a_seg, &mut b_seg)
        } else {
            (&mut b, &mut b_seg, &mut a_seg)
        };
        let i = x - 1;
        let line_sum = other.prod(v[i]..).1 + other.prod(..v[i]).0 * v[i];
        sum -= line_sum;
        cur.apply(v[i], |cs| CountSum(cs.0 - 1, cs.1 - v[i]));
        v[i] = y;
        cur.apply(v[i], |cs| CountSum(cs.0 + 1, cs.1 + v[i]));
        let line_sum = other.prod(v[i]..).1 + other.prod(..v[i]).0 * v[i];
        sum += line_sum;

        writer.ln(sum);
    }
}
