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
    let mut ans = mi(1);
    let mvt = ModValTable::new(n + 1);
    let mut bit = BinaryIndexedTree::<Addition<i64>>::from(n);
    for i in (0..n).rev() {
        let ai = a[i];
        ans += mvt.factorial((n - i - 1) as i64) * bit.fold(0..ai);
        bit.add(ai, 1);
    }
    writer.ln(ans);
}
#[derive(Debug)]
pub struct ModValTable<M> {
    fact: Vec<M>,
    fact_inv: Vec<M>,
}
impl<M: Mod> ModValTable<ModInt<M>> {
    pub fn new(n: usize) -> Self {
        let mut fact = vec![ModInt::<M>::new(1); n + 1];
        let mut fact_inv = vec![ModInt::<M>::new(1); n + 1];
        let mut inv = vec![ModInt::<M>::new(1); n + 1];
        for i in 2..=n {
            fact[i] = fact[i - 1] * i as i64;
            inv[i] = inv[0] / i as i64;
            fact_inv[i] = fact_inv[i - 1] * inv[i];
        }
        Self { fact, fact_inv }
    }
    pub fn factorial(&self, n: i64) -> ModInt<M> {
        self.fact[n as usize]
    }
    pub fn permutation(&self, n: i64, r: i64) -> ModInt<M> {
        if n < r {
            0.into()
        } else {
            self.fact[n as usize] * self.fact_inv[(n - r) as usize]
        }
    }
    pub fn combination(&self, n: i64, r: i64) -> ModInt<M> {
        if n < r {
            0.into()
        } else {
            self.permutation(n, r) * self.fact_inv[r as usize]
        }
    }
    pub fn combinations_with_replacement(&self, n: i64, r: i64) -> ModInt<M> {
        if n < r {
            0.into()
        } else {
            self.fact[(n + r) as usize] * self.fact_inv[r as usize] * self.fact_inv[n as usize - 1]
        }
    }
}

pub type Mi = ModInt<Mod1e9p7>;
pub fn mi(i: i64) -> Mi {
    Mi::new(i)
}
pub trait Mod: Copy + Clone + Debug {
    fn get() -> i64;
}
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Mod1e9p7;
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Mod1e9p9;
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Mod998244353;
impl Mod for Mod1e9p7 {
    fn get() -> i64 {
        1_000_000_007
    }
}
impl Mod for Mod1e9p9 {
    fn get() -> i64 {
        1_000_000_009
    }
}
impl Mod for Mod998244353 {
    fn get() -> i64 {
        998_244_353
    }
}
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct ModInt<M: Mod>(i64, PhantomData<fn() -> M>);
impl<M: Mod> ModInt<M> {
    pub fn new(mut n: i64) -> Self {
        if n < 0 || n >= M::get() {
            n = n.rem_euclid(M::get());
        }
        Self(n, PhantomData)
    }
    pub fn pow(mut self, mut e: i64) -> ModInt<M> {
        let m = e < 0;
        e = e.abs();
        let mut result = Self::new(1);
        while e > 0 {
            if e & 1 == 1 {
                result *= self.0;
            }
            e >>= 1;
            self *= self.0;
        }
        if m {
            Self::new(1) / result
        } else {
            result
        }
    }
    pub fn get(self) -> i64 {
        self.0
    }
}
impl<M: Mod> Add<i64> for ModInt<M> {
    type Output = Self;
    fn add(self, rhs: i64) -> Self {
        self + ModInt::new(rhs)
    }
}
impl<M: Mod> Add<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}
impl<M: Mod> AddAssign<i64> for ModInt<M> {
    fn add_assign(&mut self, rhs: i64) {
        *self = *self + rhs
    }
}
impl<M: Mod> AddAssign<ModInt<M>> for ModInt<M> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = if self.0 + rhs.0 >= M::get() {
            self.0 + rhs.0 - M::get()
        } else {
            self.0 + rhs.0
        }
    }
}
impl<M: Mod> Neg for ModInt<M> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.0)
    }
}
impl<M: Mod> Sub<i64> for ModInt<M> {
    type Output = Self;
    fn sub(self, rhs: i64) -> Self {
        self - ModInt::new(rhs)
    }
}
impl<M: Mod> Sub<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}
impl<M: Mod> SubAssign<i64> for ModInt<M> {
    fn sub_assign(&mut self, rhs: i64) {
        *self = *self - rhs
    }
}
impl<M: Mod> SubAssign<ModInt<M>> for ModInt<M> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            self.0 - rhs.0 + M::get()
        }
    }
}
impl<M: Mod> Mul<i64> for ModInt<M> {
    type Output = Self;
    fn mul(self, rhs: i64) -> Self {
        ModInt::new(self.0 * (rhs % M::get()))
    }
}
impl<M: Mod> Mul<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self * rhs.0
    }
}
impl<M: Mod> MulAssign<i64> for ModInt<M> {
    fn mul_assign(&mut self, rhs: i64) {
        *self = *self * rhs
    }
}
impl<M: Mod> MulAssign<ModInt<M>> for ModInt<M> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}
impl<M: Mod> Div<i64> for ModInt<M> {
    type Output = Self;
    fn div(self, rhs: i64) -> Self {
        self * ModInt::new(rhs).pow(M::get() - 2)
    }
}
impl<M: Mod> Div<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self / rhs.0
    }
}
impl<M: Mod> DivAssign<i64> for ModInt<M> {
    fn div_assign(&mut self, rhs: i64) {
        *self = *self / rhs
    }
}
impl<M: Mod> DivAssign<ModInt<M>> for ModInt<M> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs
    }
}
impl<M: Mod> Display for ModInt<M> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl<M: Mod> Debug for ModInt<M> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl<M: Mod> Deref for ModInt<M> {
    type Target = i64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<M: Mod> DerefMut for ModInt<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<M: Mod> Sum for ModInt<M> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(0), |x, a| x + a)
    }
}
impl<M: Mod> From<i64> for ModInt<M> {
    fn from(i: i64) -> Self {
        Self::new(i)
    }
}
impl<M: Mod> From<ModInt<M>> for i64 {
    fn from(m: ModInt<M>) -> Self {
        m.0
    }
}
impl<M: Mod> Zero for ModInt<M> {
    fn zero() -> Self {
        Self::new(0)
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

#[derive(Clone)]
pub struct BinaryIndexedTree<A: Magma> {
    n: usize,
    bit: Vec<A::M>,
}
impl<A: AbelianGroup> From<usize> for BinaryIndexedTree<A> {
    fn from(length: usize) -> Self {
        Self {
            n: length,
            bit: vec![A::unit(); length + 1],
        }
    }
}
impl<A: AbelianGroup> From<&[A::M]> for BinaryIndexedTree<A> {
    fn from(src: &[A::M]) -> Self {
        let mut bit = Self::from(src.len());
        src.iter()
            .enumerate()
            .for_each(|(i, item)| bit.add(i + 1, item.clone()));
        bit
    }
}
impl<A: AbelianGroup> BinaryIndexedTree<A> {
    pub fn add(&mut self, i: usize, x: A::M) {
        let mut idx = i as i32 + 1;
        while idx <= self.n as i32 {
            self.bit[idx as usize] = A::op(&self.bit[idx as usize], &x);
            idx += idx & -idx;
        }
    }
    pub fn sum(&self, i: usize) -> A::M {
        let mut ret = A::unit();
        let mut idx = i as i32 + 1;
        while idx > 0 {
            ret = A::op(&ret, &self.bit[idx as usize]);
            idx -= idx & -idx;
        }
        ret
    }
    pub fn fold<R: RangeBounds<usize>>(&self, range: R) -> A::M {
        let (a, b) = to_lr(&range, self.n);
        if b == 0 {
            A::unit()
        } else if a == 0 {
            self.sum(b - 1)
        } else {
            A::op(&self.sum(b - 1), &A::inv(&self.sum(a - 1)))
        }
    }
}
impl<A: AbelianGroup> Debug for BinaryIndexedTree<A>
where
    A::M: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let v = (0..self.n)
            .map(|i| format!("{:?}", self.fold(i..=i)))
            .collect::<Vec<_>>()
            .join(" ");
        let v2 = (0..self.n)
            .map(|i| format!("{:?}", self.sum(i)))
            .collect::<Vec<_>>()
            .join(" ");
        write!(f, "\n{}\n{}", v, v2)
    }
}
