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
macro_rules! min {($a:expr $(,)*) => {{$a}};($a:expr, $b:expr $(,)*) => {{if $a > $b {$b} else {$a}}};($a:expr, $($rest:expr),+ $(,)*) => {{let b = min!($($rest),+);if $a > b {b} else {$a}}};}
#[allow(unused_macros)]
macro_rules! max {($a:expr $(,)*) => {{$a}};($a:expr, $b:expr $(,)*) => {{if $a > $b {$a} else {$b}}};($a:expr, $($rest:expr),+ $(,)*) => {{let b = max!($($rest),+);if $a > b {$a} else {b}}};}

pub fn to_lr<R: RangeBounds<usize>>(range: &R, length: usize) -> (usize, usize) {
    let l = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
    };
    let r = match range.end_bound() {
        Bound::Unbounded => length,
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
    };
    assert!(l <= r && r <= length);
    (l, r)
}

/// stdin reader
pub struct Reader<R> {
    reader: R,
    buf: VecDeque<String>,
}
impl<R: BufRead> Iterator for Reader<R> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            let mut buf = Vec::new();
            self.reader.read_to_end(&mut buf).unwrap();
            let s = from_utf8(&buf).expect("utf8でない文字列が入力されました.");
            s.split_whitespace()
                .map(ToString::to_string)
                .for_each(|s| self.buf.push_back(s));
        }
        self.buf.pop_front()
    }
}
impl<R: BufRead> Reader<R> {
    pub fn new(reader: R) -> Reader<R> {
        Reader {
            reader,
            buf: VecDeque::new(),
        }
    }
    pub fn val<T: FromStr>(&mut self) -> T {
        self.next()
            .map(|token| token.parse().ok().expect("型変換エラー"))
            .expect("入力が足りません")
    }
    pub fn vec<T: FromStr>(&mut self, length: usize) -> Vec<T> {
        (0..length).map(|_| self.val()).collect()
    }
    pub fn chars(&mut self) -> Vec<char> {
        self.val::<String>().chars().collect()
    }
    pub fn digits(&mut self) -> Vec<i64> {
        self.val::<String>()
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

/// stdin writer
pub struct Writer<W: Write> {
    writer: BufWriter<W>,
}
impl<W: Write> Writer<W> {
    pub fn new(write: W) -> Self {
        Self {
            writer: BufWriter::new(write),
        }
    }
    pub fn println<S: Display>(&mut self, s: S) {
        writeln!(self.writer, "{}", s).expect("Failed to write.")
    }
    pub fn print<S: Display>(&mut self, s: S) {
        write!(self.writer, "{}", s).expect("Failed to write.")
    }
    pub fn print_join<S: Display>(&mut self, v: &[S], separator: &str) {
        v.iter().fold("", |sep, arg| {
            write!(self.writer, "{}{}", sep, arg).expect("Failed to write.");
            separator
        });
        writeln!(self.writer).expect("Failed to write.");
    }
}

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}

pub fn mi(i: i64) -> Mi {
    Mi::new(i)
}

pub trait Mod: Copy + Clone + Debug {
    fn get() -> i64;
}

pub type Mi = ModInt<Mod1e9p7>;

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
pub struct ModInt<M: Mod> {
    n: i64,
    _p: PhantomData<M>,
}

impl<M: Mod> ModInt<M> {
    pub fn new(n: i64) -> Self {
        Self {
            n: n.rem_euclid(M::get()),
            _p: PhantomData,
        }
    }

    pub fn pow(mut self, mut e: i64) -> ModInt<M> {
        let mut result = Self::new(1);
        while e > 0 {
            if e & 1 == 1 {
                result *= self.n;
            }
            e >>= 1;
            self *= self.n;
        }
        result
    }

    pub fn get(self) -> i64 {
        self.n
    }
}

impl<M: Mod> Add<i64> for ModInt<M> {
    type Output = Self;
    fn add(self, rhs: i64) -> Self {
        ModInt::new(self.n + rhs.rem_euclid(M::get()))
    }
}

impl<M: Mod> Add<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self + rhs.n
    }
}

impl<M: Mod> AddAssign<i64> for ModInt<M> {
    fn add_assign(&mut self, rhs: i64) {
        *self = *self + rhs
    }
}

impl<M: Mod> AddAssign<ModInt<M>> for ModInt<M> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl<M: Mod> Neg for ModInt<M> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.n)
    }
}

impl<M: Mod> Sub<i64> for ModInt<M> {
    type Output = Self;
    fn sub(self, rhs: i64) -> Self {
        ModInt::new(self.n - rhs.rem_euclid(M::get()))
    }
}

impl<M: Mod> Sub<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self - rhs.n
    }
}

impl<M: Mod> SubAssign<i64> for ModInt<M> {
    fn sub_assign(&mut self, rhs: i64) {
        *self = *self - rhs
    }
}

impl<M: Mod> SubAssign<ModInt<M>> for ModInt<M> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}

impl<M: Mod> Mul<i64> for ModInt<M> {
    type Output = Self;
    fn mul(self, rhs: i64) -> Self {
        ModInt::new(self.n * (rhs % M::get()))
    }
}

impl<M: Mod> Mul<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self * rhs.n
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
        self / rhs.n
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
        write!(f, "{}", self.n)
    }
}

impl<M: Mod> Debug for ModInt<M> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self.n)
    }
}

impl<M: Mod> Deref for ModInt<M> {
    type Target = i64;
    fn deref(&self) -> &Self::Target {
        &self.n
    }
}

impl<M: Mod> DerefMut for ModInt<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.n
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
        m.n
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
    /// add x to i
    pub fn add(&mut self, i: usize, x: A::M) {
        let mut idx = i as i32 + 1;
        while idx <= self.n as i32 {
            self.bit[idx as usize] = A::op(&self.bit[idx as usize], &x);
            idx += idx & -idx;
        }
    }

    /// sum of [0, i)
    pub fn sum(&self, i: usize) -> A::M {
        let mut ret = A::unit();
        let mut idx = i as i32 + 1;
        while idx > 0 {
            ret = A::op(&ret, &self.bit[idx as usize]);
            idx -= idx & -idx;
        }
        ret
    }

    /// sum of range
    pub fn fold<R: RangeBounds<usize>>(&self, range: &R) -> A::M {
        let (a, b) = to_lr(range, self.n);
        if b == 0 {
            A::unit()
        } else if a == 0 {
            self.sum(b - 1)
        } else {
            A::op(&self.sum(b - 1), &A::inv(&self.sum(a - 1)))
        }
    }
}

impl<A: AbelianGroup> Debug for BinaryIndexedTree<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let v = (0..self.n)
            .map(|i| format!("{:?}", self.fold(&(i..=i))))
            .collect::<Vec<_>>()
            .join(" ");
        let v2 = (0..self.n)
            .map(|i| format!("{:?}", self.sum(i)))
            .collect::<Vec<_>>()
            .join(" ");
        write!(f, "\n{}\n{}", v, v2)
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

/// 単位的
pub trait Unital: Magma {
    /// 単位元 identity element: $`e`$
    fn unit() -> Self::M;
}

/// 可換
pub trait Commutative: Magma {}

/// 可逆的
/// $`\exists e \in T, \forall a \in T, \exists b,c \in T, b \circ a = a \circ c = e`$
pub trait Invertible: Magma {
    /// $`a`$ where $`a \circ x = e`$
    fn inv(x: &Self::M) -> Self::M;
}

/// 冪等性
pub trait Idempotent: Magma {}

/// 半群
/// 1. 結合則
pub trait SemiGroup {}
impl<M: Magma + Associative> SemiGroup for M {}

/// モノイド
/// 1. 結合則
/// 1. 単位元
pub trait Monoid: Magma + Associative + Unital {
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
impl<M: Magma + Associative + Unital> Monoid for M {}

/// 可換モノイド
pub trait CommutativeMonoid: Magma + Associative + Unital + Commutative {}
impl<M: Magma + Associative + Unital + Commutative> CommutativeMonoid for M {}

/// 群
/// 1. 結合法則
/// 1. 単位元
/// 1. 逆元
pub trait Group: Magma + Associative + Unital + Invertible {}
impl<M: Magma + Associative + Unital + Invertible> Group for M {}

/// アーベル群
pub trait AbelianGroup: Magma + Associative + Unital + Commutative + Invertible {}
impl<M: Magma + Associative + Unital + Commutative + Invertible> AbelianGroup for M {}

/// Band
/// 1. 結合法則
/// 1. 冪等律
pub trait Band: Magma + Associative + Idempotent {}
impl<M: Magma + Associative + Idempotent> Band for M {}

/// 作用付きモノイド
pub trait MapMonoid {
    /// モノイドM
    type Mono: Monoid;
    type Func: Monoid;
    /// 値xと値yを併合する
    fn op(x: &<Self::Mono as Magma>::M, y: &<Self::Mono as Magma>::M) -> <Self::Mono as Magma>::M {
        Self::Mono::op(x, y)
    }
    fn unit() -> <Self::Mono as Magma>::M {
        Self::Mono::unit()
    }
    /// 作用fをvalueに作用させる
    fn apply(
        f: &<Self::Func as Magma>::M,
        value: &<Self::Mono as Magma>::M,
    ) -> <Self::Mono as Magma>::M;
    /// 作用fの単位元
    fn identity_map() -> <Self::Func as Magma>::M {
        Self::Func::unit()
    }
    /// composition:
    /// $`h() = f(g())`$
    fn compose(
        f: &<Self::Func as Magma>::M,
        g: &<Self::Func as Magma>::M,
    ) -> <Self::Func as Magma>::M {
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
#[rustfmt::skip]
pub trait Integral: 'static + Send + Sync + Copy + Ord + Display + Debug
+ Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Rem<Output = Self>
+ AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + Product
+ BitOr<Output = Self> + BitAnd<Output = Self> + BitXor<Output = Self> + Not<Output = Self> + Shl<Output = Self> + Shr<Output = Self>
+ BitOrAssign + BitAndAssign + BitXorAssign + ShlAssign + ShrAssign
+ Zero + One + BoundedBelow + BoundedAbove{}

macro_rules! impl_integral {
    ($($ty:ty),*) => {
        $(
            impl Zero for $ty { fn zero() -> Self { 0 }}
            impl One for $ty { fn one() -> Self { 1 }}
            impl BoundedBelow for $ty { fn min_value() -> Self { Self::min_value() }}
            impl BoundedAbove for $ty { fn max_value() -> Self { Self::max_value() }}
            impl Integral for $ty {}
        )*
    };
}
impl_integral!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

#[derive(Clone, Debug)]
pub struct Addition<S>(Infallible, PhantomData<fn() -> S>);
impl<S: Zero + Copy + Add<Output = S> + Ord + Debug> Magma for Addition<S> {
    type M = S;
    fn op(x: &S, y: &S) -> S {
        *x + *y
    }
}
impl<S: Zero + Copy + Add<Output = S> + Ord + Debug> Associative for Addition<S> {}
impl<S: Zero + Copy + Add<Output = S> + Ord + Debug> Unital for Addition<S> {
    fn unit() -> S {
        S::zero()
    }
}
impl<S: Zero + Copy + Add<Output = S> + Ord + Debug> Commutative for Addition<S> {}
impl<S: Zero + Copy + Add<Output = S> + Ord + Debug + Neg<Output = S>> Invertible for Addition<S> {
    fn inv(x: &S) -> S {
        (*x).neg()
    }
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let (n, k) = (reader.val::<usize>(), reader.val::<usize>());
    let a = reader.vec::<usize>(n);
    let mut bit = BinaryIndexedTree::<Addition<i64>>::from(2001);
    let mut f = vec![0; n];
    let mut g = vec![0; n];
    for i in 0..n {
        f[i] = i as i64 - bit.sum(a[i]);
        bit.add(a[i], 1);
    }
    for i in 0..n {
        g[i] = n as i64 - bit.sum(a[i]);
    }

    let mut ans = mi(0);
    for i in 0..n {
        ans += k as i64 * f[i];
        ans += mi(k as i64) * (k - 1) as i64 * g[i] / 2;
    }
    writer.println(ans);
}
