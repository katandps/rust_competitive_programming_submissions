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

/// マグマ
/// 二項演算: $`M \circ M \to M`$
pub trait Magma {
    /// マグマを構成する集合$`M`$
    type M: Clone + PartialEq;
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
    /// 作用fをvalueに作用させる
    fn apply(
        &self,
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
        &self,
        f: &<Self::Func as Magma>::M,
        g: &<Self::Func as Magma>::M,
    ) -> <Self::Func as Magma>::M {
        Self::Func::op(g, f)
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

/// 1-indexedで配列の内容を詰めたセグメント木を生成する
impl<M: MapMonoid> From<(M, &Vec<<M::Mono as Magma>::M>)> for LazySegmentTree<M> {
    fn from((m, v): (M, &Vec<<M::Mono as Magma>::M>)) -> Self {
        let mut segtree = Self::from((m, v.len() + 1));
        segtree.node[segtree.n..segtree.n + v.len() - 1].clone_from_slice(v);
        (0..segtree.n - 1).rev().for_each(|i| segtree.calc(i));
        segtree
    }
}
impl<M: MapMonoid> LazySegmentTree<M> {
    /// 一点更新
    pub fn update_at(&mut self, mut i: usize, f: <M::Func as Magma>::M) {
        assert!(i < self.n);
        i += self.n;
        (1..=self.log).rev().for_each(|j| self.propagate(i >> j));
        self.node[i] = self.m.apply(&f, &self.node[i]);
        (1..=self.log).for_each(|j| self.calc(i >> j));
    }

    /// 区間更新 [l, r)
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

    /// i番目の値を取得する
    pub fn get(&mut self, mut i: usize) -> <M::Mono as Magma>::M {
        assert!(i < self.n);
        i += self.n;
        for j in (1..=self.log).rev() {
            self.propagate(i >> j);
        }
        self.node[i].clone()
    }

    /// 区間 $`[l, r)`$ の値を取得する
    /// $`l == r`$ のときは $`unit`$ を返す
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

    /// k番目の区間を内包する区間の値から計算する
    fn calc(&mut self, k: usize) {
        assert!(2 * k + 1 < self.node.len());
        self.node[k] = self.m.op(&self.node[2 * k], &self.node[2 * k + 1]);
    }

    /// k番目の区間の値に作用を適用する
    fn eval(&mut self, k: usize, f: <M::Func as Magma>::M) {
        self.node[k] = self.m.apply(&f, &self.node[k]);
        if k < self.n {
            self.lazy[k] = self.m.compose(&f, &self.lazy[k]);
        }
    }

    /// k番目の区間に作用を適用し、その区間が含む区間に作用を伝播させる
    fn propagate(&mut self, k: usize) {
        self.eval(2 * k, self.lazy[k].clone());
        self.eval(2 * k + 1, self.lazy[k].clone());
        self.lazy[k] = M::identity_map();
    }
}

pub fn mi(i: i64) -> Mi {
    Mi::new(i)
}

pub trait Mod: Copy + Clone + Debug {
    fn get() -> i64;
}

pub type Mi = ModInt<Mod998244353>;
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
    _p: PhantomData<fn() -> M>,
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
        self + ModInt::new(rhs.rem_euclid(M::get()))
    }
}
impl<M: Mod> Add<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut n = self.n + rhs.n;
        if n >= M::get() {
            n -= M::get();
        }
        Self { n, _p: self._p }
    }
}
impl<M: Mod> AddAssign<i64> for ModInt<M> {
    #[inline]
    fn add_assign(&mut self, rhs: i64) {
        *self = *self + rhs
    }
}
impl<M: Mod> AddAssign<ModInt<M>> for ModInt<M> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}
impl<M: Mod> Neg for ModInt<M> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.n)
    }
}
impl<M: Mod> Sub<i64> for ModInt<M> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: i64) -> Self {
        ModInt::new(self.n - rhs.rem_euclid(M::get()))
    }
}
impl<M: Mod> Sub<ModInt<M>> for ModInt<M> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self - rhs.n
    }
}
impl<M: Mod> SubAssign<i64> for ModInt<M> {
    #[inline]
    fn sub_assign(&mut self, rhs: i64) {
        *self = *self - rhs
    }
}
impl<M: Mod> SubAssign<ModInt<M>> for ModInt<M> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs
    }
}
impl<M: Mod> Mul<i64> for ModInt<M> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: i64) -> Self {
        ModInt::new(self.n * (rhs % M::get()))
    }
}
impl<M: Mod> Mul<ModInt<M>> for ModInt<M> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs.n
    }
}
impl<M: Mod> MulAssign<i64> for ModInt<M> {
    #[inline]
    fn mul_assign(&mut self, rhs: i64) {
        *self = *self * rhs
    }
}
impl<M: Mod> MulAssign<ModInt<M>> for ModInt<M> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}
impl<M: Mod> Div<i64> for ModInt<M> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: i64) -> Self {
        self * ModInt::new(rhs).pow(M::get() - 2)
    }
}
impl<M: Mod> Div<ModInt<M>> for ModInt<M> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self / rhs.n
    }
}
impl<M: Mod> DivAssign<i64> for ModInt<M> {
    #[inline]
    fn div_assign(&mut self, rhs: i64) {
        *self = *self / rhs
    }
}
impl<M: Mod> DivAssign<ModInt<M>> for ModInt<M> {
    #[inline]
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

impl<M: Mod> Zero for ModInt<M> {
    fn zero() -> Self {
        Self::new(0)
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

/// 区間和を求めるセグメント木に載せる値
/// ### algo
/// 例えば、[0, 4)の区間に3を足した時、 合計の値は3に区間の幅をかけた12増える
/// 区間の長さを持たせることで計算できるようになる
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

struct M {
    ten_pow: Vec<Mi>,
    ones: Vec<Mi>,
}
impl MapMonoid for M {
    type Mono = Addition<Segment<Mi>>;
    type Func = OverwriteOperation<Mi>;

    fn op(
        &self,
        x: &<Self::Mono as Magma>::M,
        y: &<Self::Mono as Magma>::M,
    ) -> <Self::Mono as Magma>::M {
        x.clone() + y.clone() * self.ten_pow[x.size as usize]
    }

    fn apply(
        &self,
        f: &<Self::Func as Magma>::M,
        value: &<Self::Mono as Magma>::M,
    ) -> <Self::Mono as Magma>::M {
        if let Some(f) = f {
            Segment {
                value: *f * self.ones[value.size as usize],
                size: value.size,
            }
        } else {
            value.clone()
        }
    }
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let (n, q) = (reader.val::<usize>(), reader.val::<usize>());
    let mut ones = vec![mi(0)];
    for i in 0..200100 {
        ones.push(ones[i] * 10 + 1);
    }
    let mut ten_pow = vec![mi(1)];
    for i in 0..200100 {
        ten_pow.push(ten_pow[i] * 10);
    }
    let m = M { ones, ten_pow };
    let mut segtree: LazySegmentTree<M> = LazySegmentTree::from((m, n));

    segtree.update_range(0..n, Some(mi(1)));
    for _ in 0..q {
        let (l, r, d) = (
            reader.val::<usize>(),
            reader.val::<usize>(),
            reader.val::<usize>(),
        );

        segtree.update_range(n - r..n - l + 1, Some(mi(d as i64)));
        writer.println(segtree.prod(0..n).value / 10);
    }
}
