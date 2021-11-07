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

pub fn to_lr<R: RangeBounds<usize>>(range: R, length: usize) -> (usize, usize) {
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
//////////////////////////////////

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
pub trait SemiGroup: Magma + Associative {}
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

pub struct Gcd<S>(Infallible, PhantomData<fn() -> S>);
impl<S: Clone + RemAssign + PartialOrd + Zero> Magma for Gcd<S> {
    type M = S;
    fn op(x: &S, y: &S) -> S {
        let (mut x, mut y) = (x.clone(), y.clone());
        if y > x {
            swap(&mut x, &mut y);
        }
        while y != S::zero() {
            x %= y.clone();
            swap(&mut x, &mut y);
        }
        x
    }
}
impl<S: Clone + RemAssign + PartialOrd + Zero> Associative for Gcd<S> {}
impl<S: Clone + RemAssign + PartialOrd + Zero> Unital for Gcd<S> {
    fn unit() -> S {
        S::zero()
    }
}
impl<S: Clone + RemAssign + PartialOrd + Zero> Commutative for Gcd<S> {}
impl<S: Clone + RemAssign + PartialOrd + Zero> Idempotent for Gcd<S> {}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let n = reader.val();
    let xy = (0..n)
        .map(|_| (reader.val(), reader.val()))
        .collect::<Vec<(i64, i64)>>();

    let mut set = HashSet::new();
    for i in 0..n {
        let (x, y) = xy[i];
        for j in 0..n {
            if i == j {
                continue;
            }
            let (x2, y2) = xy[j];
            let (mut x2, mut y2) = (x2 - x, y2 - y);
            if x2 < 0 {
                x2 *= -1;
                y2 *= -1;
            }
            let g = Gcd::op(&x2.abs(), &y2.abs());
            x2 /= g;
            y2 /= g;
            if x2 <= 0 && y2 <= 0 {
                x2 *= -1;
                y2 *= -1;
            }
            set.insert((x2, y2));
        }
    }
    writer.println(set.len() * 2);
}
