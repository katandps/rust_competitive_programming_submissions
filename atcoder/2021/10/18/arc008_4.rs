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

impl<M: SemiGroup + Unital> Monoid for M {
    type M = M::M;
    fn op(x: &M::M, y: &M::M) -> M::M {
        M::op(x, y)
    }

    fn unit() -> Self::M {
        M::unit()
    }
}

/// 可逆的
/// $`\exists e \in T, \forall a \in T, \exists b,c \in T, b \circ a = a \circ c = e`$
pub trait Invertible: Magma {
    /// $`a`$ where $`a \circ x = e`$
    fn inv(&self, x: &Self::M) -> Self::M;
}

pub trait InvertibleMonoid: Monoid + Invertible {}
impl<M: Monoid + Invertible> InvertibleMonoid for M {}

/// 群
pub trait Group {}
impl<M: Monoid + Invertible> Group for M {}

/// 作用付きモノイド
/// 値Mono、作用Funcはモノイドで、
pub trait MapMonoid {
    /// モノイドM
    type Mono: Monoid;
    type Func: Monoid;
    /// 値xと値yを併合する
    fn op(
        x: &<Self::Mono as Monoid>::M,
        y: &<Self::Mono as Monoid>::M,
    ) -> <Self::Mono as Monoid>::M {
        Self::Mono::op(x, y)
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
    /// composition:
    /// $`h() = f(g())`$
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

/// 動的セグメント木
/// セグメント木よりメモリアクセスが遅いが、メモリ使用量は挿入したノードの数を上界とする。
/// データの挿入が$`O(logN)`$となっていることに注意。
#[derive(Clone, Debug)]
pub struct DynamicSegmentTree<M: Monoid> {
    root: node::Node<M>,
}

impl<M: Monoid> Default for DynamicSegmentTree<M> {
    fn default() -> Self {
        Self {
            root: node::Node::<M>::default(),
        }
    }
}

impl<M: Monoid> DynamicSegmentTree<M> {
    /// 最大幅を $`2^{BIT_LEN}`$ とする
    pub const BIT_LEN: i32 = 62;
    /// 値iをvalueに更新する
    /// ## 計算量
    /// $`O(logN)`$
    pub fn set(&mut self, i: u64, value: M::M) {
        self.root.set(i, Self::BIT_LEN - 1, value);
    }
    /// 値iをvalueに更新する
    /// ## 計算量
    /// $`O(logN)`$
    pub fn get(&self, i: u64) -> M::M {
        self.root.get(i, Self::BIT_LEN - 1)
    }
    /// Rangeで与えられた区間の値を取得する
    /// ## 計算量
    /// $`O(logN)`$
    pub fn prod<R>(&self, range: R) -> M::M
        where
            R: RangeBounds<usize>,
    {
        let (l, r) = Self::make_lr(range);
        self.root.prod(l, r, 0, 1 << Self::BIT_LEN)
    }

    /// Range to [l, r)
    fn make_lr<R: RangeBounds<usize>>(range: R) -> (u64, u64) {
        use Bound::*;
        let l = match range.start_bound() {
            Unbounded => 0,
            Included(&s) => s,
            Excluded(&s) => s + 1,
        };
        let r = match range.end_bound() {
            Unbounded => 1 << Self::BIT_LEN,
            Included(&e) => e + 1,
            Excluded(&e) => e,
        };
        (l as u64, r as u64)
    }
}

mod node {
    use super::Monoid;
    #[derive(Clone, Debug)]
    pub struct Node<M: Monoid> {
        value: M::M,
        child: Vec<Option<Node<M>>>,
    }

    impl<M: Monoid> Default for Node<M> {
        fn default() -> Self {
            Self {
                value: M::unit(),
                child: vec![None, None],
            }
        }
    }

    impl<M: Monoid> Node<M> {
        pub fn set(&mut self, pos: u64, bit: i32, value: M::M) {
            if bit < 0 {
                self.value = value;
                return;
            }
            let dst = (pos >> bit & 1) as usize;
            if let Some(c) = self.child[dst].as_mut() {
                c.set(pos, bit - 1, value);
            } else {
                let mut node = Node::default();
                node.set(pos, bit - 1, value);
                self.child[dst] = Some(node);
            }
            self.value = M::op(
                &self.child[0]
                    .as_ref()
                    .map_or(M::unit(), |c| c.value.clone()),
                &self.child[1]
                    .as_ref()
                    .map_or(M::unit(), |c| c.value.clone()),
            );
        }

        pub fn get(&self, pos: u64, bit: i32) -> M::M {
            if bit < 0 {
                return self.value.clone();
            }
            let dst = (pos >> bit & 1) as usize;
            if let Some(c) = &self.child[dst] {
                c.get(pos, bit - 1)
            } else {
                M::unit()
            }
        }

        /// [left, right)のうち、[lower_bound, upper_bound)の内部にあるものをprodして返す
        pub fn prod(&self, left: u64, right: u64, lower_bound: u64, upper_bound: u64) -> M::M {
            if right <= lower_bound || upper_bound <= left {
                M::unit()
            } else if left <= lower_bound && upper_bound <= right {
                self.value.clone()
            } else {
                let mid = (lower_bound + upper_bound) >> 1;
                M::op(
                    &self.child[0]
                        .as_ref()
                        .map_or(M::unit(), |c| c.prod(left, right, lower_bound, mid)),
                    &self.child[1]
                        .as_ref()
                        .map_or(M::unit(), |c| c.prod(left, right, mid, upper_bound)),
                )
            }
        }
    }
}

#[derive(Default)]
struct M;
impl Associative for M {}
impl Unital for M {
    fn unit() -> Self::M {
        (1.0, 0.0)
    }
}
impl Magma for M {
    type M = (f64, f64);

    fn op(x: &Self::M, y: &Self::M) -> Self::M {
        (x.0 * y.0, x.1 * y.0 + y.1)
    }
}

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let (_n, m) = reader.u2();

    let mut segtree = DynamicSegmentTree::<M>::default();
    let mut min = 1.0;
    let mut max = 1.0;
    for _ in 0..m {
        let p = reader.u();
        let (a, b) = (reader.f(), reader.f());
        segtree.set(p as u64, (a, b));
        let k = segtree.prod(..);
        let ai = 1.0 * k.0 + k.1;
        if ai < min {
            min = ai;
        }
        if ai > max {
            max = ai;
        }
    }
    writer.println(&min);
    writer.println(&max);
}
