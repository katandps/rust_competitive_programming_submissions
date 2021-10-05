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

/// 遅延評価セグメント木
/// 区間更新、区間取得
///
/// 実装内部は1-indexed
#[derive(Debug, Clone)]
pub struct LazySegmentTree<M: MapMonoid> {
    n: usize,
    log: usize,
    node: Vec<<<M as MapMonoid>::Mono as Monoid>::M>,
    lazy: Vec<<M::Func as Monoid>::M>,
}

/// 1-indexedで配列の内容を詰めたセグメント木を生成する
impl<M: MapMonoid> From<&Vec<<M::Mono as Monoid>::M>> for LazySegmentTree<M> {
    fn from(v: &Vec<<M::Mono as Monoid>::M>) -> Self {
        let mut segtree = Self::new(v.len() + 1);
        segtree.node[segtree.n + 1..segtree.n + v.len()].clone_from_slice(&v);
        for i in (0..segtree.n - 1).rev() {
            segtree.calc(i);
        }
        segtree
    }
}
impl<M: MapMonoid> LazySegmentTree<M> {
    pub fn new(n: usize) -> Self {
        let n = (n + 1).next_power_of_two();
        let log = n.trailing_zeros() as usize;
        let node = vec![M::unit(); 2 * n];
        let lazy = vec![M::identity_map(); n];
        let mut segtree = Self { n, log, node, lazy };
        for i in (1..n).rev() {
            segtree.calc(i)
        }
        segtree
    }

    /// 一点更新
    pub fn update_at(&mut self, mut i: usize, f: <M::Func as Monoid>::M) {
        assert!(i < self.n);
        i += self.n;
        for j in (1..=self.log).rev() {
            self.propagate(i >> j);
        }
        self.node[i] = M::apply(&f, &self.node[i]);
        for j in 1..=self.log {
            self.calc(i >> j)
        }
    }

    /// 区間更新 [l, r)
    pub fn update_range<R: RangeBounds<usize>>(&mut self, range: R, f: <M::Func as Monoid>::M) {
        let (mut l, mut r) = self.to_lr(range);
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

    fn to_lr<R: RangeBounds<usize>>(&self, range: R) -> (usize, usize) {
        use Bound::*;
        let l = match range.start_bound() {
            Unbounded => 0,
            Included(&s) => s,
            Excluded(&s) => s + 1,
        };
        let r = match range.end_bound() {
            Unbounded => self.n,
            Included(&e) => e + 1,
            Excluded(&e) => e,
        };
        assert!(l <= r && r <= self.n);
        (l, r)
    }

    /// i番目の値を取得する
    pub fn get(&mut self, mut i: usize) -> <M::Mono as Monoid>::M {
        assert!(i < self.n);
        i += self.n;
        for j in (1..=self.log).rev() {
            self.propagate(i >> j);
        }
        self.node[i].clone()
    }

    /// 区間 $`[l, r)`$ の値を取得する
    /// $`l == r`$ のときは $`unit`$ を返す
    pub fn prod<R: RangeBounds<usize>>(&mut self, range: R) -> <M::Mono as Monoid>::M {
        let (mut l, mut r) = self.to_lr(range);
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
                sml = M::op(&sml, &self.node[l]);
                l += 1;
            }
            if r & 1 != 0 {
                r -= 1;
                smr = M::op(&self.node[r], &smr);
            }
            l >>= 1;
            r >>= 1;
        }
        M::op(&sml, &smr)
    }

    /// k番目の区間を内包する区間の値から計算する
    fn calc(&mut self, k: usize) {
        assert!(2 * k + 1 < self.node.len());
        self.node[k] = M::op(&self.node[2 * k], &self.node[2 * k + 1]);
    }

    /// k番目の区間の値に作用を適用する
    fn eval(&mut self, k: usize, f: <M::Func as Monoid>::M) {
        self.node[k] = M::apply(&f, &self.node[k]);
        if k < self.n {
            self.lazy[k] = M::compose(&f, &self.lazy[k]);
        }
    }

    /// k番目の区間に作用を適用し、その区間が含む区間に作用を伝播させる
    fn propagate(&mut self, k: usize) {
        self.eval(2 * k, self.lazy[k].clone());
        self.eval(2 * k + 1, self.lazy[k].clone());
        self.lazy[k] = M::identity_map();
    }
}

#[derive(Clone, Debug)]
pub struct Min<S>(Infallible, PhantomData<fn() -> S>);

impl<S> Magma for Min<S>
    where
        S: BoundedAbove + Copy + Ord + Debug,
{
    type M = S;

    fn op(x: &Self::M, y: &Self::M) -> Self::M {
        std::cmp::min(*x, *y)
    }
}

impl<S> Associative for Min<S> where S: BoundedAbove + Copy + Ord + Debug {}

impl<S> Unital for Min<S>
    where
        S: BoundedAbove + Copy + Ord + Debug,
{
    fn unit() -> Self::M {
        S::max_value()
    }
}

#[derive(Debug)]
struct UpdateMin;
impl MapMonoid for UpdateMin {
    type Mono = Min<i64>;
    type Func = Min<i64>;

    fn apply(
        f: &<Self::Func as Monoid>::M,
        value: &<Self::Mono as Monoid>::M,
    ) -> <Self::Mono as Monoid>::M {
        min(*f, *value)
    }
}

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut output: Writer<W>) {
    let (n, q) = reader.u2();
    let mut row: LazySegmentTree<UpdateMin> = LazySegmentTree::new(n);
    row.update_range(.., n as i64 - 1);
    let mut col: LazySegmentTree<UpdateMin> = LazySegmentTree::new(n);
    col.update_range(.., n as i64 - 1);

    let mut ans = (n as i64 - 2) * (n as i64 - 2);
    for _ in 0..q {
        match reader.u() {
            1 => {
                let c = reader.i() - 1;
                let r = row.get(c as usize);
                ans -= r - 1;
                col.update_range(..r as usize, c);
            }
            2 => {
                let r = reader.i() - 1;
                let c = col.get(r as usize);
                ans -= c - 1;
                row.update_range(..c as usize, r);
            }
            _ => unreachable!(),
        }
    }
    output.println(&ans);
}
