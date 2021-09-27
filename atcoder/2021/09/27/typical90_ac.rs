pub use reader::*;
#[allow(unused_imports)]
use {
    itertools::Itertools,
    num::Integer,
    proconio::fastout,
    std::convert::TryInto,
    std::{cmp::*, collections::*, io::*, num::*, str::*},
};

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

#[allow(dead_code)]
#[rustfmt::skip]
pub mod reader { #[allow(unused_imports)] use itertools::Itertools; use std::{fmt::Debug, io::*, str::*};  pub struct Reader<R: BufRead> { reader: R, buf: Vec<u8>, pos: usize, }  macro_rules! prim_method { ($name:ident: $T: ty) => { pub fn $name(&mut self) -> $T { self.n::<$T>() } }; ($name:ident) => { prim_method!($name: $name); } } macro_rules! prim_methods { ($name:ident: $T:ty; $($rest:tt)*) => { prim_method!($name:$T); prim_methods!($($rest)*); }; ($name:ident; $($rest:tt)*) => { prim_method!($name); prim_methods!($($rest)*); }; () => () }  macro_rules! replace_expr { ($_t:tt $sub:expr) => { $sub }; } macro_rules! tuple_method { ($name: ident: ($($T:ident),+)) => { pub fn $name(&mut self) -> ($($T),+) { ($(replace_expr!($T self.n())),+) } } } macro_rules! tuple_methods { ($name:ident: ($($T:ident),+); $($rest:tt)*) => { tuple_method!($name:($($T),+)); tuple_methods!($($rest)*); }; () => () } macro_rules! vec_method { ($name: ident: ($($T:ty),+)) => { pub fn $name(&mut self, n: usize) -> Vec<($($T),+)> { (0..n).map(|_|($(replace_expr!($T self.n())),+)).collect_vec() } }; ($name: ident: $T:ty) => { pub fn $name(&mut self, n: usize) -> Vec<$T> { (0..n).map(|_|self.n()).collect_vec() } }; } macro_rules! vec_methods { ($name:ident: ($($T:ty),+); $($rest:tt)*) => { vec_method!($name:($($T),+)); vec_methods!($($rest)*); }; ($name:ident: $T:ty; $($rest:tt)*) => { vec_method!($name:$T); vec_methods!($($rest)*); }; () => () } impl<R: BufRead> Reader<R> { pub fn new(reader: R) -> Reader<R> { let (buf, pos) = (Vec::new(), 0); Reader { reader, buf, pos } } prim_methods! { u: usize; i: i64; f: f64; str: String; c: char; string: String; u8; u16; u32; u64; u128; usize; i8; i16; i32; i64; i128; isize; f32; f64; char; } tuple_methods! { u2: (usize, usize); u3: (usize, usize, usize); u4: (usize, usize, usize, usize); i2: (i64, i64); i3: (i64, i64, i64); i4: (i64, i64, i64, i64); cuu: (char, usize, usize); } vec_methods! { uv: usize; uv2: (usize, usize); uv3: (usize, usize, usize); iv: i64; iv2: (i64, i64); iv3: (i64, i64, i64); vq: (char, usize, usize); }  pub fn n<T: FromStr>(&mut self) -> T where T::Err: Debug, { self.n_op().unwrap() }  pub fn n_op<T: FromStr>(&mut self) -> Option<T> where T::Err: Debug, { if self.buf.is_empty() { self._read_next_line(); } let mut start = None; while self.pos != self.buf.len() { match (self.buf[self.pos], start.is_some()) { (b' ', true) | (b'\n', true) => break, (_, true) | (b' ', false) => self.pos += 1, (b'\n', false) => self._read_next_line(), (_, false) => start = Some(self.pos), } } start.map(|s| from_utf8(&self.buf[s..self.pos]).unwrap().parse().unwrap()) }  fn _read_next_line(&mut self) { self.pos = 0; self.buf.clear(); self.reader.read_until(b'\n', &mut self.buf).unwrap(); } pub fn s(&mut self) -> Vec<char> { self.n::<String>().chars().collect() } pub fn digits(&mut self) -> Vec<i64> { self.n::<String>() .chars() .map(|c| (c as u8 - b'0') as i64) .collect() } pub fn char_map(&mut self, h: usize) -> Vec<Vec<char>> { (0..h).map(|_| self.s()).collect() } pub fn bool_map(&mut self, h: usize, ng: char) -> Vec<Vec<bool>> { self.char_map(h) .iter() .map(|v| v.iter().map(|&c| c != ng).collect()) .collect() } pub fn matrix(&mut self, h: usize, w: usize) -> Vec<Vec<i64>> { (0..h).map(|_| self.iv(w)).collect() } } }

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    solve(Reader::new(stdin.lock()));
}

/////////////////////////////////////////////////////////

pub use max::*;
pub mod max {
    use super::*;
    use std::convert::Infallible;
    use std::fmt::Debug;
    use std::marker::PhantomData;

    #[derive(Clone, Debug)]
    pub struct Max<S>(Infallible, PhantomData<fn() -> S>);

    impl<S> Magma for Max<S>
        where
            S: BoundedBelow + Copy + Ord + Debug,
    {
        type M = S;

        fn op(x: &Self::M, y: &Self::M) -> Self::M {
            std::cmp::max(*x, *y)
        }
    }

    impl<S> Associative for Max<S> where S: BoundedBelow + Copy + Ord + Debug {}

    impl<S> Unital for Max<S>
        where
            S: BoundedBelow + Copy + Ord + Debug,
    {
        fn unit() -> Self::M {
            S::min_value()
        }
    }
}

pub use algebra::*;
pub mod algebra {
    use std::fmt;
    use std::fmt::Debug;
    use std::iter::{Product, Sum};
    use std::ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
        SubAssign,
    };

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
    pub trait MapMonoid: Debug {
        /// モノイドM
        type Mono: Monoid;
        type Func: Clone + Debug;
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
        fn apply(f: &Self::Func, value: &<Self::Mono as Monoid>::M) -> <Self::Mono as Monoid>::M;
        /// 作用fの単位元
        fn identity_map() -> Self::Func;
        /// 作用fと作用gを合成する
        fn compose(f: &Self::Func, g: &Self::Func) -> Self::Func;
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
    + fmt::Display
    + fmt::Debug
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
}

pub use max_max::*;
pub mod max_max {
    use super::*;
    use std::fmt::Debug;

    #[derive(Debug)]
    pub struct MaxMax;
    impl MapMonoid for MaxMax {
        type Mono = Max<i64>;
        type Func = i64;

        fn apply(f: &Self::Func, value: &<Self::Mono as Monoid>::M) -> <Self::Mono as Monoid>::M {
            std::cmp::max(*f, *value)
        }

        fn identity_map() -> Self::Func {
            <Self::Mono as Monoid>::unit()
        }

        fn compose(f: &Self::Func, g: &Self::Func) -> Self::Func {
            std::cmp::max(*f, *g)
        }
    }
}
#[allow(unused_imports)]
use lazy_segment_tree::*;

#[allow(dead_code)]
pub mod lazy_segment_tree {
    use super::*;

    /// 遅延評価セグメント木
    /// 区間更新、区間取得
    ///
    /// 実装内部は1-indexed
    #[derive(Debug, Clone)]
    pub struct LazySegmentTree<M: MapMonoid> {
        n: usize,
        log: usize,
        node: Vec<<<M as MapMonoid>::Mono as Monoid>::M>,
        lazy: Vec<M::Func>,
    }

    impl<M: MapMonoid> From<&Vec<<M::Mono as Monoid>::M>> for LazySegmentTree<M> {
        fn from(v: &Vec<<M::Mono as Monoid>::M>) -> Self {
            let mut segtree = Self::new(v.len());
            segtree.node[segtree.n..segtree.n + v.len()].clone_from_slice(&v);
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
        pub fn update_at(&mut self, mut i: usize, f: M::Func) {
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
        pub fn update_range(&mut self, mut l: usize, mut r: usize, f: M::Func) {
            assert!(l <= r && r <= self.n);
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
        pub fn get(&mut self, mut i: usize) -> <M::Mono as Monoid>::M {
            assert!(i < self.n);
            i += self.n;
            for j in (1..self.log).rev() {
                self.propagate(i >> j);
            }
            self.node[i].clone()
        }

        /// 区間 $`[l, r)`$ の値を取得する
        /// $`l == r`$ のときは $`unit`$ を返す
        pub fn prod(&mut self, mut l: usize, mut r: usize) -> <M::Mono as Monoid>::M {
            assert!(l <= r && r <= self.n);
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
        fn eval(&mut self, k: usize, f: M::Func) {
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
}

#[fastout]
pub fn solve<R: BufRead>(mut reader: Reader<R>) {
    let (w, n) = reader.u2();

    let mut segtree = LazySegmentTree::<MaxMax>::from(&vec![0; w + 1]);
    for _ in 0..n {
        let (l, r) = reader.u2();
        let max = segtree.prod(l, r + 1);
        println!("{}", max + 1);
        segtree.update_range(l, r + 1, max + 1);
    }
}
