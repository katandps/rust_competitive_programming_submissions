pub use reader::*;
#[allow(unused_imports)]
use {
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
pub use reader::*;

#[allow(dead_code)]
pub mod reader {
    #[allow(unused_imports)]
    use std::{fmt::Debug, io::*, str::*};

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
}

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    solve(Reader::new(stdin.lock()));
}

pub use algebra::*;
pub mod algebra {
    use std::fmt;
    use std::iter::{Product, Sum};
    use std::ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Not, Rem, RemAssign, Sub, SubAssign,
    };

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

    /// 半群
    pub trait SemiGroup: Magma + Associative {}

    /// 単位的
    pub trait Unital: Magma {
        /// 単位元 identity element: $`e`$
        fn unit() -> Self::M;
    }

    /// モノイド
    /// 結合則と、単位元を持つ
    pub trait Monoid: SemiGroup + Unital {
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

    impl<M: SemiGroup + Unital> Monoid for M {}

    /// 可逆的
    /// $`\exists e \in T, \forall a \in T, \exists b,c \in T, b \circ a = a \circ c = e`$
    pub trait Invertible: Magma {
        /// $`a`$ where $`a \circ x = e`$
        fn inv(&self, x: &Self::M) -> Self::M;
    }

    /// 群
    pub trait Group: Monoid + Invertible {}

    /// 作用付きモノイド
    pub trait MapMonoid {
        /// モノイドM
        type M: Monoid;
        type F: Clone;
        /// 値xと値yを併合する
        fn op(x: &<Self::M as Magma>::M, y: &<Self::M as Magma>::M) -> <Self::M as Magma>::M {
            Self::M::op(&x, &y)
        }
        /// 作用fをvalueに作用させる
        fn apply(f: &Self::F, value: &<Self::M as Magma>::M) -> <Self::M as Magma>::M;
        /// 作用fと作用gを合成する
        fn compose(f: &Self::F, g: &Self::F) -> Self::F;
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

pub use min::*;
pub mod min {
    use super::*;
    use std::convert::Infallible;
    use std::marker::PhantomData;

    #[derive(Clone, Debug)]
    pub struct Min<S>(Infallible, PhantomData<fn() -> S>);

    impl<S> SemiGroup for Min<S> where S: BoundedAbove + Copy + Ord {}

    impl<S> Magma for Min<S>
        where
            S: BoundedAbove + Copy + Ord,
    {
        type M = S;

        fn op(x: &Self::M, y: &Self::M) -> Self::M {
            std::cmp::min(*x, *y)
        }
    }

    impl<S> Associative for Min<S> where S: BoundedAbove + Copy + Ord {}

    impl<S> Unital for Min<S>
        where
            S: BoundedAbove + Copy + Ord,
    {
        fn unit() -> Self::M {
            S::max_value()
        }
    }
}

#[allow(unused_imports)]
use segment_tree::*;

#[allow(dead_code)]
pub mod segment_tree {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct SegmentTree<M: Monoid> {
        n: usize,
        node: Vec<M::M>,
    }

    impl<M: Monoid> SegmentTree<M> {
        pub fn new(v: &Vec<M::M>) -> Self {
            let n = (v.len() + 1).next_power_of_two();
            let mut node = vec![M::unit(); 2 * n - 1];
            for i in 0..v.len() {
                node[i + n - 1] = v[i].clone();
            }
            for i in (0..n - 1).rev() {
                node[i] = M::op(&node[2 * i + 1], &node[2 * i + 2]);
            }
            Self { n, node }
        }

        /// 値iをvalueに更新する
        pub fn update_at(&mut self, mut i: usize, value: M::M) {
            i += self.n - 1;
            self.node[i] = value;
            while i > 0 {
                i = (i - 1) / 2;
                self.node[i] = M::op(&self.node[2 * i + 1], &self.node[2 * i + 2]);
            }
        }

        /// 区間[a, b)の値を取得する
        pub fn get(&self, a: usize, b: usize) -> M::M {
            self.g(a, b, None, None, None)
        }

        /// k: 自分がいるノードのインデックス
        /// l, r: 対象区間 [l, r)
        fn g(
            &self,
            a: usize,
            b: usize,
            k: Option<usize>,
            l: Option<usize>,
            r: Option<usize>,
        ) -> M::M {
            let (k, l, r) = (k.unwrap_or(0), l.unwrap_or(0), r.unwrap_or(self.n));
            if r <= a || b <= l {
                M::unit()
            } else if a <= l && r <= b {
                self.node[k].clone()
            } else {
                M::op(
                    &self.g(a, b, Some(2 * k + 1), Some(l), Some((l + r) / 2)),
                    &self.g(a, b, Some(2 * k + 2), Some((l + r) / 2), Some(r)),
                )
            }
        }
    }
}

pub fn solve<R: BufRead>(mut reader: Reader<R>) {
    let (n, q) = reader.u2();
    let initial = vec![(1 << 31) - 1; n];

    let mut segtree: SegmentTree<Min<_>> = SegmentTree::new(&initial);

    for _ in 0..q {
        let com = reader.u();
        let (x, y) = reader.u2();
        match com {
            0 => {
                segtree.update_at(x, y as i64);
            }
            1 => {
                println!("{}", segtree.get(x, y + 1));
            }
            _ => unreachable!(),
        }
    }
}
