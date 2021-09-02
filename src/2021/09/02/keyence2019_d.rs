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

#[fastout]
pub fn solve<R: BufRead>(mut reader: Reader<R>) {
    let (n, m) = reader.u2();
    let a = reader.uv(n);
    let b = reader.uv(m);
    let a: Vec<_> = a.into_iter().sorted().dedup().collect();
    let b: Vec<_> = b.into_iter().sorted().dedup().collect();

    if a.len() != n || b.len() != m {
        println!("{}", 0);
        return;
    }
    let mut expect = vec![0; n * m + 1];
    let mut fixed = vec![false; n * m + 1];
    for i in 0..n {
        for j in 0..m {
            expect[min(a[i], b[j])] += 1;
            if a[i] == b[j] {
                fixed[a[i]] = true;
            }
        }
    }
    let mvt = ModValTable::new(n * m + 2);
    let mut ans = mi(1);
    let mut rest = 0;
    let mut min = 0;
    for i in 1..=n * m {
        if expect[i] == 0 {
            continue;
        }
        while min + 1 <= i {
            rest += 1;
            min += 1;
        }

        ans *= mvt.permutation(rest - 1, expect[i] - 1);
        if !fixed[i] {
            ans *= expect[i];
        }
        rest -= expect[i];
    }

    println!("{}", ans);
}

#[allow(unused_imports)]
pub use mod_val_table::ModValTable;

#[allow(dead_code)]
mod mod_val_table {
    use super::mod_int::*;

    /// 剰余類Modについて、組み合わせや順列を数え上げる
    #[derive(std::fmt::Debug)]
    pub struct ModValTable<M> {
        fact: Vec<M>,
        fact_inv: Vec<M>,
        inv: Vec<M>,
    }

    impl<M: Mod> ModValTable<ModInt<M>> {
        ///
        /// あるnについてModValTableを初期化する
        ///
        /// nを超える値を呼び出したとき、panicする
        /// ```rust, should_panic
        /// # use atcoder_lib::mod_int::mod_int::Mi;
        /// # use atcoder_lib::mod_val_table::ModValTable;
        /// let fact = ModValTable::<Mi>::new(10);
        /// fact.combination(11, 11);
        /// ```
        pub fn new(n: usize) -> Self {
            let mut fact = vec![ModInt::<M>::new(1); n + 1];
            let mut fact_inv = vec![ModInt::<M>::new(1); n + 1];
            let mut inv = vec![ModInt::<M>::new(1); n + 1];
            for i in 2..=n {
                fact[i] = fact[i - 1] * i as i64;
                inv[i] = inv[0] / i as i64;
                fact_inv[i] = fact_inv[i - 1] * inv[i];
            }
            Self {
                fact,
                fact_inv,
                inv,
            }
        }

        /// Factorial 階乗 n!
        /// ```
        /// # use atcoder_lib::mod_int::mod_int::Mi;
        /// # use atcoder_lib::mod_val_table::ModValTable;
        /// let five = ModValTable::<Mi>::new(5);
        /// let res = vec![1, 1, 2, 6, 24, 120];
        /// for i in 0..=5 {
        ///     assert_eq!(res[i], five.factorial(i as i64).get());
        /// }
        /// ```

        pub fn factorial(&self, n: i64) -> ModInt<M> {
            self.fact[n as usize]
        }

        /// Permutation 順列 nPr = n! / (n - r)!
        /// ```
        /// # use atcoder_lib::mod_int::mod_int::Mi;
        /// # use atcoder_lib::mod_val_table::ModValTable;
        /// let five = ModValTable::<Mi>::new(5);
        /// assert_eq!(1, five.permutation(5, 0).get());
        /// assert_eq!(5, five.permutation(5, 1).get());
        /// assert_eq!(20, five.permutation(5, 2).get());
        /// assert_eq!(60, five.permutation(5, 3).get());
        /// assert_eq!(120, five.permutation(5, 4).get());
        /// assert_eq!(120, five.permutation(5, 5).get());
        /// ```
        pub fn permutation(&self, n: i64, r: i64) -> ModInt<M> {
            if n < r {
                0.into()
            } else {
                self.fact[n as usize] * self.fact_inv[(n - r) as usize]
            }
        }

        /// Combination 組合せ nCr = n! / (n - r)! r! = nPr / r!
        /// Binomial Coefficient
        /// ```
        /// use atcoder_lib::mod_int::mod_int::Mi;
        /// use atcoder_lib::mod_val_table::ModValTable;
        /// let five = ModValTable::<Mi>::new(5);
        /// assert_eq!(1, five.combination(5, 0).get());
        /// assert_eq!(5, five.combination(5, 1).get());
        /// assert_eq!(10, five.combination(5, 2).get());
        /// assert_eq!(10, five.combination(5, 3).get());
        /// assert_eq!(5, five.combination(5, 4).get());
        /// assert_eq!(1, five.combination(5, 5).get());
        /// ```
        pub fn combination(&self, n: i64, r: i64) -> ModInt<M> {
            if n < r {
                0.into()
            } else {
                self.permutation(n, r) * self.fact_inv[r as usize]
            }
        }

        /// Combinations with Replacement 重複組み合わせ nHr = (n+r)! / k!(n-1)!
        pub fn combinations_with_replacement(&self, n: i64, r: i64) -> ModInt<M> {
            if n < r {
                0.into()
            } else {
                self.fact[(n + r) as usize]
                    * self.fact_inv[r as usize]
                    * self.fact_inv[n as usize - 1]
            }
        }
    }
}

#[allow(unused_imports)]
use mod_int::*;

#[allow(dead_code)]
pub mod mod_int {
    use std::marker::PhantomData;
    use std::ops::*;

    pub fn mi(i: i64) -> Mi {
        Mi::new(i)
    }

    pub trait Mod: Copy + Clone + std::fmt::Debug {
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

        pub fn get(&self) -> i64 {
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

    impl<M: Mod> std::fmt::Display for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.n)
        }
    }

    impl<M: Mod> std::fmt::Debug for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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
}
