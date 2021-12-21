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
        let m = e < 0;
        e = e.abs();
        let mut result = Self::new(1);
        while e > 0 {
            if e & 1 == 1 {
                result *= self.n;
            }
            e >>= 1;
            self *= self.n;
        }
        if m {
            Self::new(1) / result
        } else {
            result
        }
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
        self - ModInt::new(rhs.rem_euclid(M::get()))
    }
}
impl<M: Mod> Sub<ModInt<M>> for ModInt<M> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut n = self.n - rhs.n;
        if n < 0 {
            n += M::get();
        }
        Self { n, _p: self._p }
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
pub fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(|| stdin.lock()), Writer::new(stdout.lock()));
}
pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let d: usize = reader.v();
    let n = reader.digits();
    //dp[上限まで使っている][Dで割ったあまり]
    let mut dp = vec![vec![mi(0); d]; 2];
    dp[1][0] += 1;
    for i in 0..n.len() {
        let mut next = vec![vec![mi(0); d]; 2];
        let ni = n[i] as usize;
        for src in 0..d {
            for k in 0..10 {
                if k == ni {
                    next[1][(k + src) % d] += dp[1][src];
                } else if k < ni {
                    next[0][(k + src) % d] += dp[1][src];
                }
                next[0][(k + src) % d] += dp[0][src];
            }
        }
        dp = next;
    }

    // 0を引く
    dp[0][0] -= 1;
    writer.ln(dp[0][0] + dp[1][0]);
}
