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
        Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, RangeBounds,
        Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
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
pub struct Reader<R> {
    reader: R,
    buf: VecDeque<String>,
}
impl<R: Read> Iterator for Reader<R> {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            let mut buf = Vec::new();
            self.reader.read_to_end(&mut buf).unwrap();
            let s = from_utf8(&buf).expect("Not UTF-8 format input.");
            s.split_whitespace()
                .map(ToString::to_string)
                .for_each(|s| self.buf.push_back(s));
        }
        self.buf.pop_front()
    }
}
impl<R: Read> Reader<R> {
    pub fn new(reader: R) -> Reader<R> {
        Reader {
            reader,
            buf: VecDeque::new(),
        }
    }
    pub fn val<T: FromStr>(&mut self) -> T {
        self.next()
            .map(|token| token.parse().ok().expect("Failed to parse."))
            .expect("Insufficient input.")
    }
    pub fn val2<T1: FromStr, T2: FromStr>(&mut self) -> (T1, T2) {
        (self.val(), self.val())
    }
    pub fn val3<T1: FromStr, T2: FromStr, T3: FromStr>(&mut self) -> (T1, T2, T3) {
        (self.val(), self.val(), self.val())
    }
    pub fn val4<T1: FromStr, T2: FromStr, T3: FromStr, T4: FromStr>(&mut self) -> (T1, T2, T3, T4) {
        (self.val(), self.val(), self.val(), self.val())
    }
    pub fn val5<T1: FromStr, T2: FromStr, T3: FromStr, T4: FromStr, T5: FromStr>(
        &mut self,
    ) -> (T1, T2, T3, T4, T5) {
        (self.val(), self.val(), self.val(), self.val(), self.val())
    }
    pub fn vec<T: FromStr>(&mut self, length: usize) -> Vec<T> {
        (0..length).map(|_| self.val()).collect()
    }
    pub fn vec2<T1: FromStr, T2: FromStr>(&mut self, length: usize) -> Vec<(T1, T2)> {
        (0..length).map(|_| self.val2()).collect()
    }
    pub fn vec3<T1: FromStr, T2: FromStr, T3: FromStr>(
        &mut self,
        length: usize,
    ) -> Vec<(T1, T2, T3)> {
        (0..length).map(|_| self.val3()).collect()
    }
    pub fn vec4<T1: FromStr, T2: FromStr, T3: FromStr, T4: FromStr>(
        &mut self,
        length: usize,
    ) -> Vec<(T1, T2, T3, T4)> {
        (0..length).map(|_| self.val4()).collect()
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
pub type Mi = ModInt<Mod998244353>;
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
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}
pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let (n, m): (usize, usize) = reader.val2();
    let mut dp = vec![vec![mi(0); n + 1]; n + 1];
    dp[0][0] = mi(1);
    for i in 1..=n {
        for j in 1..=n {
            dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j] * (j - min(j, (i - 1) / m)) as i64;
        }
    }
    for i in 1..=n {
        writer.println(dp[n][i]);
    }
}
