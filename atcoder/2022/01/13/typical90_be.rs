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
pub fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(|| stdin.lock()), Writer::new(stdout.lock()));
}
pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let (n, m) = reader.v2::<usize, usize>();
    let mut bm = BitMatrix::new(n, m);
    for i in 0..n {
        let k = reader.v::<usize>();
        for _ in 0..k {
            bm.val[i].set(reader.v::<usize>() - 1, true);
        }
    }
    let mut s = reader.vec::<i64>(m);
    bm.elimination(false);
    let mut j = 0;
    for i in (0..m).rev() {
        if s[i] == 0 {
            continue;
        }
        if s[i] == 1 {
            while j < n && !bm.val[j][i] {
                j += 1;
            }
            if j >= n {
                return writer.ln(0);
            }
            for k in 0..m {
                if bm.val[j][k] {
                    s[k] *= -1;
                    s[k] += 1;
                }
            }
            j += 1;
        }
    }
    if s.iter().filter(|si| **si == 1).count() > 0 {
        return writer.ln(0);
    }
    while j < n && bm.val[j].count_ones() > 0 {
        j += 1;
    }
    writer.ln(mi(2).pow((n - j) as i64));
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

#[derive(Clone, Debug)]
pub struct BitMatrix {
    height: usize,
    width: usize,
    val: Vec<BitSet>,
}
impl BitMatrix {
    pub fn new(height: usize, width: usize) -> BitMatrix {
        let val = vec![BitSet::new(width); height];
        BitMatrix { height, width, val }
    }
    pub fn elimination(&mut self, is_extended: bool) -> usize {
        let mut rank = 0;
        for col in (0..self.width - if is_extended { 1 } else { 0 }).rev() {
            let mut pivot = None;
            for row in rank..self.height {
                if self.val[row][col] {
                    pivot = Some(row);
                    break;
                }
            }
            if let Some(pivot) = pivot {
                self.val.swap(pivot, rank);
                for row in 0..self.height {
                    if row != rank && self.val[row][col] {
                        self.val[row] = self.val[row].clone() ^ self.val[rank].clone();
                    }
                }
                rank += 1;
            }
        }
        rank
    }
    pub fn linear_equation(&mut self, b: &[bool]) -> Option<(Vec<bool>, usize)> {
        let mut m = BitMatrix::new(self.height, self.width + 1);
        (0..self.height).for_each(|i| {
            (0..self.width).for_each(|j| {
                m.val[i].set(j, self.val[i][j]);
            });
            m.val[i].set(self.width, b[i]);
        });
        let rank = self.elimination(true);
        if m.val.iter().skip(rank).filter(|bm| bm[self.width]).count() == 0 {
            Some((
                (0..self.width).map(|i| m.val[i][self.width]).collect(),
                rank,
            ))
        } else {
            None
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct BitSet {
    bits: Vec<u64>,
    size: usize,
}
impl BitSet {
    const BLOCK_LEN: usize = 1 << Self::BLOCK_LEN_LEN;
    const BLOCK_LEN_LEN: usize = 6;
    pub fn new(size: usize) -> Self {
        let size = max(size, 64);
        let bits = vec![0; (size + Self::BLOCK_LEN - 1) / Self::BLOCK_LEN];
        Self { bits, size }
    }
    pub fn set(&mut self, index: usize, b: bool) {
        assert!(index < self.size);
        if b {
            self.bits[index >> Self::BLOCK_LEN_LEN] |= 1 << (index & (Self::BLOCK_LEN - 1));
        } else {
            self.bits[index >> Self::BLOCK_LEN_LEN] &= !(1 << (index & (Self::BLOCK_LEN - 1)));
        }
    }
    pub fn count_ones(&self) -> u32 {
        self.bits.iter().map(|b| b.count_ones()).sum()
    }
    pub fn get_u64(&self) -> u64 {
        self.bits[0]
    }
    fn chomp(&mut self) {
        let r = self.size & (Self::BLOCK_LEN - 1);
        if r != 0 {
            let d = Self::BLOCK_LEN - r;
            if let Some(x) = self.bits.last_mut() {
                *x = (*x << d) >> d;
            }
        }
    }
}
impl Debug for BitSet {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}
impl Index<usize> for BitSet {
    type Output = bool;
    fn index(&self, index: usize) -> &bool {
        assert!(index < self.size);
        &[false, true][((self.bits[index >> Self::BLOCK_LEN_LEN]
            >> (index & (Self::BLOCK_LEN - 1)))
            & 1) as usize]
    }
}
impl BitAnd for BitSet {
    type Output = BitSet;
    fn bitand(self, rhs: Self) -> Self::Output {
        assert_eq!(self.size, rhs.size);
        Self {
            bits: (0..self.bits.len())
                .map(|i| self.bits[i] & rhs.bits[i])
                .collect(),
            size: self.size,
        }
    }
}
impl BitOr for BitSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.size, rhs.size);
        Self {
            bits: (0..self.bits.len())
                .map(|i| self.bits[i] | rhs.bits[i])
                .collect(),
            size: self.size,
        }
    }
}
impl BitXor for BitSet {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.size, rhs.size);
        Self {
            bits: (0..self.bits.len())
                .map(|i| self.bits[i] ^ rhs.bits[i])
                .collect(),
            size: self.size,
        }
    }
}
impl ShlAssign<usize> for BitSet {
    fn shl_assign(&mut self, rhs: usize) {
        *self = self.clone() << rhs;
    }
}
impl Shl<usize> for BitSet {
    type Output = Self;
    fn shl(mut self, rhs: usize) -> Self::Output {
        if rhs >= self.size {
            self.bits.iter_mut().for_each(|b| *b = 0);
            return self;
        }
        let block = rhs >> Self::BLOCK_LEN_LEN;
        let inner = rhs & (Self::BLOCK_LEN - 1);
        if inner == 0 {
            (block..self.bits.len())
                .rev()
                .for_each(|i| self.bits[i] = self.bits[i - block])
        } else {
            (block + 1..self.bits.len()).rev().for_each(|i| {
                self.bits[i] = (self.bits[i - block] << inner)
                    | (self.bits[i - block - 1] >> (Self::BLOCK_LEN - inner))
            });
            self.bits[block] = self.bits[0] << inner;
        }
        self.bits[..block].iter_mut().for_each(|b| *b = 0);
        self.chomp();
        self
    }
}
impl ShrAssign<usize> for BitSet {
    fn shr_assign(&mut self, rhs: usize) {
        *self = self.clone() >> rhs;
    }
}
impl Shr<usize> for BitSet {
    type Output = Self;
    fn shr(mut self, rhs: usize) -> Self::Output {
        if rhs >= self.size {
            self.bits.iter_mut().for_each(|b| *b = 0);
            return self;
        }
        let block = rhs >> Self::BLOCK_LEN_LEN;
        let inner = rhs & (Self::BLOCK_LEN - 1);
        let len = self.bits.len();
        if inner == 0 {
            (0..len - block).for_each(|i| self.bits[i] = self.bits[i + block])
        } else {
            (0..len - block - 1).for_each(|i| {
                self.bits[i] = (self.bits[i + block] >> inner)
                    | (self.bits[i + block + 1] << (Self::BLOCK_LEN - inner))
            });
            self.bits[len - block - 1] = self.bits[len - 1] >> inner;
        }
        self.bits[len - block..].iter_mut().for_each(|b| *b = 0);
        self
    }
}
impl Not for BitSet {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self {
            bits: self.bits.iter().map(|&i| i ^ std::u64::MAX).collect(),
            size: self.size,
        }
    }
}
impl Display for BitSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            (0..self.size)
                .rev()
                .map(|i| (if self[i] { 1 } else { 0 }).to_string())
                .collect::<String>()
        )
    }
}
