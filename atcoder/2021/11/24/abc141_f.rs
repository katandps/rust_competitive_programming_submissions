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
#[derive(Clone, Eq, PartialEq)]
pub struct BitSet {
    bits: Vec<u64>,
    size: usize,
}
impl BitSet {
    const BLOCK_LEN: usize = 1 << Self::BLOCK_LEN_LEN;
    const BLOCK_LEN_LEN: usize = 6;
    pub fn new(size: usize) -> Self {
        Self {
            bits: vec![0; (size + Self::BLOCK_LEN - 1) / Self::BLOCK_LEN],
            size,
        }
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
    fn chomp(&mut self) {
        let r = self.size & (Self::BLOCK_LEN - 1);
        if r != 0 {
            let d = Self::BLOCK_LEN - r;
            if let Some(x) = self.bits.last_mut() {
                *x = (*x << d) >> d;
            }
        }
    }
    pub fn get_u64(&self) -> u64 {
        (0..min(64, self.size))
            .filter(|i| self[*i])
            .fold(0, |x, i| x + (1 << i))
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
                .map(|i| if self[i] { '1' } else { '0' })
                .collect::<String>()
        )
    }
}
impl Debug for BitSet {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        Display::fmt(self, f)
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
    pub fn gauss_jordan(&mut self, is_extended: bool) -> Vec<usize> {
        let mut rank = 0;
        let mut adopted = Vec::new();
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
                adopted.push(pivot);
                rank += 1;
            }
        }
        adopted
    }
    pub fn linear_equation(&mut self, b: &[bool]) -> Option<(Vec<bool>, usize)> {
        let mut m = BitMatrix::new(self.height, self.width + 1);
        (0..self.height).for_each(|i| {
            (0..self.width).for_each(|j| {
                m.val[i].set(j, self.val[i][j]);
            });
            m.val[i].set(self.width, b[i]);
        });
        let rank = self.gauss_jordan(true).len();
        if !m
            .val
            .iter()
            .skip(rank)
            .filter_map(|bm| if bm[self.width] { Some(()) } else { None })
            .count()
            == 0
        {
            None
        } else {
            Some((
                (0..self.width).map(|i| m.val[i][self.width]).collect(),
                rank,
            ))
        }
    }
}
pub fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}
pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let n: usize = reader.val();
    let a: Vec<u64> = reader.vec(n);
    let mut matrix = BitMatrix::new(n, 60);
    let mut b = 0;
    for i in 0..n {
        b ^= a[i];
    }
    for i in 0..n {
        let ai = a[i];
        for j in 0..60 {
            if b >> j & 1 == 0 && ai >> j & 1 == 1 {
                matrix.val[i].set(j, true);
            }
        }
    }

    let mut max = 0;
    let mut all = 0;
    let _ = matrix.gauss_jordan(false);
    for i in 0..n {
        all ^= matrix.val[i].get_u64();
        max ^= a[i];
    }
    writer.println(all + (all ^ max));
}
