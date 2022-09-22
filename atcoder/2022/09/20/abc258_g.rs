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
    pub fn bits(&mut self, i: i64, len: usize) {
        (0..len).for_each(|b| write!(self.writer, "{}", i >> b & 1).expect("Failed to write."));
        writeln!(self.writer).expect("Failed to write.")
    }
    pub fn flush(&mut self) {
        let _ = self.writer.flush();
    }
}
pub struct Reader<F> {
    init: F,
    buf: VecDeque<String>,
}
impl<R: BufRead, F: FnMut() -> R> Iterator for Reader<F> {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            let mut reader = (self.init)();
            let mut l = String::new();
            reader.read_line(&mut l).unwrap();
            self.buf
                .append(&mut l.split_whitespace().map(ToString::to_string).collect());
        }
        self.buf.pop_front()
    }
}
impl<R: BufRead, F: FnMut() -> R> Reader<F> {
    pub fn new(init: F) -> Self {
        let buf = VecDeque::new();
        Reader { init, buf }
    }
    pub fn v<T: FS>(&mut self) -> T {
        let s = self.next().expect("Insufficient input.");
        s.parse().ok().expect("Failed to parse.")
    }
    pub fn v2<T1: FS, T2: FS>(&mut self) -> (T1, T2) {
        (self.v(), self.v())
    }
    pub fn v3<T1: FS, T2: FS, T3: FS>(&mut self) -> (T1, T2, T3) {
        (self.v(), self.v(), self.v())
    }
    pub fn v4<T1: FS, T2: FS, T3: FS, T4: FS>(&mut self) -> (T1, T2, T3, T4) {
        (self.v(), self.v(), self.v(), self.v())
    }
    pub fn v5<T1: FS, T2: FS, T3: FS, T4: FS, T5: FS>(&mut self) -> (T1, T2, T3, T4, T5) {
        (self.v(), self.v(), self.v(), self.v(), self.v())
    }
    pub fn vec<T: FS>(&mut self, length: usize) -> Vec<T> {
        (0..length).map(|_| self.v()).collect()
    }
    pub fn vec2<T1: FS, T2: FS>(&mut self, length: usize) -> Vec<(T1, T2)> {
        (0..length).map(|_| self.v2()).collect()
    }
    pub fn vec3<T1: FS, T2: FS, T3: FS>(&mut self, length: usize) -> Vec<(T1, T2, T3)> {
        (0..length).map(|_| self.v3()).collect()
    }
    pub fn vec4<T1: FS, T2: FS, T3: FS, T4: FS>(&mut self, length: usize) -> Vec<(T1, T2, T3, T4)> {
        (0..length).map(|_| self.v4()).collect()
    }
    pub fn chars(&mut self) -> Vec<char> {
        self.v::<String>().chars().collect()
    }
    fn split(&mut self, zero: u8) -> Vec<usize> {
        self.v::<String>()
            .chars()
            .map(|c| (c as u8 - zero) as usize)
            .collect()
    }
    pub fn digits(&mut self) -> Vec<usize> {
        self.split(b'0')
    }
    pub fn lowercase(&mut self) -> Vec<usize> {
        self.split(b'a')
    }
    pub fn uppercase(&mut self) -> Vec<usize> {
        self.split(b'A')
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
    pub fn matrix<T: FS>(&mut self, h: usize, w: usize) -> Vec<Vec<T>> {
        (0..h).map(|_| self.vec(w)).collect()
    }
}
pub fn to_lr<T, R: RangeBounds<T>>(range: &R, length: T) -> (T, T)
where
    T: Copy + One + Zero + Add<Output = T> + PartialOrd,
{
    use Bound::{Excluded, Included, Unbounded};
    let l = match range.start_bound() {
        Unbounded => T::zero(),
        Included(&s) => s,
        Excluded(&s) => s + T::one(),
    };
    let r = match range.end_bound() {
        Unbounded => length,
        Included(&e) => e + T::one(),
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
    iter::{repeat, Product, Sum},
    marker::PhantomData,
    mem::swap,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Bound,
        Deref, DerefMut, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Not, Range,
        RangeBounds, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
    },
    str::{from_utf8, FromStr as FS},
};
#[allow(unused_macros)]
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
#[allow(unused_macros)]
macro_rules ! chmin {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_min = min ! ($ ($ cmps ) ,+ ) ; if $ base > cmp_min {$ base = cmp_min ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! min {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ b } else {$ a } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = min ! ($ ($ rest ) ,+ ) ; if $ a > b {b } else {$ a } } } ; }
#[allow(unused_macros)]
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
pub trait Magma {
    type M: Clone + PartialEq + Debug;
    fn op(x: &Self::M, y: &Self::M) -> Self::M;
}
pub trait Associative {}
pub trait Unital: Magma {
    fn unit() -> Self::M;
}
pub trait Commutative: Magma {}
pub trait Invertible: Magma {
    fn inv(x: &Self::M) -> Self::M;
}
pub trait Idempotent: Magma {}
pub trait SemiGroup: Magma + Associative {}
pub trait Monoid: Magma + Associative + Unital {
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
pub trait CommutativeMonoid: Magma + Associative + Unital + Commutative {}
impl<M: Magma + Associative + Unital + Commutative> CommutativeMonoid for M {}
pub trait Group: Magma + Associative + Unital + Invertible {}
impl<M: Magma + Associative + Unital + Invertible> Group for M {}
pub trait AbelianGroup: Magma + Associative + Unital + Commutative + Invertible {}
impl<M: Magma + Associative + Unital + Commutative + Invertible> AbelianGroup for M {}
pub trait Band: Magma + Associative + Idempotent {}
impl<M: Magma + Associative + Idempotent> Band for M {}
pub trait MapMonoid {
    type Mono: Monoid;
    type Func: Monoid;
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
    fn apply(
        &self,
        f: &<Self::Func as Magma>::M,
        value: &<Self::Mono as Magma>::M,
    ) -> <Self::Mono as Magma>::M;
    fn identity_map() -> <Self::Func as Magma>::M {
        Self::Func::unit()
    }
    fn compose(
        &self,
        f: &<Self::Func as Magma>::M,
        g: &<Self::Func as Magma>::M,
    ) -> <Self::Func as Magma>::M {
        Self::Func::op(f, g)
    }
}
pub trait Zero {
    fn zero() -> Self;
}
pub trait One {
    fn one() -> Self;
}
pub trait BoundedBelow {
    fn min_value() -> Self;
}
pub trait BoundedAbove {
    fn max_value() -> Self;
}
# [rustfmt :: skip ] pub trait Integral : 'static + Send + Sync + Copy + Ord + Display + Debug + Add < Output = Self > + Sub < Output = Self > + Mul < Output = Self > + Div < Output = Self > + Rem < Output = Self > + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + Product + BitOr < Output = Self > + BitAnd < Output = Self > + BitXor < Output = Self > + Not < Output = Self > + Shl < Output = Self > + Shr < Output = Self > + BitOrAssign + BitAndAssign + BitXorAssign + ShlAssign + ShrAssign + Zero + One + BoundedBelow + BoundedAbove {}
macro_rules ! impl_integral {($ ($ ty : ty ) ,* ) => {$ (impl Zero for $ ty {fn zero () -> Self {0 } } impl One for $ ty {fn one () -> Self {1 } } impl BoundedBelow for $ ty {fn min_value () -> Self {Self :: min_value () } } impl BoundedAbove for $ ty {fn max_value () -> Self {Self :: max_value () } } impl Integral for $ ty {} ) * } ; }
impl_integral!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
pub fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(|| stdin.lock()), Writer::new(stdout.lock()));
}
pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let n: usize = reader.v();
    let a = reader.bool_map(n, '0');
    let mut bss = Vec::with_capacity(n);
    for i in 0..n {
        let mut bs = BitSet::new(n);
        for j in 0..n {
            bs.set(j, a[i][j]);
        }
        bss.push(bs);
    }
    let mut ans = 0i64;
    for i in 0..n {
        for j in i + 1..n {
            if bss[i][j] {
                ans += (&bss[i] & &bss[j]).count_ones() as i64;
            }
        }
    }
    // dbg!(&ans);
    writer.ln(ans / 3);
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
impl BitAnd for &BitSet {
    type Output = BitSet;
    fn bitand(self, rhs: Self) -> Self::Output {
        assert_eq!(self.size, rhs.size);
        BitSet {
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
                .take(self.size)
                .map(|i| (if self[i] { 1 } else { 0 }).to_string())
                .collect::<String>()
        )
    }
}
