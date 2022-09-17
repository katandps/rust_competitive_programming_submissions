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

#[derive(Clone, Eq, PartialEq)]
pub struct Matrix<T>(Vec<Vec<T>>);
mod matrix_impl {
    use super::{
        Add, AddAssign, Debug, Div, Formatter, Matrix, Mul, MulAssign, Neg, One, Sub, SubAssign,
        Zero,
    };
    impl<T> std::convert::TryFrom<Vec<Vec<T>>> for Matrix<T> {
        type Error = &'static str;
        fn try_from(buf: Vec<Vec<T>>) -> std::result::Result<Self, Self::Error> {
            if (1..buf.len()).any(|i| buf[0].len() != buf[i].len()) {
                Err("size is invalid")
            } else {
                Ok(Self(buf))
            }
        }
    }
    impl<T> Matrix<T> {
        fn size(&self) -> (usize, usize) {
            if self.0.is_empty() {
                (0, 0)
            } else {
                (self.0.len(), self.0[0].len())
            }
        }
    }
    pub trait ZeroMatrix {
        fn zero_matrix(x: usize, y: usize) -> Self;
    }
    impl<T: Clone + Zero> ZeroMatrix for Matrix<T> {
        fn zero_matrix(x: usize, y: usize) -> Self {
            Self(vec![vec![T::zero(); x]; y])
        }
    }
    pub trait IdentityMatrix {
        fn identity_matrix(n: usize) -> Self;
    }
    impl<T: Clone + Zero + One> IdentityMatrix for Matrix<T> {
        fn identity_matrix(n: usize) -> Self {
            let mut ret = Self::zero_matrix(n, n);
            (0..n).for_each(|i| ret.0[i][i] = T::one());
            ret
        }
    }
    pub trait RowVector<T> {
        fn row_vector(v: &[T]) -> Self;
    }
    impl<T: Clone> RowVector<T> for Matrix<T> {
        fn row_vector(v: &[T]) -> Self {
            Self(vec![v.to_vec()])
        }
    }
    pub trait ColumnVector<T> {
        fn column_vector(v: &[T]) -> Self;
    }
    impl<T: Clone> ColumnVector<T> for Matrix<T> {
        fn column_vector(v: &[T]) -> Self {
            Self(v.iter().map(|cell| vec![cell.clone()]).collect())
        }
    }
    pub trait Determinant<T> {
        fn determinant(&self) -> Option<T>;
    }
    impl<
            T: Clone
                + Zero
                + One
                + MulAssign
                + Mul<Output = T>
                + SubAssign
                + Div<Output = T>
                + PartialEq,
        > Determinant<T> for Matrix<T>
    {
        fn determinant(&self) -> Option<T> {
            let (n, m) = self.size();
            if n != m {
                return None;
            }
            if n == 0 {
                return Some(T::zero());
            }
            let zero = T::zero();
            let mut res = T::one();
            let mut buf = self.0.clone();
            for i in 0..n {
                match (i..n).find(|&ni| buf[ni][i] != zero) {
                    Some(ni) => buf.swap(i, ni),
                    None => return Some(zero),
                }
                res *= buf[i][i].clone();
                let diag = T::one() / buf[i][i].clone();
                (i..n).for_each(|j| buf[i][j] *= diag.clone());
                for ni in (0..n).filter(|&ni| ni != i) {
                    let c = buf[ni][i].clone();
                    for j in i..n {
                        let d = c.clone() * buf[i][j].clone();
                        buf[ni][j] -= d;
                    }
                }
            }
            Some(res)
        }
    }
    pub trait Pow {
        fn pow(self, e: i64) -> Option<Self>
        where
            Self: Sized;
    }
    impl<T: Clone + Zero + One + Mul<Output = T> + Add<Output = T>> Pow for Matrix<T> {
        fn pow(mut self, mut e: i64) -> Option<Self> {
            let (n, m) = self.size();
            if n != m {
                return None;
            }
            let mut result = Self::identity_matrix(n);
            while e > 0 {
                if e & 1 == 1 {
                    result = (result * self.clone()).unwrap();
                }
                e >>= 1;
                self = (self.clone() * self).unwrap();
            }
            Some(result)
        }
    }
    impl<T: Default + Clone + Zero + One> Matrix<T> {
        pub fn sub_matrix(&self, x: usize, y: usize) -> Self {
            let (n, m) = self.size();
            let mut buf = vec![vec![T::default(); m - 1]; n - 1];
            for yi in (0..n).filter(|&yi| yi != y) {
                for xi in (0..m).filter(|&xi| xi != x) {
                    buf[yi - if yi < y { 0 } else { 1 }][xi - if xi < x { 0 } else { 1 }] =
                        self.0[yi][xi].clone();
                }
            }
            Matrix(buf)
        }
    }
    impl<T: AddAssign + Clone> Add<Matrix<T>> for Matrix<T> {
        type Output = Self;
        fn add(mut self, rhs: Self) -> Self {
            for i in 0..self.0.len() {
                for j in 0..self.0[0].len() {
                    self.0[i][j] += rhs.0[i][j].clone()
                }
            }
            self
        }
    }
    impl<T: AddAssign + Clone> AddAssign<T> for Matrix<T> {
        fn add_assign(&mut self, rhs: T) {
            self.0
                .iter_mut()
                .for_each(|row| row.iter_mut().for_each(|cell| *cell += rhs.clone()))
        }
    }
    impl<T: Neg<Output = T> + Clone> Neg for Matrix<T> {
        type Output = Self;
        fn neg(mut self) -> Self {
            for i in 0..self.0.len() {
                for j in 0..self.0[0].len() {
                    self.0[i][j] = -self.0[i][j].clone()
                }
            }
            self
        }
    }
    impl<T: Neg<Output = T> + AddAssign + Clone> Sub<Matrix<T>> for Matrix<T> {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            self + (-rhs)
        }
    }
    impl<T: MulAssign<i64>> MulAssign<i64> for Matrix<T> {
        fn mul_assign(&mut self, rhs: i64) {
            self.0
                .iter_mut()
                .for_each(|row| row.iter_mut().for_each(|cell| *cell *= rhs))
        }
    }
    impl<T: MulAssign<i64>> Mul<i64> for Matrix<T> {
        type Output = Self;
        fn mul(mut self, rhs: i64) -> Self {
            self *= rhs;
            self
        }
    }
    impl<T: Mul<Output = T> + Add<Output = T> + Zero + Clone> Mul<Matrix<T>> for Matrix<T> {
        type Output = Option<Self>;
        fn mul(self, rhs: Self) -> Option<Self> {
            let ((self_y, self_x), (rhs_y, rhs_x)) = (self.size(), rhs.size());
            if self_x != rhs_y {
                return None;
            }
            let mut ret = Self::zero_matrix(rhs_x, self_y);
            ret.0.iter_mut().enumerate().for_each(|(i, bufi)| {
                bufi.iter_mut().enumerate().for_each(|(j, bufij)| {
                    *bufij = (0..self_x)
                        .map(|k| self.0[i][k].clone() * rhs.0[k][j].clone())
                        .fold(T::zero(), |x, a| x + a);
                });
            });
            Some(ret)
        }
    }

    impl<T: ToString> Debug for Matrix<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{}",
                self.0
                    .iter()
                    .map(|row| row
                        .iter()
                        .map(|mi| mi.to_string())
                        .collect::<Vec<_>>()
                        .join(" "))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        }
    }
}
pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let (k, m) = reader.v2::<usize, usize>();
    let a = reader.vec::<i64>(k);
    let c = reader.vec::<i64>(k);
    if m < k {
        return writer.ln(a[m - 1]);
    }

    let mut matrix = Matrix(vec![vec![P(0); k]; k]);
    for i in 0..k {
        matrix.0[0][i] = P(c[i]);
    }
    for i in 1..k {
        matrix.0[i][i - 1] = P((1 << 32) - 1);
    }
    let mat = matrix_impl::Pow::pow(matrix, (m - k) as i64).unwrap();
    let mut ans = 0;
    for i in 0..k {
        ans ^= a[k - 1 - i] & mat.0[0][i].0;
    }
    writer.ln(ans);
}

#[derive(Clone, Debug, PartialEq)]
struct P(i64);

impl ToString for P {
    fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Add for P {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Mul for P {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl Zero for P {
    fn zero() -> Self {
        P(0)
    }
}
impl One for P {
    fn one() -> Self {
        P((1 << 32) - 1)
    }
}
