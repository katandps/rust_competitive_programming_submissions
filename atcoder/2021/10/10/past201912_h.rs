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

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let n = reader.u();
    let mut c = reader.uv(n);
    let q = reader.u();

    let mut odd_min = 1 << 60;
    for i in 0..n {
        if i % 2 == 1 {
            continue;
        }
        chmin!(odd_min, c[i]);
    }
    let mut all_min = *c.iter().min().unwrap();

    let mut ans = 0;
    let mut all = 0;
    let mut odd = 0;
    for _ in 0..q {
        match reader.u() {
            1 => {
                let (x, a) = reader.u2();
                if x % 2 == 1 {
                    if c[x - 1] >= odd + all + a {
                        c[x - 1] -= a;
                        ans += a;
                        chmin!(odd_min, c[x - 1]);
                        chmin!(all_min, c[x - 1]);
                    }
                } else {
                    if c[x - 1] >= all + a {
                        c[x - 1] -= a;
                        ans += a;
                        chmin!(all_min, c[x - 1]);
                    }
                }
            }
            2 => {
                let a = reader.u();
                if odd_min >= a {
                    odd_min -= a;
                    odd += a;
                    ans += (n + 1) / 2 * a;
                    chmin!(all_min, odd_min);
                }
            }
            3 => {
                let a = reader.u();
                if all_min >= a {
                    all += a;
                    all_min -= a;
                    odd_min -= a;
                    ans += n * a;
                }
            }
            _ => unreachable!(),
        }
    }
    writer.println(&ans);
}
