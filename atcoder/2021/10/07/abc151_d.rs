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
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Deref,
        DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Not, RangeBounds, Rem, RemAssign, Shl,
        ShlAssign, Shr, ShrAssign, Sub, SubAssign,
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

#[derive(Debug)]
pub struct Grid<T> {
    pub h: usize,
    pub w: usize,
    pub max: usize,
    pub map: Vec<T>,
}

impl<T: Clone> Grid<T> {
    pub fn new(h: usize, w: usize, input: Vec<Vec<T>>) -> Grid<T> {
        let mut map = Vec::new();
        for r in input {
            for c in r {
                map.push(c);
            }
        }
        let max = h * w;
        Grid { h, w, max, map }
    }
    pub fn key(&self, x: usize, y: usize) -> usize {
        y * self.w + x
    }
    pub fn xy(&self, k: usize) -> (usize, usize) {
        (self.x(k), self.y(k))
    }
    pub fn x(&self, k: usize) -> usize {
        k % self.w
    }
    pub fn y(&self, k: usize) -> usize {
        k / self.w
    }
    pub fn get(&self, key: usize) -> &T {
        &self.map[key]
    }
    pub fn set(&mut self, key: usize, value: T) {
        self.map[key] = value;
    }
    pub fn neighbor(&self, key: usize) -> Vec<usize> {
        let mut ret = self.one_way(key);
        if self.x(key) > 0 {
            ret.push(key - 1);
        }
        if self.y(key) > 0 {
            ret.push(key - self.w);
        }
        ret
    }
    pub fn one_way(&self, key: usize) -> Vec<usize> {
        let mut ret = Vec::new();
        if self.x(key) + 1 < self.w {
            ret.push(key + 1);
        }
        if self.y(key) + 1 < self.h {
            ret.push(key + self.w);
        }
        ret
    }
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut output: Writer<W>) {
    let (h, w) = reader.u2();
    let s = reader.bool_map(h, '#');

    let grid = Grid::new(h, w, s);
    let mut ans = 0;
    for start in 0..h * w {
        if !grid.get(start) {
            continue;
        }
        let mut dist = vec![1i64 << 60; h * w];
        dist[start] = 0;
        let mut q = VecDeque::new();
        q.push_front(start);
        while let Some(src) = q.pop_front() {
            for dst in grid.neighbor(src) {
                if !grid.get(dst) {
                    continue;
                }
                if chmin!(dist[dst], dist[src] + 1) {
                    q.push_back(dst);
                }
            }
        }
        for i in 0..h * w {
            if dist[i] < 1 << 60 {
                chmax!(ans, dist[i]);
            }
        }
    }
    output.println(&ans);
}
