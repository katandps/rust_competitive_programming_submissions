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
impl<R: BufRead, F: Fn() -> R> Iterator for Reader<F> {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            let (mut reader, mut buf) = ((self.init)(), String::new());
            reader.read_line(&mut buf).unwrap();
            self.buf = buf.split_whitespace().map(ToString::to_string).collect();
        }
        self.buf.pop_front()
    }
}
impl<R: BufRead, F: Fn() -> R> Reader<F> {
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
pub struct Writer<F> {
    init: F,
}
impl<W: Write, F: Fn() -> W> Writer<F> {
    pub fn new(init: F) -> Self {
        Self { init }
    }
    pub fn ln<S: Display>(&mut self, s: S) {
        let mut writer = (self.init)();
        writeln!(writer, "{}", s).expect("Failed to write.");
        let _ = writer.flush();
    }
    pub fn out<S: Display>(&mut self, s: S) {
        let mut writer = (self.init)();
        write!(writer, "{}", s).expect("Failed to write.");
        let _ = writer.flush();
    }
    pub fn join<S: Display>(&mut self, v: &[S], separator: &str) {
        let mut writer = (self.init)();
        v.iter().fold("", |sep, arg| {
            write!(writer, "{}{}", sep, arg).expect("Failed to write.");
            separator
        });
        writeln!(writer).expect("Failed to write.");
        let _ = writer.flush();
    }
}
pub fn main() {
    let stdout = stdout();
    let stdin = stdin();
    solve(
        Reader::new(|| stdin.lock()),
        Writer::new(|| BufWriter::new(stdout.lock())),
    );
}

pub fn solve<R: BufRead, W: Write, FR: Fn() -> R, FW: Fn() -> W>(
    mut reader: Reader<FR>,
    mut writer: Writer<FW>,
) {
    let t: usize = reader.v();
    for _ in 0..t {
        let n: usize = reader.v();
        let mut is_imposter = vec![None; n];
        let mut p = vec![0; n / 3];
        for i in 0..n / 3 {
            writer.ln(format!("? {} {} {}", 3 * i + 1, 3 * i + 2, 3 * i + 3));
            p[i] = reader.v();
        }
        let (mut a, mut b) = (0, 0);
        for i in 1..p.len() {
            if p[i - 1] != p[i] {
                let p1 = p[i - 1];
                writer.ln(format!("? {} {} {}", 3 * i - 1, 3 * i, 3 * i + 1));
                let p2 = reader.v();
                writer.ln(format!("? {} {} {}", 3 * i, 3 * i + 1, 3 * i + 2));
                let p3 = reader.v();
                let p4 = p[i];

                if p1 != p2 {
                    a = i * 3 - 3;
                    b = i * 3;
                    if p1 != 0 {
                        swap(&mut a, &mut b);
                    }
                } else if p2 != p3 {
                    a = i * 3 - 2;
                    b = i * 3 + 1;
                    if p2 != 0 {
                        swap(&mut a, &mut b);
                    }
                } else if p3 != p4 {
                    a = i * 3 - 1;
                    b = i * 3 + 2;
                    if p3 != 0 {
                        swap(&mut a, &mut b);
                    }
                }

                is_imposter[a] = Some(true);
                is_imposter[b] = Some(false);
                break;
            }
        }
        for i in 0..n / 3 {
            let k = if p[i] == 0 { b } else { a };
            writer.ln(format!("? {} {} {}", 3 * i + 1, 3 * i + 2, k + 1));
            let l = reader.v();
            writer.ln(format!("? {} {} {}", 3 * i + 2, 3 * i + 3, k + 1));
            let r = reader.v();
            let (a, b, c) = match (p[i], l, r) {
                (0, 0, 0) => (true, true, true),
                (0, 1, 0) => (false, true, true),
                (0, 0, 1) => (true, true, false),
                (0, 1, 1) => (true, false, true),
                (1, 0, 0) => (false, true, false),
                (1, 0, 1) => (true, false, false),
                (1, 1, 0) => (false, false, true),
                (1, 1, 1) => (false, false, false),
                _ => unreachable!(),
            };
            is_imposter[3 * i] = Some(a);
            is_imposter[3 * i + 1] = Some(b);
            is_imposter[3 * i + 2] = Some(c);
        }
        let mut ans = Vec::new();
        for i in 0..n {
            if is_imposter[i] == Some(true) {
                ans.push(i + 1)
            }
        }
        writer.out(format!("! {}", ans.len()));
        for a in ans {
            writer.out(format!(" {}", a));
        }
        writer.ln("");
    }
}
