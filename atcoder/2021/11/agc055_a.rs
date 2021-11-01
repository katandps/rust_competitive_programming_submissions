use itertools::Itertools;
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
    mem::swap,
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
macro_rules! min {($a:expr $(,)*) => {{$a}};($a:expr, $b:expr $(,)*) => {{if $a > $b {$b} else {$a}}};($a:expr, $($rest:expr),+ $(,)*) => {{let b = min!($($rest),+);if $a > b {b} else {$a}}};}
#[allow(unused_macros)]
macro_rules! max {($a:expr $(,)*) => {{$a}};($a:expr, $b:expr $(,)*) => {{if $a > $b {$a} else {$b}}};($a:expr, $($rest:expr),+ $(,)*) => {{let b = max!($($rest),+);if $a > b {$a} else {b}}};}

pub fn to_lr<R: RangeBounds<usize>>(range: R, length: usize) -> (usize, usize) {
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

/// stdin reader
pub struct Reader<R> {
    reader: R,
    buf: VecDeque<String>,
}
impl<R: BufRead> Iterator for Reader<R> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            let mut buf = Vec::new();
            self.reader.read_to_end(&mut buf).unwrap();
            let s = from_utf8(&buf).expect("utf8でない文字列が入力されました.");
            s.split_whitespace()
                .map(ToString::to_string)
                .for_each(|s| self.buf.push_back(s));
        }
        self.buf.pop_front()
    }
}
impl<R: BufRead> Reader<R> {
    pub fn new(reader: R) -> Reader<R> {
        Reader {
            reader,
            buf: VecDeque::new(),
        }
    }
    pub fn val<T: FromStr>(&mut self) -> T {
        self.next()
            .map(|token| token.parse().ok().expect("型変換エラー"))
            .expect("入力が足りません")
    }
    pub fn vec<T: FromStr>(&mut self, length: usize) -> Vec<T> {
        (0..length).map(|_| self.val()).collect()
    }
    pub fn chars(&mut self) -> Vec<char> {
        self.val::<String>().chars().collect()
    }
    pub fn digits(&mut self) -> Vec<i64> {
        self.val::<String>()
            .chars()
            .map(|c| (c as u8 - b'A') as i64)
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

/// stdin writer
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

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut writer: Writer<W>) {
    let n = reader.val::<usize>();
    let s = reader.digits();

    let mut l = vec![Vec::new(); 3];
    let mut r = vec![Vec::new(); 3];
    for i in 0..n {
        l[s[i] as usize].push(i);
        r[s[i + 2 * n] as usize].push(i + 2 * n);
    }
    let mut ans = vec![0; n * 3];
    let mut map = HashMap::new();
    map.insert((0, 1), 1);
    map.insert((1, 0), 2);
    map.insert((0, 2), 3);
    map.insert((2, 0), 4);
    map.insert((1, 2), 5);
    map.insert((2, 1), 6);
    let mut cnt = vec![0; 7];
    'total: for lp in (0..3).permutations(3) {
        let mut left = Vec::new();
        for &i in &lp {
            for index in &l[i] {
                left.push((i, index));
            }
        }
        'case: for rp in (0..3).permutations(3) {
            let mut right = Vec::new();
            for &i in &rp {
                for index in &r[i] {
                    right.push((i, index));
                }
            }

            for i in 0..left.len() {
                if left[i].0 == right[i].0 {
                    continue 'case;
                }
            }
            for i in 0..n {
                let (l, li) = left[i];
                let (r, ri) = right[i];
                let p = *map.get(&(l, r)).unwrap();
                ans[*li] = p;
                ans[*ri] = p;
                cnt[p] += 1;
            }
            break 'total;
        }
    }
    for i in 0..n {
        match s[i + n] {
            0 => {
                if cnt[5] > 0 {
                    cnt[5] -= 1;
                    ans[i + n] = 5;
                } else {
                    cnt[6] -= 1;
                    ans[i + n] = 6;
                }
            }
            1 => {
                if cnt[3] > 0 {
                    cnt[3] -= 1;
                    ans[i + n] = 3;
                } else {
                    cnt[4] -= 1;
                    ans[i + n] = 4;
                }
            }
            2 => {
                if cnt[1] > 0 {
                    cnt[1] -= 1;
                    ans[i + n] = 1;
                } else {
                    cnt[2] -= 1;
                    ans[i + n] = 2;
                }
            }
            _ => unreachable!(),
        }
    }
    writer.print_join(&ans, "");
}
