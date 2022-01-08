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
    let n: usize = reader.v();
    let k: usize = reader.v();
    let p = reader.vec::<u64>(n);
    let mut trie = BinaryTrie::default();
    for i in 0..k - 1 {
        trie.insert(p[i]);
    }
    for i in k..=n {
        trie.insert(p[i - 1]);
        writer.ln(trie.nth(i - k));
    }
}

type TrieValue = u64;
#[derive(Clone, Default)]
pub struct BinaryTrie {
    root: TrieNode,
    xor_val: u64,
}
impl BinaryTrie {
    pub const BIT_LEN: i64 = 60;
    pub fn size(&self) -> usize {
        self.root.count
    }
    pub fn insert(&mut self, v: u64) {
        self.root.add(v, Self::BIT_LEN - 1);
    }
    pub fn erase(&mut self, v: TrieValue) {
        self.root.sub(v, Self::BIT_LEN - 1);
    }
    pub fn min_element(&self) -> TrieValue {
        self.root.get_min(self.xor_val, Self::BIT_LEN - 1)
    }
    pub fn max_element(&self) -> TrieValue {
        self.root.get_min(self.rev_xor_val(), Self::BIT_LEN - 1)
    }
    pub fn nth(&self, k: usize) -> TrieValue {
        assert!(k <= self.size());
        self.root.get(k, Self::BIT_LEN - 1)
    }
    pub fn set_xor_val(&mut self, val: u64) {
        self.xor_val = val
    }
    fn rev_xor_val(&self) -> u64 {
        self.xor_val ^ ((1 << Self::BIT_LEN) - 1)
    }
}
impl Debug for BinaryTrie {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            (0..self.size())
                .map(|i| self.nth(i).to_string())
                .collect::<Vec<_>>()
                .join(" ")
        )
    }
}
#[derive(Clone)]
struct TrieNode {
    count: usize,
    child: Vec<Option<TrieNode>>,
}
impl Default for TrieNode {
    fn default() -> Self {
        Self {
            count: 0,
            child: vec![None, None],
        }
    }
}
impl TrieNode {
    fn add(&mut self, v: TrieValue, b: i64) {
        self.count += 1;
        if b >= 0 {
            let dst = (v >> b & 1) as usize;
            if let Some(node) = self.child[dst].as_mut() {
                node.add(v, b - 1);
            } else {
                let mut node = TrieNode::default();
                node.add(v, b - 1);
                self.child[dst] = Some(node);
            }
        }
    }
    fn sub(&mut self, v: TrieValue, b: i64) {
        self.count -= 1;
        if b >= 0 {
            let dst = (v >> b & 1) as usize;
            self.child[dst].iter_mut().for_each(|c| c.sub(v, b - 1));
        }
    }
    fn get_min(&self, bias: TrieValue, b: i64) -> TrieValue {
        if b < 0 {
            return 0;
        }
        let mut dst = bias >> b & 1;
        if self.child[dst as usize].is_none() {
            dst ^= 1;
        }
        self.child[dst as usize]
            .as_ref()
            .map_or(0, |c| c.get_min(bias, b - 1))
            | (dst << b)
    }
    fn get(&self, k: usize, b: i64) -> TrieValue {
        if b < 0 {
            return 0;
        }
        let m = self.child[0].as_ref().map_or(0, |c| c.count);
        if k < m {
            self.child[0].as_ref().unwrap().get(k, b - 1)
        } else {
            self.child[1].as_ref().map_or(0, |c| c.get(k - m, b - 1)) | (1 << b)
        }
    }
}
