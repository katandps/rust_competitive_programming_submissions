use crate::graph::{Graph, INF};
pub use reader::*;
#[allow(unused_imports)]
use {
    reader::Reader,
    std::convert::TryInto,
    std::{cmp::*, collections::*, io::*, num::*, str::*},
    writer::Writer,
};

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

#[allow(dead_code)]
pub mod reader {
    #[allow(unused_imports)]
    use std::{fmt::Debug, io::*, str::*};

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
}

pub mod writer {
    use itertools::Itertools;
    use std::fmt::Display;
    use std::io::{BufWriter, Write};

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

        pub fn join_space<S: Display>(&mut self, v: &[S]) {
            writeln!(self.w, "{}", v.iter().join(" ")).unwrap()
        }

        pub fn join_newline<S: Display>(&mut self, v: &[S]) {
            writeln!(self.w, "{}", v.iter().join("\n")).unwrap()
        }
    }
}

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(stdin.lock()), Writer::new(stdout.lock()));
}

#[allow(dead_code)]
pub mod graph {
    use std::cmp::Ordering;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::fmt::{Debug, Formatter};

    pub type Weight = i64;

    pub const INF: Weight = 1 << 60;

    #[derive(Copy, Clone)]
    pub struct Edge {
        pub src: usize,
        pub dst: usize,
        pub weight: Weight,
    }

    impl Debug for Edge {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} -> {} : {}", self.src, self.dst, self.weight)
        }
    }

    impl Edge {
        pub fn default() -> Edge {
            let (src, dst, weight) = (0, 0, 0);
            Edge { src, dst, weight }
        }

        pub fn new(src: usize, dst: usize, weight: Weight) -> Edge {
            Edge { src, dst, weight }
        }
    }

    impl std::cmp::PartialEq for Edge {
        fn eq(&self, other: &Self) -> bool {
            self.weight.eq(&other.weight)
        }
    }

    impl std::cmp::Eq for Edge {}

    impl std::cmp::PartialOrd for Edge {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.weight.partial_cmp(&other.weight)
        }
    }

    impl std::cmp::Ord for Edge {
        fn cmp(&self, other: &Self) -> Ordering {
            self.weight.cmp(&other.weight)
        }
    }

    /// 辺の情報を使用してグラフの問題を解くためのライブラリ
    #[derive(Clone, Debug)]
    pub struct Graph {
        pub n: usize,
        pub edges: Vec<Vec<Edge>>,
        pub rev_edges: Vec<Vec<Edge>>,
    }

    impl Graph {
        /// n: 頂点数
        pub fn new(n: usize) -> Self {
            Self {
                n,
                edges: vec![Vec::new(); n],
                rev_edges: vec![Vec::new(); n],
            }
        }

        /// 辺行列からグラフを生成する O(N^2)
        pub fn from_matrix(weights: &[Vec<Weight>], n: usize) -> Graph {
            let mut ret = Self::new(n);
            for i in 0..n {
                for j in i + 1..n {
                    if weights[i][j] == -1 {
                        continue;
                    }
                    ret.add_edge(i, j, weights[i as usize][j as usize]);
                    ret.add_edge(j, i, weights[j as usize][i as usize]);
                }
            }
            ret
        }

        /// 相互に行き来できる辺をつける
        pub fn add_edge(&mut self, a: usize, b: usize, w: Weight) {
            self.edges[a].push(Edge::new(a, b, w));
            self.edges[b].push(Edge::new(b, a, w));
            self.rev_edges[a].push(Edge::new(a, b, w));
            self.rev_edges[b].push(Edge::new(b, a, w));
        }

        /// 1方向にのみ移動できる辺をつける
        pub fn add_arc(&mut self, a: usize, b: usize, w: Weight) {
            self.edges[a].push(Edge::new(a, b, w));
            self.rev_edges[b].push(Edge::new(b, a, w));
        }

        ///
        /// Prim法でMinimumSpanningTree(最小全域木)を求める
        /// rから開始する (= rと連結でない点は無視する)
        /// ## 計算量
        /// 頂点数をV、辺数をEとすると
        /// 二分ヒープによる実装なのでO(ElogV)
        /// ```
        /// use library::graph::graph::Graph;
        /// let data = vec![
        ///     vec![-1, 2, 3, 1, -1],
        ///     vec![2, -1, -1, 4, -1],
        ///     vec![3, -1, -1, 1, 1],
        ///     vec![1, 4, 1, -1, 3],
        ///     vec![-1, -1, 1, 3, -1],
        /// ];
        ///
        /// let graph = Graph::from_matrix(&data, 5);
        /// assert_eq!(5, graph.prim(0));
        /// ```
        ///
        pub fn prim(&self, r: usize) -> Weight {
            let mut t = Vec::new();
            let mut total: Weight = 0;
            let mut visits = vec![false; self.n];
            let mut q = BinaryHeap::new();
            q.push(Reverse(Edge::new(self.n, r, 0)));
            while !q.is_empty() {
                let Reverse(e) = q.pop().unwrap();
                if visits[e.dst as usize] {
                    continue;
                }
                visits[e.dst as usize] = true;
                total += e.weight;
                if e.src != self.n {
                    t.push(e)
                }
                self.edges[e.dst].iter().for_each(|f| {
                    if !visits[f.dst as usize] {
                        q.push(Reverse(*f));
                    }
                });
            }
            total
        }

        ///
        ///  ベルマンフォード法でlからrへの最小コストを求める
        /// ## 計算量
        ///  O(NM)
        pub fn bellman_ford(&self, l: usize, r: usize) -> Weight {
            let mut dist = vec![INF; self.n];
            dist[l] = 0;
            for _step1 in 1..self.n {
                for src in 0..self.n {
                    if dist[src] != INF {
                        self.edges[src].iter().for_each(|e| {
                            let _ = chmin!(dist[e.dst], dist[src] + e.weight);
                        });
                    }
                }
            }
            let mut neg = vec![false; self.n];
            for _step2 in 0..self.n {
                for src in 0..self.n {
                    if dist[src] != INF {
                        self.edges[src].iter().for_each(|e| {
                            neg[e.dst] |= neg[src] | chmin!(dist[e.dst], dist[src] + e.weight)
                        });
                    }
                }
            }
            if neg[r] {
                INF
            } else {
                dist[r]
            }
        }

        ///
        /// dijkstra法でlから各頂点への最小コストを求める
        /// 負辺がある場合は使えない
        /// ## 計算量
        ///  O(NlogN)
        pub fn dijkstra(&self, l: usize) -> Vec<Weight> {
            let mut dist = vec![INF; self.n];
            let mut heap = BinaryHeap::new();
            dist[l] = 0;
            heap.push((Reverse(0), l));
            while let Some((Reverse(d), src)) = heap.pop() {
                if dist[src] != d {
                    continue;
                }
                self.edges[src].iter().for_each(|e| {
                    if dist[e.dst] > dist[src] + e.weight {
                        dist[e.dst] = dist[src] + e.weight;
                        heap.push((Reverse(dist[e.dst]), e.dst))
                    }
                });
            }
            dist
        }

        /// 各頂点の入次数を返す
        pub fn indegree(&self) -> Vec<i32> {
            (0..self.n)
                .map(|dst| self.rev_edges[dst].len() as i32)
                .collect()
        }

        /// 各頂点の出次数を返す
        pub fn outdegree(&self) -> Vec<i32> {
            (0..self.n)
                .map(|src| self.edges[src].len() as i32)
                .collect()
        }
    }
}

pub fn solve<R: BufRead, W: Write>(mut reader: Reader<R>, mut output: Writer<W>) {
    let n = reader.u();
    let mut g = Graph::new(n);
    for _ in 0..n - 1 {
        g.add_edge(reader.u() - 1, reader.u() - 1, 1);
    }

    if n == 2 {
        output.println(&1);
        return;
    }

    let mut dist1 = vec![INF; n];
    dist1[0] = 0;
    let mut q = VecDeque::new();
    q.push_front(0);
    while let Some(src) = q.pop_front() {
        for e in &g.edges[src] {
            if dist1[e.dst] > dist1[src] + 1 {
                dist1[e.dst] = dist1[src] + 1;
                q.push_front(e.dst);
            }
        }
    }
    let mut d = 0;
    let mut s = 0;
    for i in 0..n {
        if d < dist1[i] {
            d = dist1[i];
            s = i;
        }
    }

    let mut dist2 = vec![INF; n];
    dist2[s] = 0;
    let mut q = VecDeque::new();
    q.push_front(s);
    while let Some(src) = q.pop_front() {
        for e in &g.edges[src] {
            if dist2[e.dst] > dist2[src] + 1 {
                dist2[e.dst] = dist2[src] + 1;
                q.push_front(e.dst);
            }
        }
    }

    let mut d = 0;
    let mut t = 0;
    for i in 0..n {
        if d < dist2[i] {
            d = dist2[i];
            t = i;
        }
    }

    let mut diameter = VecDeque::new();
    let mut path = VecDeque::new();
    path.push_front(s);
    dfs(s, s, t, &g, &mut path, &mut diameter);

    let mut ans = mi(1);
    if d % 2 == 1 {
        let x = diameter[diameter.len() / 2];
        let y = diameter[diameter.len() / 2 - 1];
        let mut dist3 = vec![INF; n];
        dist3[x] = 0;
        dist3[y] = -1;
        let mut q = VecDeque::new();
        q.push_front(x);
        while let Some(src) = q.pop_front() {
            for e in &g.edges[src] {
                if dist3[e.dst] > dist3[src] + 1 {
                    dist3[e.dst] = dist3[src] + 1;
                    q.push_front(e.dst);
                }
            }
        }
        let x_count = dist3.iter().filter(|&&di| di == d / 2).count();
        let mut dist3 = vec![INF; n];
        dist3[x] = -1;
        dist3[y] = 0;
        let mut q = VecDeque::new();
        q.push_front(y);
        while let Some(src) = q.pop_front() {
            for e in &g.edges[src] {
                if dist3[e.dst] > dist3[src] + 1 {
                    dist3[e.dst] = dist3[src] + 1;
                    q.push_front(e.dst);
                }
            }
        }
        let y_count = dist3.iter().filter(|&&di| di == d / 2).count();
        ans = mi(x_count as i64) * y_count as i64;
    } else {
        let c = diameter[diameter.len() / 2];
        ans = mi(1);
        let mut counts = Vec::new();
        let mut dist3 = vec![INF; n];
        for e in &g.edges[c] {
            let mut count = 0;
            dist3[c] = -1;
            dist3[e.dst] = 0;
            if dist3[e.dst] == d / 2 - 1 {
                count += 1;
            }
            let mut q = VecDeque::new();
            q.push_front(e.dst);
            while let Some(src) = q.pop_front() {
                for e in &g.edges[src] {
                    if dist3[e.dst] > dist3[src] + 1 {
                        dist3[e.dst] = dist3[src] + 1;
                        if dist3[e.dst] == d / 2 - 1 {
                            count += 1;
                        }
                        q.push_front(e.dst);
                    }
                }
            }
            counts.push(count);
        }
        for &count in &counts {
            ans *= count + 1;
        }

        for &count in &counts {
            ans -= count;
        }
        ans -= 1;
    }
    output.println(&ans);
}

fn dfs(
    cur: usize,
    par: usize,
    to: usize,
    g: &Graph,
    path: &mut VecDeque<usize>,
    diameter: &mut VecDeque<usize>,
) {
    if cur == to {
        diameter.append(path);
        return;
    }
    for e in &g.edges[cur] {
        if e.dst == par {
            continue;
        }
        path.push_back(e.dst);
        dfs(e.dst, e.src, to, g, path, diameter);
        path.pop_back();
    }
}

#[allow(unused_imports)]
pub use mod_int::*;

#[allow(dead_code)]
pub mod mod_int {
    use std::marker::PhantomData;
    use std::ops::*;

    pub fn mi(i: i64) -> Mi {
        Mi::new(i)
    }

    pub trait Mod: Copy + Clone + std::fmt::Debug {
        fn get() -> i64;
    }

    pub type Mi = ModInt<Mod998244353>;

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
        _p: PhantomData<M>,
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
            ModInt::new(self.n + rhs.rem_euclid(M::get()))
        }
    }

    impl<M: Mod> Add<ModInt<M>> for ModInt<M> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            self + rhs.n
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
            ModInt::new(self.n - rhs.rem_euclid(M::get()))
        }
    }

    impl<M: Mod> Sub<ModInt<M>> for ModInt<M> {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            self - rhs.n
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

    impl<M: Mod> std::fmt::Display for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.n)
        }
    }

    impl<M: Mod> std::fmt::Debug for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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
}
