pub use reader::*;
#[allow(unused_imports)]
use {
    itertools::Itertools,
    num::Integer,
    proconio::fastout,
    std::convert::TryInto,
    std::{cmp::*, collections::*, io::*, num::*, str::*},
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
#[rustfmt::skip]
pub mod reader { #[allow(unused_imports)] use itertools::Itertools; use std::{fmt::Debug, io::*, str::*};  pub struct Reader<R: BufRead> { reader: R, buf: Vec<u8>, pos: usize, }  macro_rules! prim_method { ($name:ident: $T: ty) => { pub fn $name(&mut self) -> $T { self.n::<$T>() } }; ($name:ident) => { prim_method!($name: $name); } } macro_rules! prim_methods { ($name:ident: $T:ty; $($rest:tt)*) => { prim_method!($name:$T); prim_methods!($($rest)*); }; ($name:ident; $($rest:tt)*) => { prim_method!($name); prim_methods!($($rest)*); }; () => () }  macro_rules! replace_expr { ($_t:tt $sub:expr) => { $sub }; } macro_rules! tuple_method { ($name: ident: ($($T:ident),+)) => { pub fn $name(&mut self) -> ($($T),+) { ($(replace_expr!($T self.n())),+) } } } macro_rules! tuple_methods { ($name:ident: ($($T:ident),+); $($rest:tt)*) => { tuple_method!($name:($($T),+)); tuple_methods!($($rest)*); }; () => () } macro_rules! vec_method { ($name: ident: ($($T:ty),+)) => { pub fn $name(&mut self, n: usize) -> Vec<($($T),+)> { (0..n).map(|_|($(replace_expr!($T self.n())),+)).collect_vec() } }; ($name: ident: $T:ty) => { pub fn $name(&mut self, n: usize) -> Vec<$T> { (0..n).map(|_|self.n()).collect_vec() } }; } macro_rules! vec_methods { ($name:ident: ($($T:ty),+); $($rest:tt)*) => { vec_method!($name:($($T),+)); vec_methods!($($rest)*); }; ($name:ident: $T:ty; $($rest:tt)*) => { vec_method!($name:$T); vec_methods!($($rest)*); }; () => () } impl<R: BufRead> Reader<R> { pub fn new(reader: R) -> Reader<R> { let (buf, pos) = (Vec::new(), 0); Reader { reader, buf, pos } } prim_methods! { u: usize; i: i64; f: f64; str: String; c: char; string: String; u8; u16; u32; u64; u128; usize; i8; i16; i32; i64; i128; isize; f32; f64; char; } tuple_methods! { u2: (usize, usize); u3: (usize, usize, usize); u4: (usize, usize, usize, usize); i2: (i64, i64); i3: (i64, i64, i64); i4: (i64, i64, i64, i64); cuu: (char, usize, usize); } vec_methods! { uv: usize; uv2: (usize, usize); uv3: (usize, usize, usize); iv: i64; iv2: (i64, i64); iv3: (i64, i64, i64); vq: (char, usize, usize); }  pub fn n<T: FromStr>(&mut self) -> T where T::Err: Debug, { self.n_op().unwrap() }  pub fn n_op<T: FromStr>(&mut self) -> Option<T> where T::Err: Debug, { if self.buf.is_empty() { self._read_next_line(); } let mut start = None; while self.pos != self.buf.len() { match (self.buf[self.pos], start.is_some()) { (b' ', true) | (b'\n', true) => break, (_, true) | (b' ', false) => self.pos += 1, (b'\n', false) => self._read_next_line(), (_, false) => start = Some(self.pos), } } start.map(|s| from_utf8(&self.buf[s..self.pos]).unwrap().parse().unwrap()) }  fn _read_next_line(&mut self) { self.pos = 0; self.buf.clear(); self.reader.read_until(b'\n', &mut self.buf).unwrap(); } pub fn s(&mut self) -> Vec<char> { self.n::<String>().chars().collect() } pub fn digits(&mut self) -> Vec<i64> { self.n::<String>() .chars() .map(|c| (c as u8 - b'0') as i64) .collect() } pub fn char_map(&mut self, h: usize) -> Vec<Vec<char>> { (0..h).map(|_| self.s()).collect() } pub fn bool_map(&mut self, h: usize, ng: char) -> Vec<Vec<bool>> { self.char_map(h) .iter() .map(|v| v.iter().map(|&c| c != ng).collect()) .collect() } pub fn matrix(&mut self, h: usize, w: usize) -> Vec<Vec<i64>> { (0..h).map(|_| self.iv(w)).collect() } } }

#[allow(unused_imports)]
use graph::*;

#[allow(dead_code)]
pub mod graph {
    use std::cmp::Ordering;
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, VecDeque};
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
        n: usize,
        edges: Vec<Vec<Edge>>,
        rev_edges: Vec<Vec<Edge>>,
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
        pub fn from_matrix(weights: &Vec<Vec<Weight>>, n: usize) -> Graph {
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
        fn indegree(&self) -> Vec<i32> {
            (0..self.n)
                .map(|dst| self.rev_edges[dst].len() as i32)
                .collect()
        }

        /// 各頂点の出次数を返す
        fn outdegree(&self) -> Vec<i32> {
            (0..self.n)
                .map(|src| self.edges[src].len() as i32)
                .collect()
        }

        /// 頂点をトポロジカルソートして返す
        /// グラフがDAGの場合に使用可
        pub fn topological_sort(&self) -> Vec<usize> {
            let mut deg = self.indegree();

            let mut q = VecDeque::new();
            for i in 0..self.n {
                if deg[i] == 0 {
                    q.push_back(i);
                }
            }

            let mut ret = Vec::new();
            while let Some(src) = q.pop_front() {
                self.edges[src].iter().for_each(|e| {
                    deg[e.dst] -= 1;
                    if deg[e.dst] == 0 {
                        q.push_back(e.dst)
                    }
                });
                ret.push(src);
            }
            ret
        }

        /// lを始点とする各点までの経路数を求める
        /// グラフがDAGの場合に使用可
        pub fn path(&self, l: usize) -> Vec<usize> {
            let list = self.topological_sort();
            let mut dp = vec![0; self.n];
            dp[l] = 1;
            for src in list {
                for e in &self.edges[src] {
                    dp[e.dst] += dp[src];
                }
            }
            dp
        }

        /// 後退解析で各点をスタートとしたときの勝敗を求める
        /// 0: 未定/引き分け
        /// 1: 勝ち
        /// 2: 負け
        pub fn retrograde_analysis(&self) -> Vec<usize> {
            #[derive(Clone, Copy, Debug, PartialEq)]
            enum Res {
                DRAW,
                WIN,
                LOSE,
            }
            let mut deg = self.outdegree();
            let mut ret = vec![Res::DRAW; self.n];

            let mut q = VecDeque::new();
            for i in 0..self.n {
                if deg[i] == 0 {
                    ret[i] = Res::LOSE;
                    q.push_back(i);
                }
            }
            while let Some(src) = q.pop_front() {
                self.rev_edges[src].iter().for_each(|e| {
                    if ret[e.dst] == Res::DRAW {
                        deg[e.dst] -= 1;
                        if ret[src] == Res::LOSE {
                            ret[e.dst] = Res::WIN;
                            q.push_back(e.dst);
                        } else if deg[e.dst] == 0 {
                            ret[e.dst] = Res::LOSE;
                            q.push_back(e.dst);
                        }
                    }
                });
            }
            ret.into_iter().map(|r| r as usize).collect()
        }
    }
}

#[allow(dead_code)]
fn main() {
    let stdin = stdin();
    solve(Reader::new(stdin.lock()));
}

#[fastout]
pub fn solve<R: BufRead>(mut reader: Reader<R>) {
    let n = reader.u();

    // println!("{}", (0..58).map(|i| (i + b'A') as char).join(" "));
    //
    let mut start = Vec::new();
    let mut g = Graph::new(58 * 58 * 58);
    for _ in 0..n {
        let s = reader.s();
        let prefix = char3_to_usize(s[0], s[1], s[2]);
        let suffix = char3_to_usize(s[s.len() - 3], s[s.len() - 2], s[s.len() - 1]);
        start.push(suffix);
        g.add_arc(prefix, suffix, 1);
    }

    let res = g.retrograde_analysis();
    for s in start {
        println!(
            "{}",
            match res[s] {
                0 => "Draw",
                1 => "Aoki",
                2 => "Takahashi",
                _ => unreachable!(),
            }
        );
    }
}

fn char3_to_usize(c1: char, c2: char, c3: char) -> usize {
    (c1 as u8 - b'A') as usize * 58 * 58
        + (c2 as u8 - b'A') as usize * 58
        + (c3 as u8 - b'A') as usize
}
