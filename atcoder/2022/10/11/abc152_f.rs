# [rustfmt :: skip ] pub struct Writer < W : Write > {writer : BufWriter < W > , }
# [rustfmt :: skip ] impl < W : Write > Writer < W > {pub fn new (write : W ) -> Self {Self {writer : BufWriter :: new (write ) , } } pub fn ln < S : Display > (& mut self , s : S ) {writeln ! (self . writer , "{}" , s ) . expect ("Failed to write." ) } pub fn out < S : Display > (& mut self , s : S ) {write ! (self . writer , "{}" , s ) . expect ("Failed to write." ) } pub fn join < S : Display > (& mut self , v : & [S ] , separator : & str ) {v . iter () . fold ("" , | sep , arg | {write ! (self . writer , "{}{}" , sep , arg ) . expect ("Failed to write." ) ; separator } ) ; writeln ! (self . writer ) . expect ("Failed to write." ) ; } pub fn bits (& mut self , i : i64 , len : usize ) {(0 .. len ) . for_each (| b | write ! (self . writer , "{}" , i >> b & 1 ) . expect ("Failed to write." ) ) ; writeln ! (self . writer ) . expect ("Failed to write." ) } pub fn flush (& mut self ) {let _ = self . writer . flush () ; } }
# [rustfmt :: skip ] pub struct Reader < F > {init : F , buf : VecDeque < String > , }
# [rustfmt :: skip ] mod reader_impl {use super :: {BufRead , Reader , VecDeque , FromStr as FS } ; impl < R : BufRead , F : FnMut () -> R > Iterator for Reader < F > {type Item = String ; fn next (& mut self ) -> Option < String > {if self . buf . is_empty () {let mut reader = (self . init ) () ; let mut l = String :: new () ; reader . read_line (& mut l ) . unwrap () ; self . buf . append (& mut l . split_whitespace () . map (ToString :: to_string ) . collect () ) ; } self . buf . pop_front () } } impl < R : BufRead , F : FnMut () -> R > Reader < F > {pub fn new (init : F ) -> Self {let buf = VecDeque :: new () ; Reader {init , buf } } pub fn v < T : FS > (& mut self ) -> T {let s = self . next () . expect ("Insufficient input." ) ; s . parse () . ok () . expect ("Failed to parse." ) } pub fn v2 < T1 : FS , T2 : FS > (& mut self ) -> (T1 , T2 ) {(self . v () , self . v () ) } pub fn v3 < T1 : FS , T2 : FS , T3 : FS > (& mut self ) -> (T1 , T2 , T3 ) {(self . v () , self . v () , self . v () ) } pub fn v4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 ) {(self . v () , self . v () , self . v () , self . v () ) } pub fn v5 < T1 : FS , T2 : FS , T3 : FS , T4 : FS , T5 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 , T5 ) {(self . v () , self . v () , self . v () , self . v () , self . v () ) } pub fn vec < T : FS > (& mut self , length : usize ) -> Vec < T > {(0 .. length ) . map (| _ | self . v () ) . collect () } pub fn vec2 < T1 : FS , T2 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 ) > {(0 .. length ) . map (| _ | self . v2 () ) . collect () } pub fn vec3 < T1 : FS , T2 : FS , T3 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 ) > {(0 .. length ) . map (| _ | self . v3 () ) . collect () } pub fn vec4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 , T4 ) > {(0 .. length ) . map (| _ | self . v4 () ) . collect () } pub fn chars (& mut self ) -> Vec < char > {self . v :: < String > () . chars () . collect () } fn split (& mut self , zero : u8 ) -> Vec < usize > {self . v :: < String > () . chars () . map (| c | (c as u8 - zero ) as usize ) . collect () } pub fn digits (& mut self ) -> Vec < usize > {self . split (b'0' ) } pub fn lowercase (& mut self ) -> Vec < usize > {self . split (b'a' ) } pub fn uppercase (& mut self ) -> Vec < usize > {self . split (b'A' ) } pub fn char_map (& mut self , h : usize ) -> Vec < Vec < char > > {(0 .. h ) . map (| _ | self . chars () ) . collect () } pub fn bool_map (& mut self , h : usize , ng : char ) -> Vec < Vec < bool > > {self . char_map (h ) . iter () . map (| v | v . iter () . map (| & c | c != ng ) . collect () ) . collect () } pub fn matrix < T : FS > (& mut self , h : usize , w : usize ) -> Vec < Vec < T > > {(0 .. h ) . map (| _ | self . vec (w ) ) . collect () } } }
# [rustfmt :: skip ] pub trait ToLR < T > {fn to_lr (& self ) -> (T , T ) ; }
# [rustfmt :: skip ] impl < R : RangeBounds < T > , T : Copy + BoundedAbove + BoundedBelow + One + Add < Output = T > > ToLR < T > for R {fn to_lr (& self ) -> (T , T ) {use Bound :: {Excluded , Included , Unbounded } ; let l = match self . start_bound () {Unbounded => T :: min_value () , Included (& s ) => s , Excluded (& s ) => s + T :: one () , } ; let r = match self . end_bound () {Unbounded => T :: max_value () , Included (& e ) => e + T :: one () , Excluded (& e ) => e , } ; (l , r ) } }
# [rustfmt :: skip ] pub use std :: {cmp :: {max , min , Ordering , Reverse } , collections :: {hash_map :: RandomState , BTreeMap , BTreeSet , BinaryHeap , HashMap , HashSet , VecDeque , } , convert :: Infallible , convert :: {TryFrom , TryInto } , fmt :: {Debug , Display , Formatter } , hash :: Hash , io :: {stdin , stdout , BufRead , BufWriter , Read , Write } , iter :: {repeat , Product , Sum } , marker :: PhantomData , mem :: swap , ops :: {Add , AddAssign , BitAnd , BitAndAssign , BitOr , BitOrAssign , BitXor , BitXorAssign , Bound , Deref , DerefMut , Div , DivAssign , Index , IndexMut , Mul , MulAssign , Neg , Not , Range , RangeBounds , Rem , RemAssign , Shl , ShlAssign , Shr , ShrAssign , Sub , SubAssign , } , str :: {from_utf8 , FromStr } , } ;
#[allow(unused_macros)]
macro_rules ! chmin {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_min = min ! ($ ($ cmps ) ,+ ) ; if $ base > cmp_min {$ base = cmp_min ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! min {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ b } else {$ a } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = min ! ($ ($ rest ) ,+ ) ; if $ a > b {b } else {$ a } } } ; }
#[allow(unused_macros)]
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
#[allow(unused_macros)]
macro_rules ! dbg {($ ($ x : tt ) * ) => {{# [cfg (debug_assertions ) ] {std :: dbg ! ($ ($ x ) * ) } # [cfg (not (debug_assertions ) ) ] {($ ($ x ) * ) } } } }
# [rustfmt :: skip ] pub use algebra_traits :: {AbelianGroup , Associative , Band , BoundedAbove , BoundedBelow , Commutative , CommutativeMonoid , Group , Idempotent , Invertible , Magma , MapMonoid , Monoid , One , Pow , PrimitiveRoot , SemiGroup , Unital , Zero , } ;
# [rustfmt :: skip ] mod algebra_traits {use super :: Debug ; pub trait Magma {type M : Clone + PartialEq + Debug ; fn op (x : & Self :: M , y : & Self :: M ) -> Self :: M ; } pub trait Associative {} pub trait Unital : Magma {fn unit () -> Self :: M ; } pub trait Commutative : Magma {} pub trait Invertible : Magma {fn inv (x : & Self :: M ) -> Self :: M ; } pub trait Idempotent : Magma {} pub trait SemiGroup : Magma + Associative {} pub trait Monoid : Magma + Associative + Unital {fn pow (& self , x : Self :: M , mut n : usize ) -> Self :: M {let mut res = Self :: unit () ; let mut base = x ; while n > 0 {if n & 1 == 1 {res = Self :: op (& res , & base ) ; } base = Self :: op (& base , & base ) ; n >>= 1 ; } res } } pub trait CommutativeMonoid : Magma + Associative + Unital + Commutative {} pub trait Group : Magma + Associative + Unital + Invertible {} pub trait AbelianGroup : Magma + Associative + Unital + Commutative + Invertible {} pub trait Band : Magma + Associative + Idempotent {} impl < M : Magma + Associative > SemiGroup for M {} impl < M : Magma + Associative + Unital > Monoid for M {} impl < M : Magma + Associative + Unital + Commutative > CommutativeMonoid for M {} impl < M : Magma + Associative + Unital + Invertible > Group for M {} impl < M : Magma + Associative + Unital + Commutative + Invertible > AbelianGroup for M {} impl < M : Magma + Associative + Idempotent > Band for M {} pub trait MapMonoid {type Mono : Monoid ; type Func : Monoid ; fn op (& self , x : & < Self :: Mono as Magma > :: M , y : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M {Self :: Mono :: op (x , y ) } fn unit () -> < Self :: Mono as Magma > :: M {Self :: Mono :: unit () } fn apply (& self , f : & < Self :: Func as Magma > :: M , value : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M ; fn identity_map () -> < Self :: Func as Magma > :: M {Self :: Func :: unit () } fn compose (& self , f : & < Self :: Func as Magma > :: M , g : & < Self :: Func as Magma > :: M , ) -> < Self :: Func as Magma > :: M {Self :: Func :: op (f , g ) } } pub trait Zero {fn zero () -> Self ; } pub trait One {fn one () -> Self ; } pub trait BoundedBelow {fn min_value () -> Self ; } pub trait BoundedAbove {fn max_value () -> Self ; } pub trait Pow {fn pow (self , exp : i64 ) -> Self ; } pub trait PrimitiveRoot {fn primitive_root () -> Self ; } }
# [rustfmt :: skip ] pub trait Integral : 'static + Send + Sync + Copy + Ord + Display + Debug + Add < Output = Self > + Sub < Output = Self > + Mul < Output = Self > + Div < Output = Self > + Rem < Output = Self > + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + Product + BitOr < Output = Self > + BitAnd < Output = Self > + BitXor < Output = Self > + Not < Output = Self > + Shl < Output = Self > + Shr < Output = Self > + BitOrAssign + BitAndAssign + BitXorAssign + ShlAssign + ShrAssign + Zero + One + BoundedBelow + BoundedAbove {}
macro_rules ! impl_integral {($ ($ ty : ty ) ,* ) => {$ (impl Zero for $ ty {fn zero () -> Self {0 } } impl One for $ ty {fn one () -> Self {1 } } impl BoundedBelow for $ ty {fn min_value () -> Self {Self :: min_value () } } impl BoundedAbove for $ ty {fn max_value () -> Self {Self :: max_value () } } impl Integral for $ ty {} ) * } ; }
impl_integral!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
pub fn main() {
    let stdin = stdin();
    let stdout = stdout();
    solve(Reader::new(|| stdin.lock()), Writer::new(stdout.lock()));
}

pub trait GraphTrait {
    type Weight;
    fn size(&self) -> usize;
    fn edges(&self, src: usize) -> Vec<(usize, Self::Weight)>;
    fn rev_edges(&self, dst: usize) -> Vec<(usize, Self::Weight)>;
    fn indegree(&self) -> Vec<i32> {
        (0..self.size())
            .map(|dst| self.rev_edges(dst).len() as i32)
            .collect()
    }
    fn outdegree(&self) -> Vec<i32> {
        (0..self.size())
            .map(|src| self.edges(src).len() as i32)
            .collect()
    }
}

pub struct Graph<W> {
    pub n: usize,
    pub edges: Vec<(usize, usize, W)>,
    pub index: Vec<Vec<usize>>,
    pub rev_index: Vec<Vec<usize>>,
}
impl<W: Clone> GraphTrait for Graph<W> {
    type Weight = W;
    fn size(&self) -> usize {
        self.n
    }
    fn edges(&self, src: usize) -> Vec<(usize, W)> {
        self.index[src]
            .iter()
            .map(|i| {
                let (_src, dst, w) = &self.edges[*i];
                (*dst, w.clone())
            })
            .collect()
    }
    fn rev_edges(&self, dst: usize) -> Vec<(usize, W)> {
        self.rev_index[dst]
            .iter()
            .map(|i| {
                let (src, _dst, w) = &self.edges[*i];
                (*src, w.clone())
            })
            .collect()
    }
}
impl<W: Clone> Clone for Graph<W> {
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            edges: self.edges.clone(),
            index: self.index.clone(),
            rev_index: self.rev_index.clone(),
        }
    }
}
impl<W: Clone> Graph<W> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
            index: vec![Vec::new(); n],
            rev_index: vec![Vec::new(); n],
        }
    }
    pub fn add_edge(&mut self, src: usize, dst: usize, w: W) -> (usize, usize) {
        let i = self.edges.len();
        self.edges.push((src, dst, w.clone()));
        self.index[src].push(i);
        self.rev_index[dst].push(i);
        let j = self.edges.len();
        self.edges.push((dst, src, w));
        self.index[dst].push(j);
        self.rev_index[src].push(j);
        (i, j)
    }
    pub fn add_arc(&mut self, src: usize, dst: usize, w: W) -> usize {
        let i = self.edges.len();
        self.edges.push((src, dst, w));
        self.index[src].push(i);
        self.rev_index[dst].push(i);
        i
    }
    pub fn all_edges(&self) -> Vec<(usize, usize, W)> {
        self.edges.clone()
    }
}
impl<W: Debug> Debug for Graph<W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "n: {}, m: {}", self.n, self.edges.len()).unwrap();
        for (src, dst, w) in &self.edges {
            writeln!(f, "({} -> {}): {:?}", src, dst, w).unwrap();
        }
        Ok(())
    }
}
#[derive(Clone)]
pub struct SparseTable<B: Band> {
    pub size: usize,
    pub table: Vec<Vec<B::M>>,
}
mod sparse_table_impl {
    use super::{Band, Debug, Formatter, RangeBounds, SparseTable, ToLR};
    impl<B: Band> From<&[B::M]> for SparseTable<B> {
        fn from(v: &[B::M]) -> Self {
            let size = v.len();
            let l = v.len();
            let lg = 63 - l.leading_zeros();
            let mut table = vec![Vec::new(); lg as usize + 1];
            table[0] = v.to_vec();
            let mut k = 1;
            while 1 << k <= size {
                table[k] = (0..=size - (1 << k))
                    .map(|i| B::op(&table[k - 1][i], &table[k - 1][i + (1 << (k - 1))]))
                    .collect();
                k += 1;
            }
            Self { size, table }
        }
    }
    impl<B: Band> SparseTable<B> {
        pub fn query<R: RangeBounds<usize>>(&self, range: R) -> B::M {
            let (l, r) = range.to_lr();
            let lg = 63 - (r - l).leading_zeros();
            B::op(
                &self.table[lg as usize][l],
                &self.table[lg as usize][r - (1 << lg)],
            )
        }
    }
    impl<B: Band> Debug for SparseTable<B>
    where
        B::M: Debug,
    {
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            for i in 0..self.size {
                writeln!(f, "{:?}", self.query(i..=i))?;
            }
            Ok(())
        }
    }
}

#[derive(Clone, Debug)]
pub struct EulerTour {
    pub time_in: Vec<usize>,
    pub time_out: Vec<usize>,
    pub depth: Vec<usize>,
    pub parent: Vec<usize>,
    pub tour: Vec<usize>,
}
impl EulerTour {
    pub fn new<G: GraphTrait>(g: &G, root: usize) -> Self {
        let mut tour = EulerTour {
            time_in: vec![0; g.size()],
            time_out: vec![0; g.size()],
            depth: vec![0; g.size()],
            parent: vec![0; g.size()],
            tour: Vec::new(),
        };
        tour.dfs(root, root, 0, g);
        tour
    }
    fn dfs<G: GraphTrait>(&mut self, cur: usize, par: usize, d: usize, g: &G) {
        self.parent[cur] = par;
        self.depth[cur] = d;
        self.time_in[cur] = self.tour.len();
        self.tour.push(cur);
        for (dst, _) in g.edges(cur) {
            if dst == par {
                continue;
            }
            self.dfs(dst, cur, d + 1, g);
            self.tour.push(cur);
        }
        self.time_out[cur] = self.tour.len();
    }
}

#[derive(Clone, Debug, Default)]
pub struct Minimization<S>(PhantomData<fn() -> S>);
mod minimization_impl {
    use super::{
        Associative, BoundedAbove, Commutative, Debug, Idempotent, Magma, Minimization, Unital,
    };
    impl<S: Clone + Debug + PartialOrd> Magma for Minimization<S> {
        type M = S;
        fn op(x: &Self::M, y: &Self::M) -> Self::M {
            if x <= y {
                x.clone()
            } else {
                y.clone()
            }
        }
    }
    impl<S: BoundedAbove + Debug + Clone + PartialOrd> Unital for Minimization<S> {
        fn unit() -> Self::M {
            S::max_value()
        }
    }
    impl<S: Clone + Debug + PartialOrd> Associative for Minimization<S> {}
    impl<S: Clone + Debug + PartialOrd> Commutative for Minimization<S> {}
    impl<S: Clone + Debug + PartialOrd> Idempotent for Minimization<S> {}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntWithIndex<I: Integral> {
    pub value: I,
    pub index: usize,
}
impl<I: Integral> PartialOrd for IntWithIndex<I> {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        use Ordering::*;
        let r = match self.value.cmp(&rhs.value) {
            Greater => Greater,
            Less => Less,
            Equal => self.index.cmp(&rhs.index),
        };
        Some(r)
    }
}
impl<I: Integral> Ord for IntWithIndex<I> {
    fn cmp(&self, rhs: &Self) -> Ordering {
        self.partial_cmp(rhs).unwrap()
    }
}
impl<I: Integral> From<(usize, I)> for IntWithIndex<I> {
    fn from((index, value): (usize, I)) -> Self {
        IntWithIndex { value, index }
    }
}
pub struct LowestCommonAncestor {
    tour: EulerTour,
    depth: SparseTable<Minimization<IntWithIndex<usize>>>,
}
impl LowestCommonAncestor {
    pub fn new<G: GraphTrait>(g: &G, root: usize) -> Self {
        let tour = EulerTour::new(g, root);
        let depth = SparseTable::<Minimization<IntWithIndex<usize>>>::from(
            &tour
                .tour
                .iter()
                .map(|i| tour.depth[*i])
                .enumerate()
                .map(IntWithIndex::from)
                .collect::<Vec<_>>()[..],
        );
        Self { tour, depth }
    }
    pub fn query(&self, u: usize, v: usize) -> usize {
        let (mut l, mut r) = (self.tour.time_in[u], self.tour.time_out[v]);
        if l > r {
            swap(&mut l, &mut r)
        }
        self.tour.tour[self.depth.query(l..r).index]
    }
    pub fn path(&self, mut u: usize, mut v: usize) -> Vec<usize> {
        let lca = self.query(u, v);
        let mut left = Vec::new();
        while u != lca {
            left.push(u);
            u = self.tour.parent[u];
        }
        left.push(lca);
        let mut right = Vec::new();
        while v != lca {
            right.push(v);
            v = self.tour.parent[v];
        }
        right.reverse();
        left.append(&mut right);
        left
    }
    pub fn dist(&self, u: usize, v: usize) -> usize {
        let lca = self.query(u, v);
        self.tour.depth[u] + self.tour.depth[v] - 2 * self.tour.depth[lca]
    }
    pub fn on_path(&self, u: usize, v: usize, a: usize) -> bool {
        self.dist(u, a) + self.dist(a, v) == self.dist(u, v)
    }
    pub fn auxiliary_tree(&self, vs: &mut Vec<usize>) -> Vec<(usize, usize)> {
        vs.sort_by_key(|v| self.tour.time_in[*v]);
        let mut stack = vec![vs[0]];
        let mut edges = Vec::new();
        for i in 1..vs.len() {
            let lca = self.query(vs[i - 1], vs[i]);
            if lca != vs[i - 1] {
                let mut last = stack.pop().unwrap();
                while !stack.is_empty()
                    && self.tour.depth[lca] < self.tour.depth[stack[stack.len() - 1]]
                {
                    edges.push((stack[stack.len() - 1], last));
                    last = stack.pop().unwrap();
                }
                if stack.is_empty() || stack[stack.len() - 1] != lca {
                    stack.push(lca);
                    vs.push(lca);
                }
                edges.push((lca, last));
            }
            stack.push(vs[i]);
        }
        for i in 1..stack.len() {
            edges.push((stack[i - 1], stack[i]));
        }
        edges
    }
}
pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let n: usize = reader.v();
    let ab = reader.vec2::<usize, usize>(n - 1);
    let m = reader.v::<usize>();
    let uv = reader.vec2::<usize, usize>(m);
    let mut graph = Graph::new(n);
    for (a, b) in ab {
        graph.add_edge(a - 1, b - 1, ());
    }
    let lca = LowestCommonAncestor::new(&graph, 0);
    let mut ans = 0i64;
    for p in 0usize..1 << m {
        let mut used = 0usize;
        for i in 0..m {
            if p >> i & 1 == 1 {
                let (mut u, mut v) = (uv[i].0 - 1, uv[i].1 - 1);
                let a = lca.query(u, v);
                while u != a {
                    used |= 1 << u;
                    u = lca.tour.parent[u];
                }
                while v != a {
                    used |= 1 << v;
                    v = lca.tour.parent[v];
                }
            }
        }
        let rest = n - 1 - used.count_ones() as usize;
        // dbg!(p, rest);
        if p.count_ones() % 2 == 0 {
            ans += 1 << rest;
        } else {
            ans -= 1 << rest;
        }
    }
    if ans < 0 {
        panic!()
    }
    writer.ln(ans);
}
