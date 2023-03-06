# [rustfmt :: skip ] pub use string_util_impl :: {AddLineTrait , BitsTrait , JoinTrait } ;
# [rustfmt :: skip ] mod string_util_impl {use super :: {Display , Integral } ; pub trait AddLineTrait {fn ln (& self ) -> String ; } impl < D : Display > AddLineTrait for D {fn ln (& self ) -> String {self . to_string () + "\n" } } pub trait JoinTrait {fn join (self , separator : & str ) -> String ; } impl < D : Display , I : IntoIterator < Item = D > > JoinTrait for I {fn join (self , separator : & str ) -> String {let mut buf = String :: new () ; self . into_iter () . fold ("" , | sep , arg | {buf . push_str (& format ! ("{}{}" , sep , arg ) ) ; separator } ) ; buf } } pub trait BitsTrait {fn bits (self , length : Self ) -> String ; } impl < I : Integral > BitsTrait for I {fn bits (self , length : Self ) -> String {let mut buf = String :: new () ; let mut i = I :: zero () ; while i < length {buf . push_str (& format ! ("{}" , self >> i & I :: one () ) ) ; i += I :: one () ; } buf + "\n" } } }
# [rustfmt :: skip ] pub trait ToLR < T > {fn to_lr (& self ) -> (T , T ) ; }
# [rustfmt :: skip ] impl < R : RangeBounds < T > , T : Copy + BoundedAbove + BoundedBelow + One + Add < Output = T > > ToLR < T > for R {# [inline ] fn to_lr (& self ) -> (T , T ) {use Bound :: {Excluded , Included , Unbounded } ; let l = match self . start_bound () {Unbounded => T :: min_value () , Included (& s ) => s , Excluded (& s ) => s + T :: one () , } ; let r = match self . end_bound () {Unbounded => T :: max_value () , Included (& e ) => e + T :: one () , Excluded (& e ) => e , } ; (l , r ) } }
# [rustfmt :: skip ] pub trait RangeProduct < I > {type Magma : Magma ; fn product < R : RangeBounds < I > > (& self , range : R ) -> < Self :: Magma as Magma > :: M ; }
# [rustfmt :: skip ] pub trait RangeProductMut < I > {type Magma : Magma ; fn product < R : RangeBounds < I > > (& mut self , range : R ) -> < Self :: Magma as Magma > :: M ; }
# [rustfmt :: skip ] pub use std :: {cmp :: {max , min , Ordering , Reverse } , collections :: {hash_map :: RandomState , BTreeMap , BTreeSet , BinaryHeap , VecDeque , } , convert :: Infallible , convert :: {TryFrom , TryInto } , default :: Default , fmt :: {Debug , Display , Formatter } , hash :: {Hash , BuildHasherDefault , Hasher } , io :: {stdin , stdout , BufRead , BufWriter , Read , Write , StdoutLock } , iter :: {repeat , Product , Sum } , marker :: PhantomData , mem :: swap , ops :: {Add , AddAssign , BitAnd , BitAndAssign , BitOr , BitOrAssign , BitXor , BitXorAssign , Bound , Deref , DerefMut , Div , DivAssign , Index , IndexMut , Mul , MulAssign , Neg , Not , Range , RangeBounds , Rem , RemAssign , Shl , ShlAssign , Shr , ShrAssign , Sub , SubAssign , } , str :: {from_utf8 , FromStr } , } ;
#[macro_export]
#[allow(unused_macros)]
macro_rules ! min {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ b } else {$ a } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = min ! ($ ($ rest ) ,+ ) ; if $ a > b {b } else {$ a } } } ; }
#[macro_export]
#[allow(unused_macros)]
macro_rules ! chmin {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_min = min ! ($ ($ cmps ) ,+ ) ; if $ base > cmp_min {$ base = cmp_min ; true } else {false } } } ; }
#[macro_export]
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
#[macro_export]
#[allow(unused_macros)]
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
# [rustfmt :: skip ] pub use io_impl :: {ReaderFromStdin , ReaderFromStr , ReaderTrait , WriterToStdout , WriterTrait , IO } ;
# [rustfmt :: skip ] mod io_impl {use super :: {stdin , stdout , BufRead , BufWriter , Display , FromStr as FS , VecDeque , Write } ; # [derive (Clone , Debug , Default ) ] pub struct IO {reader : ReaderFromStdin , writer : WriterToStdout , } pub trait ReaderTrait {fn next (& mut self ) -> Option < String > ; fn v < T : FS > (& mut self ) -> T {let s = self . next () . expect ("Insufficient input." ) ; s . parse () . ok () . expect ("Failed to parse." ) } fn v2 < T1 : FS , T2 : FS > (& mut self ) -> (T1 , T2 ) {(self . v () , self . v () ) } fn v3 < T1 : FS , T2 : FS , T3 : FS > (& mut self ) -> (T1 , T2 , T3 ) {(self . v () , self . v () , self . v () ) } fn v4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 ) {(self . v () , self . v () , self . v () , self . v () ) } fn v5 < T1 : FS , T2 : FS , T3 : FS , T4 : FS , T5 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 , T5 ) {(self . v () , self . v () , self . v () , self . v () , self . v () ) } fn vec < T : FS > (& mut self , length : usize ) -> Vec < T > {(0 .. length ) . map (| _ | self . v () ) . collect () } fn vec2 < T1 : FS , T2 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 ) > {(0 .. length ) . map (| _ | self . v2 () ) . collect () } fn vec3 < T1 : FS , T2 : FS , T3 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 ) > {(0 .. length ) . map (| _ | self . v3 () ) . collect () } fn vec4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 , T4 ) > {(0 .. length ) . map (| _ | self . v4 () ) . collect () } fn chars (& mut self ) -> Vec < char > {self . v :: < String > () . chars () . collect () } fn split (& mut self , zero : u8 ) -> Vec < usize > {self . v :: < String > () . chars () . map (| c | (c as u8 - zero ) as usize ) . collect () } fn digits (& mut self ) -> Vec < usize > {self . split (b'0' ) } fn lowercase (& mut self ) -> Vec < usize > {self . split (b'a' ) } fn uppercase (& mut self ) -> Vec < usize > {self . split (b'A' ) } fn char_map (& mut self , h : usize ) -> Vec < Vec < char > > {(0 .. h ) . map (| _ | self . chars () ) . collect () } fn bool_map (& mut self , h : usize , ng : char ) -> Vec < Vec < bool > > {self . char_map (h ) . iter () . map (| v | v . iter () . map (| & c | c != ng ) . collect () ) . collect () } fn matrix < T : FS > (& mut self , h : usize , w : usize ) -> Vec < Vec < T > > {(0 .. h ) . map (| _ | self . vec (w ) ) . collect () } } pub struct ReaderFromStr {buf : VecDeque < String > , } impl ReaderTrait for ReaderFromStr {fn next (& mut self ) -> Option < String > {self . buf . pop_front () } } impl ReaderFromStr {pub fn new (src : & str ) -> Self {Self {buf : src . split_whitespace () . map (ToString :: to_string ) . collect :: < Vec < String > > () . into () , } } pub fn push (& mut self , src : & str ) {for s in src . split_whitespace () . map (ToString :: to_string ) {self . buf . push_back (s ) ; } } pub fn from_file (path : & str ) -> Result < Self , Box < dyn std :: error :: Error > > {Ok (Self :: new (& std :: fs :: read_to_string (path ) ? ) ) } } impl WriterTrait for ReaderFromStr {fn out < S : Display > (& mut self , s : S ) {self . push (& s . to_string () ) ; } fn flush (& mut self ) {} } # [derive (Clone , Debug , Default ) ] pub struct ReaderFromStdin {buf : VecDeque < String > , } impl ReaderTrait for ReaderFromStdin {fn next (& mut self ) -> Option < String > {while self . buf . is_empty () {let stdin = stdin () ; let mut reader = stdin . lock () ; let mut l = String :: new () ; reader . read_line (& mut l ) . unwrap () ; self . buf . append (& mut l . split_whitespace () . map (ToString :: to_string ) . collect () ) ; } self . buf . pop_front () } } pub trait WriterTrait {fn out < S : Display > (& mut self , s : S ) ; fn flush (& mut self ) ; } # [derive (Clone , Debug , Default ) ] pub struct WriterToStdout {buf : String , } impl WriterTrait for WriterToStdout {fn out < S : Display > (& mut self , s : S ) {self . buf . push_str (& s . to_string () ) ; } fn flush (& mut self ) {if ! self . buf . is_empty () {let stdout = stdout () ; let mut writer = BufWriter :: new (stdout . lock () ) ; write ! (writer , "{}" , self . buf ) . expect ("Failed to write." ) ; let _ = writer . flush () ; self . buf . clear () ; } } } impl ReaderTrait for IO {fn next (& mut self ) -> Option < String > {self . reader . next () } } impl WriterTrait for IO {fn out < S : std :: fmt :: Display > (& mut self , s : S ) {self . writer . out (s ) } fn flush (& mut self ) {self . writer . flush () } } }
# [rustfmt :: skip ] pub use io_debug_impl :: IODebug ;
# [rustfmt :: skip ] mod io_debug_impl {use super :: {stdout , BufWriter , Display , ReaderFromStr , ReaderTrait , Write , WriterTrait } ; pub struct IODebug < F > {pub reader : ReaderFromStr , pub test_reader : ReaderFromStr , pub buf : String , stdout : bool , f : F , } impl < F : FnMut (& mut ReaderFromStr , & mut ReaderFromStr ) > WriterTrait for IODebug < F > {fn out < S : Display > (& mut self , s : S ) {self . buf . push_str (& s . to_string () ) ; } fn flush (& mut self ) {if self . stdout {let stdout = stdout () ; let mut writer = BufWriter :: new (stdout . lock () ) ; write ! (writer , "{}" , self . buf ) . expect ("Failed to write." ) ; let _ = writer . flush () ; } self . test_reader . push (& self . buf ) ; self . buf . clear () ; (self . f ) (& mut self . test_reader , & mut self . reader ) } } impl < F > ReaderTrait for IODebug < F > {fn next (& mut self ) -> Option < String > {self . reader . next () } } impl < F > IODebug < F > {pub fn new (str : & str , stdout : bool , f : F ) -> Self {Self {reader : ReaderFromStr :: new (str ) , test_reader : ReaderFromStr :: new ("" ) , buf : String :: new () , stdout , f , } } } }
# [rustfmt :: skip ] pub use self :: fxhasher_impl :: {FxHashMap as HashMap , FxHashSet as HashSet } ;
# [rustfmt :: skip ] mod fxhasher_impl {use super :: {BitXor , BuildHasherDefault , Hasher , TryInto } ; use std :: collections :: {HashMap , HashSet } ; # [derive (Default ) ] pub struct FxHasher {hash : u64 , } type BuildHasher = BuildHasherDefault < FxHasher > ; pub type FxHashMap < K , V > = HashMap < K , V , BuildHasher > ; pub type FxHashSet < V > = HashSet < V , BuildHasher > ; const ROTATE : u32 = 5 ; const SEED : u64 = 0x51_7c_c1_b7_27_22_0a_95 ; impl Hasher for FxHasher {# [inline ] fn finish (& self ) -> u64 {self . hash } # [inline ] fn write (& mut self , mut bytes : & [u8 ] ) {while bytes . len () >= 8 {self . add_to_hash (u64 :: from_ne_bytes (bytes [.. 8 ] . try_into () . unwrap () ) ) ; bytes = & bytes [8 .. ] ; } while bytes . len () >= 4 {self . add_to_hash (u64 :: from (u32 :: from_ne_bytes (bytes [.. 4 ] . try_into () . unwrap () , ) ) ) ; bytes = & bytes [4 .. ] ; } while bytes . len () >= 2 {self . add_to_hash (u64 :: from (u16 :: from_ne_bytes (bytes [.. 2 ] . try_into () . unwrap () , ) ) ) ; } if let Some (& byte ) = bytes . first () {self . add_to_hash (u64 :: from (byte ) ) ; } } } impl FxHasher {# [inline ] pub fn add_to_hash (& mut self , i : u64 ) {self . hash = self . hash . rotate_left (ROTATE ) . bitxor (i ) . wrapping_mul (SEED ) ; } } }
#[allow(unused_macros)]
macro_rules ! dbg {($ ($ x : tt ) * ) => {{# [cfg (debug_assertions ) ] {std :: dbg ! ($ ($ x ) * ) } # [cfg (not (debug_assertions ) ) ] {($ ($ x ) * ) } } } }
# [rustfmt :: skip ] pub use algebra_traits :: {AbelianGroup , Associative , Band , BoundedAbove , BoundedBelow , Commutative , CommutativeMonoid , Group , Idempotent , Integral , Invertible , Magma , MapMonoid , Monoid , One , Pow , PrimitiveRoot , SemiGroup , TrailingZeros , Unital , Zero , } ;
# [rustfmt :: skip ] mod algebra_traits {use super :: {Add , AddAssign , BitAnd , BitAndAssign , BitOr , BitOrAssign , BitXor , BitXorAssign , Debug , Display , Div , DivAssign , Mul , MulAssign , Not , Product , Rem , RemAssign , Shl , ShlAssign , Shr , ShrAssign , Sub , SubAssign , Sum , } ; pub trait Magma {type M : Clone + PartialEq + Debug ; fn op (x : & Self :: M , y : & Self :: M ) -> Self :: M ; } pub trait Associative {} pub trait Unital : Magma {fn unit () -> Self :: M ; } pub trait Commutative : Magma {} pub trait Invertible : Magma {fn inv (x : & Self :: M ) -> Self :: M ; } pub trait Idempotent : Magma {} pub trait SemiGroup : Magma + Associative {} pub trait Monoid : Magma + Associative + Unital {fn pow (& self , x : Self :: M , mut n : usize ) -> Self :: M {let mut res = Self :: unit () ; let mut base = x ; while n > 0 {if n & 1 == 1 {res = Self :: op (& res , & base ) ; } base = Self :: op (& base , & base ) ; n >>= 1 ; } res } } pub trait CommutativeMonoid : Magma + Associative + Unital + Commutative {} pub trait Group : Magma + Associative + Unital + Invertible {} pub trait AbelianGroup : Magma + Associative + Unital + Commutative + Invertible {} pub trait Band : Magma + Associative + Idempotent {} impl < M : Magma + Associative > SemiGroup for M {} impl < M : Magma + Associative + Unital > Monoid for M {} impl < M : Magma + Associative + Unital + Commutative > CommutativeMonoid for M {} impl < M : Magma + Associative + Unital + Invertible > Group for M {} impl < M : Magma + Associative + Unital + Commutative + Invertible > AbelianGroup for M {} impl < M : Magma + Associative + Idempotent > Band for M {} pub trait MapMonoid {type Mono : Monoid ; type Func : Monoid ; fn op (& self , x : & < Self :: Mono as Magma > :: M , y : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M {Self :: Mono :: op (x , y ) } fn unit () -> < Self :: Mono as Magma > :: M {Self :: Mono :: unit () } fn apply (& self , f : & < Self :: Func as Magma > :: M , value : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M ; fn identity_map () -> < Self :: Func as Magma > :: M {Self :: Func :: unit () } fn compose (& self , f : & < Self :: Func as Magma > :: M , g : & < Self :: Func as Magma > :: M , ) -> < Self :: Func as Magma > :: M {Self :: Func :: op (f , g ) } } pub trait Zero {fn zero () -> Self ; } pub trait One {fn one () -> Self ; } pub trait BoundedBelow {fn min_value () -> Self ; } pub trait BoundedAbove {fn max_value () -> Self ; } pub trait Pow {fn pow (self , exp : i64 ) -> Self ; } pub trait PrimitiveRoot {const DIVIDE_LIMIT : usize ; fn primitive_root () -> Self ; } pub trait TrailingZeros {fn trailing_zero (self ) -> Self ; } pub trait Integral : 'static + Send + Sync + Copy + Ord + Display + Debug + Add < Output = Self > + Sub < Output = Self > + Mul < Output = Self > + Div < Output = Self > + Rem < Output = Self > + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + Product + BitOr < Output = Self > + BitAnd < Output = Self > + BitXor < Output = Self > + Not < Output = Self > + Shl < Output = Self > + Shr < Output = Self > + BitOrAssign + BitAndAssign + BitXorAssign + ShlAssign + ShrAssign + Zero + One + BoundedBelow + BoundedAbove + TrailingZeros {} macro_rules ! impl_integral {($ ($ ty : ty ) ,* ) => {$ (impl Zero for $ ty {# [inline ] fn zero () -> Self {0 } } impl One for $ ty {# [inline ] fn one () -> Self {1 } } impl BoundedBelow for $ ty {# [inline ] fn min_value () -> Self {Self :: min_value () } } impl BoundedAbove for $ ty {# [inline ] fn max_value () -> Self {Self :: max_value () } } impl TrailingZeros for $ ty {# [inline ] fn trailing_zero (self ) -> Self {self . trailing_zeros () as $ ty } } impl Integral for $ ty {} ) * } ; } impl_integral ! (i8 , i16 , i32 , i64 , i128 , isize , u8 , u16 , u32 , u64 , u128 , usize ) ; }
# [rustfmt :: skip ] pub fn main () {std :: thread :: Builder :: new () . name ("extend stack size" . into () ) . stack_size (128 * 1024 * 1024 ) . spawn (move | | {let mut io = IO :: default () ; solve (& mut io ) ; io . flush () ; } ) . unwrap () . join () . unwrap () }

#[derive(Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}
impl UnionFind {
    pub fn new(n: usize) -> Self {
        let parent = (0..n + 1).collect::<Vec<_>>();
        let rank = vec![0; n + 1];
        let size = vec![1; n + 1];
        Self { parent, rank, size }
    }
    pub fn root(&mut self, x: usize) -> usize {
        if self.parent[x] == x {
            x
        } else {
            self.parent[x] = self.root(self.parent[x]);
            self.parent[x]
        }
    }
    pub fn rank(&self, x: usize) -> usize {
        self.rank[x]
    }
    pub fn size(&mut self, x: usize) -> usize {
        let root = self.root(x);
        self.size[root]
    }
    pub fn same(&mut self, x: usize, y: usize) -> bool {
        self.root(x) == self.root(y)
    }
    pub fn unite(&mut self, x: usize, y: usize) -> bool {
        let mut x = self.root(x);
        let mut y = self.root(y);
        if x == y {
            return false;
        }
        if self.rank(x) < self.rank(y) {
            swap(&mut x, &mut y);
        }
        if self.rank(x) == self.rank(y) {
            self.rank[x] += 1;
        }
        self.parent[y] = x;
        self.size[x] += self.size[y];
        true
    }
}

#[derive(Clone, Debug)]
pub struct XorShift {
    seed: u64,
}
mod xor_shift_impl {
    use super::{RangeBounds, ToLR, XorShift};
    use std::time::SystemTime;
    impl Default for XorShift {
        #[inline]
        fn default() -> Self {
            let seed = 0xf0fb588ca2196dac;
            Self { seed }
        }
    }
    impl Iterator for XorShift {
        type Item = u64;
        #[inline]
        fn next(&mut self) -> Option<u64> {
            self.seed ^= self.seed << 13;
            self.seed ^= self.seed >> 7;
            self.seed ^= self.seed << 17;
            Some(self.seed)
        }
    }
    impl XorShift {
        pub fn from_time() -> Self {
            match SystemTime::now().elapsed() {
                Ok(elapsed) => Self {
                    seed: elapsed.as_millis() as u64,
                },
                Err(e) => {
                    panic!("{}", e);
                }
            }
        }
        pub fn with_seed(seed: u64) -> Self {
            Self { seed }
        }
        #[inline]
        pub fn rand(&mut self, m: u64) -> u64 {
            self.next().unwrap() % m
        }
        #[inline]
        pub fn rand_range<R: RangeBounds<i64>>(&mut self, range: R) -> i64 {
            let (l, r) = range.to_lr();
            let k = self.next().unwrap() as i64;
            k.rem_euclid(r - l) + l
        }
        #[inline]
        pub fn randf(&mut self) -> f64 {
            const UPPER_MASK: u64 = 0x3FF0000000000000;
            const LOWER_MASK: u64 = 0xFFFFFFFFFFFFF;
            f64::from_bits(UPPER_MASK | (self.next().unwrap() & LOWER_MASK)) - 1.0
        }
        #[inline]
        pub fn shuffle<T>(&mut self, s: &mut [T]) {
            for i in 0..s.len() {
                s.swap(i, self.rand_range(i as i64..s.len() as i64) as usize);
            }
        }
    }
}

pub trait CumulativeSum {
    type Item;
    fn cumsum(self, initial: Self::Item) -> Vec<Self::Item>;
}
impl<T: Clone + Add<Output = T>, I: IntoIterator<Item = T>> CumulativeSum for I {
    type Item = I::Item;
    fn cumsum(self, initial: T) -> Vec<T> {
        let mut ret = vec![initial];
        for t in self {
            ret.push(ret[ret.len() - 1].clone() + t);
        }
        ret
    }
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
    pub rev: Vec<Option<usize>>,
}
mod impl_graph_adjacency_list {
    use super::{Debug, Formatter, Graph, GraphTrait, Index};
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
                rev: self.rev.clone(),
            }
        }
    }
    impl<W> Graph<W> {
        pub fn new(n: usize) -> Self {
            Self {
                n,
                edges: Vec::new(),
                index: vec![Vec::new(); n],
                rev_index: vec![Vec::new(); n],
                rev: Vec::new(),
            }
        }
        pub fn add_arc(&mut self, src: usize, dst: usize, w: W) -> usize {
            let i = self.edges.len();
            self.edges.push((src, dst, w));
            self.index[src].push(i);
            self.rev_index[dst].push(i);
            self.rev.push(None);
            i
        }
    }
    impl<W> Index<usize> for Graph<W> {
        type Output = (usize, usize, W);
        fn index(&self, index: usize) -> &Self::Output {
            &self.edges[index]
        }
    }
    impl<W: Clone> Graph<W> {
        pub fn add_edge(&mut self, src: usize, dst: usize, w: W) -> (usize, usize) {
            let i = self.add_arc(src, dst, w.clone());
            let j = self.add_arc(dst, src, w);
            self.rev.push(None);
            self.rev.push(None);
            self.rev[i] = Some(j);
            self.rev[j] = Some(i);
            (i, j)
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
}

pub fn sqrt(a: i64) -> (i64, i64) {
    let x = (a as f64).sqrt() as i64;
    match a.cmp(&(x * x)) {
        Ordering::Greater => (x, x + 1),
        Ordering::Less => (x - 1, x),
        Ordering::Equal => (x, x),
    }
}

pub fn solve<IO: ReaderTrait + WriterTrait>(io: &mut IO) {
    let (n, w, k, c) = io.v4::<usize, usize, usize, u64>();
    let ab = io.vec2::<usize, usize>(w);
    let cd = io.vec2::<usize, usize>(k);

    let mut solver = Solver::new(n, w, k, c, ab, cd);
    solver.solve(io);
}

#[derive(Clone, Copy, Debug)]
enum Cost {
    PowZero,
    PowOne,
    PowTwo,
    PowThree,
    PowFour,
    PowFive,
    PowSix,
    PowSeven,
}

impl Cost {
    fn new(c: u64) -> Self {
        use Cost::*;
        if c <= 1 {
            PowZero
        } else if c <= 2 {
            PowOne
        } else if c <= 4 {
            PowTwo
        } else if c <= 8 {
            PowThree
        } else if c <= 16 {
            PowFour
        } else if c <= 32 {
            PowFive
        } else if c <= 64 {
            PowSix
        } else {
            PowSeven
        }
    }

    fn c(&self) -> usize {
        use Cost::*;
        match self {
            PowZero => 1,
            PowOne => 2,
            PowTwo => 4,
            PowThree => 8,
            PowFour => 16,
            PowFive => 32,
            PowSix => 64,
            PowSeven => 128,
        }
    }
    // 1回あたりのパワー
    // todo 改善の余地あり
    // そもそもsが5000に近い値のところは掘りたくない
    // cが大きい場合は仕方ないこともあるかもしれない
    fn initial_power(&self) -> i64 {
        const TABLE: [i64; 8] = [20, 24, 28, 40, 60, 80, 100, 150];
        let k = 63 - self.c().leading_zeros() as usize;
        TABLE[k] as i64
    }

    // 試し掘りのグリッドサイズ
    fn grid_size(&self) -> usize {
        16
    }

    // 真っすぐ進むときに寄り道を考えるかどうか
    fn straight_threashold(&self) -> usize {
        16
    }

    // 試し掘りする1回あたりの深さ
    fn depth(&self) -> i64 {
        const TABLE: [i64; 8] = [14, 25, 30, 30, 50, 65, 120, 110];
        let k = 63 - self.c().leading_zeros() as usize;
        TABLE[k] as i64
    }

    // 試し掘りの分割回数
    fn div_cnt(&self) -> i64 {
        const TABLE: [i64; 8] = [10, 6, 5, 5, 3, 2, 1, 1];
        let k = 63 - self.c().leading_zeros() as usize;
        TABLE[k] as i64
    }

    fn spot_dist(&self, d: i64) -> i64 {
        use Cost::*;
        let d2 = d * d;
        let d3 = d2 * d;
        match self {
            PowZero => d3.saturating_mul(d2),
            PowOne => d3.saturating_mul(d2),
            PowTwo => d3.saturating_mul(d2),
            PowThree => d2.saturating_mul(d2),
            PowFour => d2.saturating_mul(d2),
            PowFive => d3,
            PowSix => d3,
            PowSeven => d3,
        }
    }

    fn minimum_unit(&self) -> i64 {
        8
    }

    fn dig_with_predict_coefficient(&self, upper_predict: bool) -> Vec<(i64, i64)> {
        use Cost::*;
        if upper_predict {
            match self {
                PowZero => vec![(9, 10), (1, 9)],
                PowOne => vec![(9, 10), (1, 8)],
                PowTwo => vec![(9, 10), (1, 7)],
                PowThree => vec![(9, 10), (1, 7)],
                PowFour => vec![(1, 1), (1, 6)],
                PowFive => vec![(1, 1), (1, 5)],
                PowSix => vec![(1, 1), (1, 4)],
                PowSeven => vec![(1, 1), (2, 5)],
            }
        } else {
            match self {
                PowZero => vec![(2, 3), (1, 9)],
                PowOne => vec![(2, 3), (1, 8)],
                PowTwo => vec![(2, 3), (1, 7)],
                PowThree => vec![(2, 3), (1, 7)],
                PowFour => vec![(3, 4), (1, 6)],
                PowFive => vec![(3, 4), (1, 5)],
                PowSix => vec![(3, 4), (1, 4)],
                PowSeven => vec![(3, 4), (2, 5)],
            }
        }
    }
}

struct Solver {
    n: usize,
    w: usize,
    k: usize,
    c: Cost,
    ab: Vec<(usize, usize)>,
    cd: Vec<(usize, usize)>,
    // 破壊したかどうか
    destructed: Vec<Vec<bool>>,
    // いままでにどれくらい掘ったか
    history: Vec<Vec<i64>>,
    // 各頂点との接続情報 id = n*nは水源そのもの
    watered: UnionFind,
    // 掘ろうとしている頂点
    dig_queue: Vec<Vec<bool>>,
}
impl Solver {
    fn new(
        n: usize,
        w: usize,
        k: usize,
        c: u64,
        ab: Vec<(usize, usize)>,
        cd: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            n,
            w,
            k,
            c: Cost::new(c),
            ab,
            cd,
            destructed: vec![vec![false; n]; n],
            history: vec![vec![0; n]; n],
            watered: UnionFind::new(n * n + 1),
            dig_queue: vec![vec![false; n]; n],
        }
    }
    fn solve<IO: ReaderTrait + WriterTrait>(&mut self, io: &mut IO) {
        // 100掘った場所にidを振っておく(union_find用)
        let mut spot = HashMap::default();
        let mut spots = Vec::new();

        let mut water_spot = Vec::new();
        for i in 0..self.w {
            let (c, d) = self.ab[i];
            spot.insert((c, d), spots.len());
            water_spot.push(spots.len());
            spots.push((c, d));
        }
        // 家は掘りぬいておく
        let mut house_spot = Vec::new();
        for house_i in 0..self.k {
            let (c, d) = self.cd[house_i];
            self.dig_all(c, d, 100, io);
            spot.insert((c, d), spots.len());
            house_spot.push(spots.len());
            spots.push((c, d));
        }

        let grid_size = self.c.grid_size();
        for x in 0..self.n / grid_size {
            for y in 0..self.n / grid_size {
                let (y, x) = (y * grid_size + grid_size / 2, x * grid_size + grid_size / 2);
                let mut min_dist = 1 << 50;
                for &(y2, x2) in &spots {
                    chmin!(min_dist, dist_m(y, x, y2, x2));
                }
                if min_dist < 4 || min_dist > 150 {
                    continue;
                }
                let div = self.c.div_cnt();
                let p = self.c.depth();
                for _ in 1..div {
                    self.dig_once(y, x, p, io);
                }
                if self.dig_once(y, x, p, io) > 0 {
                    if spot.insert((y, x), spots.len()).is_none() {
                        spots.push((y, x));
                    }
                }
            }
        }

        let mut graph = Graph::new(spots.len());
        for i in 0..spots.len() {
            for j in i + 1..spots.len() {
                let (y1, x1) = spots[i];
                let (y2, x2) = spots[j];
                let d = self.c.spot_dist(dist_e(y1, x1, y2, x2));
                graph.add_edge(i, j, (d, self.history[y1][x1]));
            }
        }

        for i in 1..water_spot.len() {
            graph.add_edge(water_spot[i - 1], water_spot[i], (0, 0));
        }
        let mut orig_dist = vec![(1i64 << 60, 1i64 << 60); graph.size()];
        for i in 0..water_spot.len() {
            orig_dist[water_spot[i]] = (0, 0);
        }
        let mut prev = vec![graph.size(); graph.size()];

        // 距離更新キュー
        let mut heap = BinaryHeap::new();

        // 家ごとに掘る
        for _ in 0..self.k {
            self.dig_queue = vec![vec![false; self.n]; self.n];
            for i in 0..graph.size() {
                if orig_dist[i] == (0, 0) {
                    heap.push((Reverse((0, 0)), i));
                }
            }
            while let Some((Reverse((d1, d2)), src)) = heap.pop() {
                if orig_dist[src] != (d1, d2) {
                    continue;
                }
                graph.edges(src).into_iter().for_each(|(dst, (w1, w2))| {
                    if chmin!(
                        orig_dist[dst],
                        (
                            orig_dist[src].0.saturating_add(w1),
                            orig_dist[src].1.saturating_add(w2)
                        )
                    ) {
                        prev[dst] = src;
                        heap.push((Reverse(orig_dist[dst]), dst))
                    }
                });
            }
            let (mut min_dist, mut house) = (std::i64::MAX, 0);
            for i in 0..house_spot.len() {
                if orig_dist[house_spot[i]].0 > 0 && chmin!(min_dist, orig_dist[house_spot[i]].0) {
                    house = house_spot[i];
                }
            }
            let start_house = house;

            while prev[house] != graph.size() {
                let (a, b) = spots[house];
                let (c, d) = spots[prev[house]];

                orig_dist[house] = (0, 0);
                self.dig_ab_to_cd(a, b, c, d, io);
                heap.push((Reverse((0, 0)), house));
                house = prev[house];
            }

            let (sy, sx) = spots[start_house];
            let mut dist = vec![vec![1 << 30; self.n]; self.n];
            let mut prev = vec![vec![(self.n, self.n); self.n]; self.n];
            let mut q = VecDeque::new();
            q.push_back((sy, sx));
            dist[sy][sx] = 0;
            let (mut ty, mut tx) = (self.n, self.n);
            // for y in 0..self.n {
            //     eprintln!(
            //         "{}",
            //         self.dig_queue[y]
            //             .iter()
            //             .map(|&b| if b { "T" } else { "F" })
            //             .collect::<String>()
            //     );
            // }
            'search: while let Some((y, x)) = q.pop_front() {
                for w in &water_spot {
                    let (wy, wx) = spots[*w];
                    if wy == y && wx == x {
                        ty = y;
                        tx = x;
                        break 'search;
                    }
                }
                if self.is_watered(y, x) {
                    ty = y;
                    tx = x;
                    break;
                }
                let mut v = Vec::new();
                if y > 0 {
                    v.push((y - 1, x));
                }
                if y + 1 < self.n {
                    v.push((y + 1, x));
                }
                if x > 0 {
                    v.push((y, x - 1));
                }
                if x + 1 < self.n {
                    v.push((y, x + 1));
                }
                for (ty, tx) in v {
                    if (self.dig_queue[ty][tx] || self.destructed[ty][tx])
                        && chmin!(dist[ty][tx], dist[y][x] + 1)
                    {
                        q.push_back((ty, tx));
                        prev[ty][tx] = (y, x);
                    }
                }
            }
            let mut path = Vec::new();
            while ty != self.n && tx != self.n {
                path.push((ty, tx));
                let (y, x) = prev[ty][tx];
                ty = y;
                tx = x;
            }
            path.reverse();
            let (mut last, mut upper) = (self.history[path[0].0][path[0].1], false);
            for i in 1..path.len() {
                let (y, x) = path[i];
                if self.dig_with_predict(y, x, last, upper, io) == 2 {
                    return;
                }
                upper = self.history[y][x] > last;
                last = self.history[y][x];
            }

            let (c, d) = spots[house];
            self.watered.unite(self.key(c, d), self.n * self.n);
        }
    }

    // returns stat
    fn dig_ab_to_cd<IO: ReaderTrait + WriterTrait>(
        &mut self,
        mut a: usize,
        mut b: usize,
        c: usize,
        d: usize,
        io: &mut IO,
    ) -> i64 {
        if self.is_watered(a, b) && self.is_watered(c, d) {
            return 1;
        }
        self.dig_queue[a][b] = true;
        let dist = dist_m(a, b, c, d);
        let p = min!(
            self.c.initial_power(),
            if dist <= 10 {
                50
            } else if dist <= 20 {
                100
            } else if dist <= 30 {
                200
            } else {
                300
            }
        );
        if a == c {
            let k = d as i64 - b as i64;
            let dx = if k > 0 { 1i64 } else { -1 };
            let mut vs = vec![(a, (b + d) / 2)];
            let th = self.c.straight_threashold();
            if k.abs() > th as i64 {
                if a + th / 2 < self.n {
                    vs.push((a + th / 2, (b + d) / 2));
                }
                if a >= th / 2 {
                    vs.push((a - th / 2, (b + d) / 2));
                }
            }
            let (y2, x2) = self.dig_comp(&vs, p, io);
            if y2 == a {
                while b != d {
                    self.dig_queue[a][b] = true;
                    b = (b as i64 + dx) as usize;
                }
            } else {
                if self.dig_ab_to_cd(a, b, y2, x2, io) == 2 {
                    return 2;
                }
                if self.dig_ab_to_cd(y2, x2, c, d, io) == 2 {
                    return 2;
                }
            }
            self.dig_queue[a][b] = true;
        } else if b == d {
            let k = c as i64 - a as i64;
            let dy = if k > 0 { 1i64 } else { -1 };
            let mut vs = vec![((a + c) / 2, b)];
            let th = self.c.straight_threashold();
            if k.abs() > th as i64 {
                if b + th / 2 < self.n {
                    vs.push(((a + c) / 2, b + th / 2));
                }
                if b >= th / 2 {
                    vs.push(((a + c) / 2, b - th / 2));
                }
            }
            let (y2, x2) = self.dig_comp(&vs, p, io);
            if x2 == b {
                while a != c {
                    self.dig_queue[a][b] = true;
                    a = (a as i64 + dy) as usize;
                }
            } else {
                if self.dig_ab_to_cd(a, b, y2, x2, io) == 2 {
                    return 2;
                }
                if self.dig_ab_to_cd(y2, x2, c, d, io) == 2 {
                    return 2;
                }
            }

            self.dig_queue[a][b] = true;
        } else {
            // (a, d) と (c, b)の小さいほうを通りたい
            // (a+c/2, b+d/2)を通ることも考える
            let mut vs = vec![(a, d), (c, b)];
            if min!((a as i64 - c as i64).abs(), (b as i64 - d as i64).abs())
                > self.c.minimum_unit()
            {
                vs.push(((a + c) / 2, (b + d) / 2))
            }
            let (y2, x2) = self.dig_comp(&vs, p, io);
            if self.dig_ab_to_cd(a, b, y2, x2, io) == 2 {
                return 2;
            }
            if self.dig_ab_to_cd(y2, x2, c, d, io) == 2 {
                return 2;
            }
        }
        1
    }

    fn dig_comp<IO: ReaderTrait + WriterTrait>(
        &mut self,
        yx: &[(usize, usize)],
        initial_p: i64,
        io: &mut IO,
    ) -> (usize, usize) {
        let mut expected_total = initial_p;
        loop {
            for &(y, x) in yx {
                if self.history[y][x] < expected_total {
                    if self.dig_once(y, x, expected_total - self.history[y][x], io) != 0 {
                        return (y, x);
                    }
                }
            }
            expected_total += initial_p / 2
        }
    }

    fn dig_with_predict<IO: ReaderTrait + WriterTrait>(
        &mut self,
        y: usize,
        x: usize,
        predict: i64,
        upper_predict: bool,
        io: &mut IO,
    ) -> i64 {
        let coefficient = self.c.dig_with_predict_coefficient(upper_predict);
        let mut r = self.dig_once(y, x, predict * coefficient[0].0 / coefficient[0].1, io);
        if r != 0 {
            return r;
        }

        while r == 0 {
            let p = max!(predict * coefficient[1].0 / coefficient[1].1, 3);
            r = self.dig_once(y, x, p, io);
        }
        r
    }

    /// dig_all -> (終了ステータス, 掘るのにかかったパワー)
    fn dig_all<IO: ReaderTrait + WriterTrait>(
        &mut self,
        y: usize,
        x: usize,
        initial_p: i64,
        io: &mut IO,
    ) -> i64 {
        let mut r = self.dig_once(y, x, initial_p, io);

        if r != 0 {
            return r;
        }

        while r == 0 {
            r = self.dig_once(y, x, initial_p / 2, io);
        }
        r
    }

    /// dig_once -> (終了ステータス)
    /// パワーpで一回掘る
    fn dig_once<IO: ReaderTrait + WriterTrait>(
        &mut self,
        y: usize,
        x: usize,
        p: i64,
        io: &mut IO,
    ) -> i64 {
        if self.destructed[y][x] {
            return 1;
        }
        let p = min!(5000 - self.history[y][x], p);
        io.out(format!("{} {} {}\n", y, x, p));
        io.flush();
        let r = io.v();
        self.history[y][x] += p;
        if r > 0 {
            self.destructed[y][x] = true;
            self.update_dist(y, x);
        }
        r
    }

    fn key(&self, y: usize, x: usize) -> usize {
        y * self.n + x
    }

    fn is_watered(&mut self, y: usize, x: usize) -> bool {
        self.watered.same(self.n * self.n, self.key(y, x))
    }

    fn update_dist(&mut self, y: usize, x: usize) {
        if !self.destructed[y][x] {
            return;
        }
        if y > 0 && self.destructed[y - 1][x] {
            self.watered.unite(self.key(y, x), self.key(y - 1, x));
        }
        if y + 1 < self.n && self.destructed[y + 1][x] {
            self.watered.unite(self.key(y, x), self.key(y + 1, x));
        }
        if x > 0 && self.destructed[y][x - 1] {
            self.watered.unite(self.key(y, x), self.key(y, x - 1));
        }
        if x + 1 < self.n && self.destructed[y][x + 1] {
            self.watered.unite(self.key(y, x), self.key(y, x + 1));
        }
    }
}

// マンハッタン距離
fn dist_m(y1: usize, x1: usize, y2: usize, x2: usize) -> i64 {
    let (y1, y2, x1, x2) = (y1 as i64, y2 as i64, x1 as i64, x2 as i64);
    (y1 - y2).abs() + (x1 - x2).abs()
}
// ユークリッド距離
#[allow(dead_code)]
fn dist_e(y1: usize, x1: usize, y2: usize, x2: usize) -> i64 {
    let dy = (y1 as i64 - y2 as i64).abs();
    let dx = (x1 as i64 - x2 as i64).abs();
    sqrt(dx * dx + dy * dy).0
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test() {
        let mut io = IO::default();
        let (n, w, k, c) = io.v4::<usize, usize, usize, i64>();
        let mut s = io.matrix::<i64>(n, n);
        let ab = io.vec2::<usize, usize>(w);
        let cd = io.vec2::<usize, usize>(k);
        std::thread::Builder::new()
            .name("extend stack size".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || {
                let init = format!(
                    "{} {} {} {}\n{}\n{}\n",
                    n,
                    w,
                    k,
                    c,
                    ab.iter().map(|(a, b)| format!("{} {}", a, b)).join(" "),
                    cd.iter().map(|(c, d)| format!("{} {}", c, d)).join(" ")
                );
                let mut score = 0;
                let mut exit = false;
                let mut io = IODebug::new(
                    init.as_str(),
                    true,
                    |outer: &mut ReaderFromStr, inner: &mut ReaderFromStr| {
                        if exit {
                            return;
                        }
                        let (y, x, p) = outer.v3::<usize, usize, i64>();
                        assert!(
                            s[y][x] > 0,
                            "invalid operation: ({}, {}) has already been destructed",
                            y,
                            x
                        );
                        score += p + c;
                        s[y][x] -= p;
                        let mut res = usize::from(s[y][x] <= 0);

                        if res > 0 {
                            if is_finish(n, w, k, &ab, &cd, &s) {
                                res = 2;
                                exit = true;
                            }
                        }
                        inner.out(res.ln());
                    },
                );
                solve(&mut io);
                io.flush();
            })
            .unwrap()
            .join()
            .unwrap()
    }

    fn is_finish(
        n: usize,
        w: usize,
        k: usize,
        ab: &Vec<(usize, usize)>,
        cd: &Vec<(usize, usize)>,
        s: &Vec<Vec<i64>>,
    ) -> bool {
        let mut dist = vec![vec![false; n]; n];
        let mut q = VecDeque::new();
        for i in 0..w {
            let (a, b) = ab[i];
            if s[a][b] <= 0 {
                q.push_back((a, b));
                dist[a][b] = true;
            }
        }
        while let Some((y, x)) = q.pop_front() {
            if y > 0 && s[y - 1][x] <= 0 && chmax!(dist[y - 1][x], true) {
                q.push_back((y - 1, x));
            }
            if y < n - 1 && s[y + 1][x] <= 0 && chmax!(dist[y + 1][x], true) {
                q.push_back((y + 1, x));
            }
            if x > 0 && s[y][x - 1] <= 0 && chmax!(dist[y][x - 1], true) {
                q.push_back((y, x - 1));
            }
            if x < n - 1 && s[y][x + 1] <= 0 && chmax!(dist[y][x + 1], true) {
                q.push_back((y, x + 1));
            }
        }
        for i in 0..k {
            let (c, d) = cd[i];
            if !dist[c][d] {
                return false;
            }
        }
        true
    }
}
