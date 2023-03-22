# [rustfmt :: skip ] pub use string_util_impl :: {AddLineTrait , BitsTrait , JoinTrait } ;
# [rustfmt :: skip ] mod string_util_impl {use super :: {Display , Integral } ; pub trait AddLineTrait {fn line (& self ) -> String ; } impl < D : Display > AddLineTrait for D {fn line (& self ) -> String {self . to_string () + "\n" } } pub trait JoinTrait {fn join (self , separator : & str ) -> String ; } impl < D : Display , I : IntoIterator < Item = D > > JoinTrait for I {fn join (self , separator : & str ) -> String {let mut buf = String :: new () ; self . into_iter () . fold ("" , | sep , arg | {buf . push_str (& format ! ("{}{}" , sep , arg ) ) ; separator } ) ; buf } } pub trait BitsTrait {fn bits (self , length : Self ) -> String ; } impl < I : Integral > BitsTrait for I {fn bits (self , length : Self ) -> String {let mut buf = String :: new () ; let mut i = I :: zero () ; while i < length {buf . push_str (& format ! ("{}" , self >> i & I :: one () ) ) ; i += I :: one () ; } buf + "\n" } } }
# [rustfmt :: skip ] pub trait ToBounds < T > {fn lr (& self ) -> (T , T ) ; }
# [rustfmt :: skip ] impl < R : RangeBounds < T > + Clone , T : Copy + BoundedAbove + BoundedBelow + One + Add < Output = T > > ToBounds < T > for R {# [inline ] fn lr (& self ) -> (T , T ) {use Bound :: {Excluded , Included , Unbounded } ; let l = match self . start_bound () {Unbounded => T :: min_value () , Included (& s ) => s , Excluded (& s ) => s + T :: one () , } ; let r = match self . end_bound () {Unbounded => T :: max_value () , Included (& e ) => e + T :: one () , Excluded (& e ) => e , } ; (l , r ) } }
# [rustfmt :: skip ] pub trait RangeProduct < I > {type Magma : Magma ; fn product < R : ToBounds < I > > (& self , range : R ) -> < Self :: Magma as Magma > :: M ; }
# [rustfmt :: skip ] pub trait RangeProductMut < I > {type Magma : Magma ; fn product < R : ToBounds < I > > (& mut self , range : R ) -> < Self :: Magma as Magma > :: M ; }
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

pub use segment_tree_impl::SegmentTree;
mod segment_tree_impl {
    use super::{Index, Monoid, RangeProduct, ToBounds};
    #[derive(Clone, Debug)]
    pub struct SegmentTree<M: Monoid> {
        n: usize,
        node: Vec<M::M>,
    }
    impl<M: Monoid> From<&[M::M]> for SegmentTree<M> {
        fn from(v: &[M::M]) -> Self {
            let mut segtree = Self::new(v.len());
            segtree.node[segtree.n..segtree.n + v.len()].clone_from_slice(v);
            for i in (1..segtree.n).rev() {
                segtree.node[i] = M::op(&segtree.node[i << 1], &segtree.node[i << 1 | 1]);
            }
            segtree
        }
    }
    impl<M: Monoid> RangeProduct<usize> for SegmentTree<M> {
        type Magma = M;
        fn product<R: ToBounds<usize>>(&self, range: R) -> M::M {
            let (mut l, mut r) = range.lr();
            l += self.n;
            r += self.n;
            let (mut sum_l, mut sum_r) = (M::unit(), M::unit());
            while l < r {
                if l & 1 != 0 {
                    sum_l = M::op(&sum_l, &self.node[l]);
                    l += 1;
                }
                if r & 1 != 0 {
                    r -= 1;
                    sum_r = M::op(&self.node[r], &sum_r);
                }
                l >>= 1;
                r >>= 1;
            }
            M::op(&sum_l, &sum_r)
        }
    }
    impl<M: Monoid> SegmentTree<M> {
        pub fn new(n: usize) -> Self {
            let node = vec![M::unit(); n << 1];
            let mut segtree = Self { n, node };
            for i in (1..segtree.n).rev() {
                segtree.node[i] = M::op(&segtree.node[i << 1], &segtree.node[i << 1 | 1]);
            }
            segtree
        }
        pub fn update_at(&mut self, mut i: usize, value: M::M) {
            i += self.n;
            self.node[i] = value;
            while i > 0 {
                i >>= 1;
                self.node[i] = M::op(&self.node[i << 1], &self.node[i << 1 | 1]);
            }
        }
        fn top_nodes(&self, l: usize, r: usize) -> Vec<usize> {
            let (mut l, mut r) = (l + self.n, r + self.n);
            let (mut l_nodes, mut r_nodes) = (Vec::new(), Vec::new());
            while l < r {
                if l & 1 != 0 {
                    l_nodes.push(l);
                    l += 1;
                }
                if r & 1 != 0 {
                    r -= 1;
                    r_nodes.push(r);
                }
                l >>= 1;
                r >>= 1;
            }
            r_nodes.reverse();
            l_nodes.append(&mut r_nodes);
            l_nodes
        }
        pub fn upper_bound<F: Fn(&M::M) -> bool>(&self, l: usize, f: F) -> Option<usize> {
            if f(&M::unit()) {
                return Some(l);
            }
            let top_nodes = self.top_nodes(l, self.n);
            let mut cur = M::unit();
            for mut top in top_nodes {
                let t = M::op(&cur, &self.node[top]);
                if !f(&t) {
                    cur = t;
                } else {
                    while top < self.n {
                        top <<= 1;
                        let t = M::op(&cur, &self.node[top]);
                        if !f(&t) {
                            cur = t;
                            top += 1;
                        }
                    }
                    if !f(&cur) {
                        cur = M::op(&cur, &self.node[top]);
                        top += 1;
                    }
                    assert!(f(&cur));
                    return Some(top - self.n);
                }
            }
            None
        }
        pub fn lower_bound<F: Fn(&M::M) -> bool>(&self, r: usize, f: F) -> Option<usize> {
            if f(&M::unit()) {
                return Some(r);
            }
            let top_nodes = self.top_nodes(0, r);
            let mut cur = M::unit();
            for mut top in top_nodes.into_iter().rev() {
                let t = M::op(&self.node[top], &cur);
                if !f(&t) {
                    cur = t;
                } else {
                    while top < self.n {
                        top <<= 1;
                        let t = M::op(&self.node[top], &cur);
                        if f(&t) {
                            top += 1;
                        } else {
                            cur = t;
                        }
                    }
                    return Some(top - self.n);
                }
            }
            None
        }
    }
    impl<M: Monoid> Index<usize> for SegmentTree<M> {
        type Output = M::M;
        fn index(&self, i: usize) -> &M::M {
            &self.node[i + self.n]
        }
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

#[derive(Clone, Debug, Default)]
pub struct Addition<S>(PhantomData<fn() -> S>);
mod addition_impl {
    use super::{
        Add, Addition, Associative, Commutative, Debug, Invertible, Magma, Neg, Unital, Zero,
    };
    impl<S: Clone + Debug + Add<Output = S> + PartialEq> Magma for Addition<S> {
        type M = S;
        fn op(x: &S, y: &S) -> S {
            x.clone() + y.clone()
        }
    }
    impl<S: Clone + Debug + Add<Output = S> + PartialEq> Associative for Addition<S> {}
    impl<S: Clone + Debug + Add<Output = S> + PartialEq + Zero> Unital for Addition<S> {
        fn unit() -> S {
            S::zero()
        }
    }
    impl<S: Clone + Debug + Add<Output = S> + PartialEq> Commutative for Addition<S> {}
    impl<S: Clone + Debug + Add<Output = S> + PartialEq + Neg<Output = S>> Invertible for Addition<S> {
        fn inv(x: &S) -> S {
            x.clone().neg()
        }
    }
}

#[derive(Clone, Debug)]
pub struct HLDecomposition {
    graph: Vec<Vec<usize>>,
    _root: usize,
    size: Vec<usize>,
    in_time: Vec<usize>,
    rev: Vec<usize>,
    out_time: Vec<usize>,
    head: Vec<usize>,
    parent: Vec<usize>,
    depth: Vec<usize>,
    edge: bool,
}
impl HLDecomposition {
    pub fn build<G: GraphTrait>(g: &G, root: usize) -> Self {
        let mut this = Self {
            graph: vec![Vec::new(); g.size()],
            _root: root,
            size: vec![1; g.size()],
            in_time: vec![0; g.size()],
            rev: vec![0; g.size()],
            out_time: vec![0; g.size()],
            head: vec![0; g.size()],
            parent: vec![root; g.size()],
            depth: vec![0; g.size()],
            edge: true,
        };
        this.build_graph(g, root, root);
        this.dfs_size(root, root);
        this.dfs_hld(root, root, &mut 0);
        this
    }
    fn build_graph<G: GraphTrait>(&mut self, g: &G, src: usize, par: usize) {
        for (dst, _w) in g.edges(src) {
            if dst == par {
                continue;
            }
            self.graph[src].push(dst);
            self.build_graph(g, dst, src);
        }
    }
    fn dfs_size(&mut self, src: usize, par: usize) {
        self.parent[src] = par;
        for dst_i in 0..self.graph[src].len() {
            let dst = self.graph[src][dst_i];
            if dst == par {
                continue;
            }
            self.depth[dst] = self.depth[src] + 1;
            self.dfs_size(dst, src);
            self.size[src] += self.size[dst];
            if self.size[dst] > self.size[self.graph[src][0]] {
                self.graph[src].swap(0, dst_i);
            }
        }
    }
    fn dfs_hld(&mut self, src: usize, par: usize, times: &mut usize) {
        self.in_time[src] = *times;
        self.rev[self.in_time[src]] = src;
        *times += 1;
        for dst in self.graph[src].clone() {
            if dst == par {
                continue;
            }
            self.head[dst] = if self.graph[src][0] == dst {
                self.head[src]
            } else {
                dst
            };
            self.dfs_hld(dst, src, times);
        }
        self.out_time[src] = *times;
    }
    pub fn la(&self, mut v: usize, mut k: usize) -> usize {
        loop {
            let u = self.head[v];
            if self.in_time[v] - k >= self.in_time[u] {
                return self.rev[self.in_time[v] - k];
            }
            k -= self.in_time[v] - self.in_time[u] + 1;
            v = self.parent[u];
        }
    }
    pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
        loop {
            if self.in_time[u] > self.in_time[v] {
                swap(&mut u, &mut v);
            }
            if self.head[u] == self.head[v] {
                return u;
            }
            v = self.parent[self.head[v]];
        }
    }
    pub fn dist(&self, u: usize, v: usize) -> usize {
        self.depth[u] + self.depth[v] - 2 * self.depth[self.lca(u, v)]
    }
    pub fn path_to_ranges(&self, mut u: usize, mut v: usize) -> Vec<Range<usize>> {
        let mut ret = Vec::new();
        while self.head[u] != self.head[v] {
            if self.in_time[self.head[u]] > self.in_time[self.head[v]] {
                swap(&mut u, &mut v);
            }
            ret.push(self.in_time[self.head[v]]..self.in_time[v] + 1);
            v = self.parent[self.head[v]];
        }
        if self.in_time[u] > self.in_time[v] {
            swap(&mut u, &mut v)
        }
        ret.push(self.in_time[u] + usize::from(self.edge)..self.in_time[v] + 1);
        ret
    }
    pub fn subtree_to_range(&self, v: usize) -> Range<usize> {
        self.in_time[v]..self.out_time[v]
    }
}

pub fn solve<IO: ReaderTrait + WriterTrait>(io: &mut IO) {
    let n = io.v::<usize>();
    let uvw = io.vec3::<usize, usize, i64>(n - 1);
    let mut graph = Graph::new(n);

    for &(u, v, w) in &uvw {
        graph.add_edge(u - 1, v - 1, w);
    }
    let hld = HLDecomposition::build(&graph, 0);
    let mut segtree = SegmentTree::<Addition<i64>>::new(n * 5);
    for &(u, v, w) in &uvw {
        let d = hld.path_to_ranges(u - 1, v - 1);
        dbg!(&d);
        for r in d {
            let (l, r) = r.lr();
            if r > l {
                segtree.update_at(l, w);
            }
        }
    }
    for _ in 0..io.v() {
        if 1 == io.v() {
            let (i, x) = io.v2::<usize, i64>();
            let (u, v, _) = uvw[i - 1];
            for r in hld.path_to_ranges(u - 1, v - 1) {
                let (l, r) = r.lr();
                if r > l {
                    segtree.update_at(l, x);
                }
            }
        } else {
            let (u, v) = io.v2::<usize, usize>();
            let mut ans = 0;
            for r in hld.path_to_ranges(u - 1, v - 1) {
                ans += segtree.product(r);
            }
            io.out(ans.line())
        }
    }
}

#[test]
fn test() {
    let test_suits = vec![
        "5
    1 2 3
    1 3 6
    1 4 9
    4 5 10
    4
    2 2 3
    2 1 5
    1 3 1
    2 1 5
    ",
        "7
    1 2 1000000000
    2 3 1000000000
    3 4 1000000000
    4 5 1000000000
    5 6 1000000000
    6 7 1000000000
    3
    2 1 6
    1 1 294967296
    2 1 6
    ",
        "1
    1
    2 1 1
    ",
        "8
    1 2 105
    1 3 103
    2 4 105
    2 5 100
    5 6 101
    3 7 106
    3 8 100
    18
    2 2 8
    2 3 6
    1 4 108
    2 3 4
    2 3 5
    2 5 5
    2 3 1
    2 4 3
    1 1 107
    2 3 1
    2 7 6
    2 3 8
    2 1 5
    2 7 6
    2 4 7
    2 1 7
    2 5 3
    2 8 6
    ",
    ];
    for suit in test_suits {
        std::thread::Builder::new()
            .name("extend stack size".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || {
                let mut io = IODebug::new(
                    suit,
                    true,
                    |_outer: &mut ReaderFromStr, _inner: &mut ReaderFromStr| {},
                );
                solve(&mut io);
                io.flush();
            })
            .unwrap()
            .join()
            .unwrap()
    }
}
