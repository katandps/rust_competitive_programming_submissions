# [rustfmt :: skip ] pub struct Writer < W : Write > {writer : BufWriter < W > , }
# [rustfmt :: skip ] impl < W : Write > Writer < W > {pub fn new (write : W ) -> Self {Self {writer : BufWriter :: new (write ) , } } pub fn ln < S : Display > (& mut self , s : S ) {writeln ! (self . writer , "{}" , s ) . expect ("Failed to write." ) } pub fn out < S : Display > (& mut self , s : S ) {write ! (self . writer , "{}" , s ) . expect ("Failed to write." ) } pub fn join < S : Display > (& mut self , v : & [S ] , separator : & str ) {v . iter () . fold ("" , | sep , arg | {write ! (self . writer , "{}{}" , sep , arg ) . expect ("Failed to write." ) ; separator } ) ; writeln ! (self . writer ) . expect ("Failed to write." ) ; } pub fn bits (& mut self , i : i64 , len : usize ) {(0 .. len ) . for_each (| b | write ! (self . writer , "{}" , i >> b & 1 ) . expect ("Failed to write." ) ) ; writeln ! (self . writer ) . expect ("Failed to write." ) } pub fn flush (& mut self ) {let _ = self . writer . flush () ; } }
pub use reader_impl::{ReaderFromStdin, ReaderFromString, ReaderTrait};
# [rustfmt :: skip ] mod reader_impl {use super :: {stdin , BufRead , FromStr as FS , VecDeque } ; pub trait ReaderTrait {fn next (& mut self ) -> Option < String > ; fn v < T : FS > (& mut self ) -> T {let s = self . next () . expect ("Insufficient input." ) ; s . parse () . ok () . expect ("Failed to parse." ) } fn v2 < T1 : FS , T2 : FS > (& mut self ) -> (T1 , T2 ) {(self . v () , self . v () ) } fn v3 < T1 : FS , T2 : FS , T3 : FS > (& mut self ) -> (T1 , T2 , T3 ) {(self . v () , self . v () , self . v () ) } fn v4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 ) {(self . v () , self . v () , self . v () , self . v () ) } fn v5 < T1 : FS , T2 : FS , T3 : FS , T4 : FS , T5 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 , T5 ) {(self . v () , self . v () , self . v () , self . v () , self . v () ) } fn vec < T : FS > (& mut self , length : usize ) -> Vec < T > {(0 .. length ) . map (| _ | self . v () ) . collect () } fn vec2 < T1 : FS , T2 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 ) > {(0 .. length ) . map (| _ | self . v2 () ) . collect () } fn vec3 < T1 : FS , T2 : FS , T3 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 ) > {(0 .. length ) . map (| _ | self . v3 () ) . collect () } fn vec4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 , T4 ) > {(0 .. length ) . map (| _ | self . v4 () ) . collect () } fn chars (& mut self ) -> Vec < char > {self . v :: < String > () . chars () . collect () } fn split (& mut self , zero : u8 ) -> Vec < usize > {self . v :: < String > () . chars () . map (| c | (c as u8 - zero ) as usize ) . collect () } fn digits (& mut self ) -> Vec < usize > {self . split (b'0' ) } fn lowercase (& mut self ) -> Vec < usize > {self . split (b'a' ) } fn uppercase (& mut self ) -> Vec < usize > {self . split (b'A' ) } fn char_map (& mut self , h : usize ) -> Vec < Vec < char > > {(0 .. h ) . map (| _ | self . chars () ) . collect () } fn bool_map (& mut self , h : usize , ng : char ) -> Vec < Vec < bool > > {self . char_map (h ) . iter () . map (| v | v . iter () . map (| & c | c != ng ) . collect () ) . collect () } fn matrix < T : FS > (& mut self , h : usize , w : usize ) -> Vec < Vec < T > > {(0 .. h ) . map (| _ | self . vec (w ) ) . collect () } } pub struct ReaderFromString {buf : VecDeque < String > , } # [derive (Clone , Debug , Default ) ] pub struct ReaderFromStdin {buf : VecDeque < String > , } impl ReaderTrait for ReaderFromString {fn next (& mut self ) -> Option < String > {self . buf . pop_front () } } impl ReaderFromString {pub fn new (src : & str ) -> Self {Self {buf : src . split_whitespace () . map (ToString :: to_string ) . collect :: < Vec < String > > () . into () , } } pub fn from_file (path : & str ) -> Result < Self , Box < dyn std :: error :: Error > > {Ok (Self :: new (& std :: fs :: read_to_string (path ) ? ) ) } } impl ReaderTrait for ReaderFromStdin {fn next (& mut self ) -> Option < String > {while self . buf . is_empty () {let stdin = stdin () ; let mut reader = stdin . lock () ; let mut l = String :: new () ; reader . read_line (& mut l ) . unwrap () ; self . buf . append (& mut l . split_whitespace () . map (ToString :: to_string ) . collect () ) ; } self . buf . pop_front () } } }
# [rustfmt :: skip ] pub trait ToLR < T > {fn to_lr (& self ) -> (T , T ) ; }
# [rustfmt :: skip ] impl < R : RangeBounds < T > , T : Copy + BoundedAbove + BoundedBelow + One + Add < Output = T > > ToLR < T > for R {fn to_lr (& self ) -> (T , T ) {use Bound :: {Excluded , Included , Unbounded } ; let l = match self . start_bound () {Unbounded => T :: min_value () , Included (& s ) => s , Excluded (& s ) => s + T :: one () , } ; let r = match self . end_bound () {Unbounded => T :: max_value () , Included (& e ) => e + T :: one () , Excluded (& e ) => e , } ; (l , r ) } }
# [rustfmt :: skip ] pub use std :: {cmp :: {max , min , Ordering , Reverse } , collections :: {hash_map :: RandomState , BTreeMap , BTreeSet , BinaryHeap , VecDeque , } , convert :: Infallible , convert :: {TryFrom , TryInto } , fmt :: {Debug , Display , Formatter } , hash :: {Hash , BuildHasherDefault , Hasher } , io :: {stdin , stdout , BufRead , BufWriter , Read , Write } , iter :: {repeat , Product , Sum } , marker :: PhantomData , mem :: swap , ops :: {Add , AddAssign , BitAnd , BitAndAssign , BitOr , BitOrAssign , BitXor , BitXorAssign , Bound , Deref , DerefMut , Div , DivAssign , Index , IndexMut , Mul , MulAssign , Neg , Not , Range , RangeBounds , Rem , RemAssign , Shl , ShlAssign , Shr , ShrAssign , Sub , SubAssign , } , str :: {from_utf8 , FromStr } , } ;
#[allow(unused_macros)]
macro_rules ! chmin {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_min = min ! ($ ($ cmps ) ,+ ) ; if $ base > cmp_min {$ base = cmp_min ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! min {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ b } else {$ a } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = min ! ($ ($ rest ) ,+ ) ; if $ a > b {b } else {$ a } } } ; }
#[allow(unused_macros)]
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
# [rustfmt :: skip ] pub use self :: fxhasher_impl :: {FxHashMap as HashMap , FxHashSet as HashSet } ;
# [rustfmt :: skip ] mod fxhasher_impl {use super :: {BitXor , BuildHasherDefault , Hasher , TryInto } ; use std :: collections :: {HashMap , HashSet } ; pub struct FxHasher {hash : u64 , } type BuildHasher = BuildHasherDefault < FxHasher > ; pub type FxHashMap < K , V > = HashMap < K , V , BuildHasher > ; pub type FxHashSet < V > = HashSet < V , BuildHasher > ; const ROTATE : u32 = 5 ; const SEED : u64 = 0x51_7c_c1_b7_27_22_0a_95 ; impl Default for FxHasher {# [inline ] fn default () -> FxHasher {FxHasher {hash : 0 } } } impl Hasher for FxHasher {# [inline ] fn finish (& self ) -> u64 {self . hash as u64 } # [inline ] fn write (& mut self , mut bytes : & [u8 ] ) {while bytes . len () >= 8 {self . add_to_hash (u64 :: from_ne_bytes (bytes [.. 8 ] . try_into () . unwrap () ) ) ; bytes = & bytes [8 .. ] ; } while bytes . len () >= 4 {self . add_to_hash (u64 :: from (u32 :: from_ne_bytes (bytes [.. 4 ] . try_into () . unwrap () , ) ) ) ; bytes = & bytes [4 .. ] ; } while bytes . len () >= 2 {self . add_to_hash (u64 :: from (u16 :: from_ne_bytes (bytes [.. 2 ] . try_into () . unwrap () , ) ) ) ; } if let Some (& byte ) = bytes . first () {self . add_to_hash (u64 :: from (byte ) ) ; } } } impl FxHasher {# [inline ] pub fn add_to_hash (& mut self , i : u64 ) {self . hash = self . hash . rotate_left (ROTATE ) . bitxor (i ) . wrapping_mul (SEED ) ; } } }
#[allow(unused_macros)]
macro_rules ! dbg {($ ($ x : tt ) * ) => {{# [cfg (debug_assertions ) ] {std :: dbg ! ($ ($ x ) * ) } # [cfg (not (debug_assertions ) ) ] {($ ($ x ) * ) } } } }
# [rustfmt :: skip ] pub use algebra_traits :: {AbelianGroup , Associative , Band , BoundedAbove , BoundedBelow , Commutative , CommutativeMonoid , Group , Idempotent , Invertible , Magma , MapMonoid , Monoid , One , Pow , PrimitiveRoot , SemiGroup , Unital , Zero , } ;
# [rustfmt :: skip ] mod algebra_traits {use super :: Debug ; pub trait Magma {type M : Clone + PartialEq + Debug ; fn op (x : & Self :: M , y : & Self :: M ) -> Self :: M ; } pub trait Associative {} pub trait Unital : Magma {fn unit () -> Self :: M ; } pub trait Commutative : Magma {} pub trait Invertible : Magma {fn inv (x : & Self :: M ) -> Self :: M ; } pub trait Idempotent : Magma {} pub trait SemiGroup : Magma + Associative {} pub trait Monoid : Magma + Associative + Unital {fn pow (& self , x : Self :: M , mut n : usize ) -> Self :: M {let mut res = Self :: unit () ; let mut base = x ; while n > 0 {if n & 1 == 1 {res = Self :: op (& res , & base ) ; } base = Self :: op (& base , & base ) ; n >>= 1 ; } res } } pub trait CommutativeMonoid : Magma + Associative + Unital + Commutative {} pub trait Group : Magma + Associative + Unital + Invertible {} pub trait AbelianGroup : Magma + Associative + Unital + Commutative + Invertible {} pub trait Band : Magma + Associative + Idempotent {} impl < M : Magma + Associative > SemiGroup for M {} impl < M : Magma + Associative + Unital > Monoid for M {} impl < M : Magma + Associative + Unital + Commutative > CommutativeMonoid for M {} impl < M : Magma + Associative + Unital + Invertible > Group for M {} impl < M : Magma + Associative + Unital + Commutative + Invertible > AbelianGroup for M {} impl < M : Magma + Associative + Idempotent > Band for M {} pub trait MapMonoid {type Mono : Monoid ; type Func : Monoid ; fn op (& self , x : & < Self :: Mono as Magma > :: M , y : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M {Self :: Mono :: op (x , y ) } fn unit () -> < Self :: Mono as Magma > :: M {Self :: Mono :: unit () } fn apply (& self , f : & < Self :: Func as Magma > :: M , value : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M ; fn identity_map () -> < Self :: Func as Magma > :: M {Self :: Func :: unit () } fn compose (& self , f : & < Self :: Func as Magma > :: M , g : & < Self :: Func as Magma > :: M , ) -> < Self :: Func as Magma > :: M {Self :: Func :: op (f , g ) } } pub trait Zero {fn zero () -> Self ; } pub trait One {fn one () -> Self ; } pub trait BoundedBelow {fn min_value () -> Self ; } pub trait BoundedAbove {fn max_value () -> Self ; } pub trait Pow {fn pow (self , exp : i64 ) -> Self ; } pub trait PrimitiveRoot {const DIVIDE_LIMIT : usize ; fn primitive_root () -> Self ; } }
# [rustfmt :: skip ] pub trait Integral : 'static + Send + Sync + Copy + Ord + Display + Debug + Add < Output = Self > + Sub < Output = Self > + Mul < Output = Self > + Div < Output = Self > + Rem < Output = Self > + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + Product + BitOr < Output = Self > + BitAnd < Output = Self > + BitXor < Output = Self > + Not < Output = Self > + Shl < Output = Self > + Shr < Output = Self > + BitOrAssign + BitAndAssign + BitXorAssign + ShlAssign + ShrAssign + Zero + One + BoundedBelow + BoundedAbove {}
macro_rules ! impl_integral {($ ($ ty : ty ) ,* ) => {$ (impl Zero for $ ty {fn zero () -> Self {0 } } impl One for $ ty {fn one () -> Self {1 } } impl BoundedBelow for $ ty {fn min_value () -> Self {Self :: min_value () } } impl BoundedAbove for $ ty {fn max_value () -> Self {Self :: max_value () } } impl Integral for $ ty {} ) * } ; }
impl_integral!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
# [rustfmt :: skip ] pub fn main () {let stdout = stdout () ; std :: thread :: Builder :: new () . name ("extend stack size" . into () ) . stack_size (32 * 1024 * 1024 ) . spawn (move | | solve (ReaderFromStdin :: default () , Writer :: new (stdout . lock () ) ) ) . unwrap () . join () . unwrap () }

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

pub fn solve<R: ReaderTrait, W: Write>(mut reader: R, mut writer: Writer<W>) {
    let n: usize = reader.v();
    let m: i64 = reader.v();
    let mut a = reader.vec::<i64>(n);
    a.sort();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        if a[(i + 1) % n] == a[i] || (a[i] + 1) % m == a[(i + 1) % n] {
            uf.unite(i, (i + 1) % n);
        }
    }
    dbg!(&a);
    let all = a.iter().sum::<i64>();
    let mut ans = all;
    let mut sum = vec![0; n];
    for i in 0..n {
        sum[uf.root(i)] += a[i];
        dbg!((i, a[i], uf.root(i)));
    }
    for i in 0..n {
        chmin!(ans, all - sum[i]);
    }
    dbg!(&sum);
    writer.ln(ans);
}
#[test]
fn test() {
    if let Ok(reader) = ReaderFromString::from_file("./in") {
        let stdout = stdout();
        std::thread::Builder::new()
            .name("extend stack size".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || solve(reader, Writer::new(stdout.lock())))
            .unwrap()
            .join()
            .unwrap()
    } else {
        eprintln!("Failed to load file.")
    }
}
