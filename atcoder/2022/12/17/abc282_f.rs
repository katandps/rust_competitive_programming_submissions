pub use writer_impl::{AddLineTrait, BitsTrait, JoinTrait, WriterToStdout, WriterTrait};
mod writer_impl {
    use super::{stdout, BufWriter, Display, Integral, Write};
    pub trait WriterTrait {
        fn out<S: Display>(&mut self, s: S);
        fn flush(&mut self);
    }
    pub trait AddLineTrait {
        fn ln(&self) -> String;
    }
    impl<D: Display> AddLineTrait for D {
        fn ln(&self) -> String {
            self.to_string() + "\n"
        }
    }
    pub trait JoinTrait {
        fn join(self, separator: &str) -> String;
    }
    impl<D: Display, I: IntoIterator<Item = D>> JoinTrait for I {
        fn join(self, separator: &str) -> String {
            let mut buf = String::new();
            self.into_iter().fold("", |sep, arg| {
                buf.push_str(&format!("{}{}", sep, arg));
                separator
            });
            buf + "\n"
        }
    }
    pub trait BitsTrait {
        fn bits(self, length: Self) -> String;
    }
    impl<I: Integral> BitsTrait for I {
        fn bits(self, length: Self) -> String {
            let mut buf = String::new();
            let mut i = I::zero();
            while i < length {
                buf.push_str(&format!("{}", self >> i & I::one()));
                i += I::one();
            }
            buf + "\n"
        }
    }
    #[derive(Clone, Debug, Default)]
    pub struct WriterToStdout {
        buf: String,
    }
    impl WriterTrait for WriterToStdout {
        fn out<S: Display>(&mut self, s: S) {
            self.buf.push_str(&s.to_string());
        }
        fn flush(&mut self) {
            let stdout = stdout();
            let mut writer = BufWriter::new(stdout.lock());
            write!(writer, "{}", self.buf).expect("Failed to write.");
            let _ = writer.flush();
            self.buf.clear();
        }
    }
}
pub use reader_impl::{ReaderFromStdin, ReaderFromStr, ReaderTrait};
# [rustfmt :: skip ] mod reader_impl {use super :: {stdin , BufRead , FromStr as FS , VecDeque , Display , WriterTrait } ; pub trait ReaderTrait {fn next (& mut self ) -> Option < String > ; fn v < T : FS > (& mut self ) -> T {let s = self . next () . expect ("Insufficient input." ) ; s . parse () . ok () . expect ("Failed to parse." ) } fn v2 < T1 : FS , T2 : FS > (& mut self ) -> (T1 , T2 ) {(self . v () , self . v () ) } fn v3 < T1 : FS , T2 : FS , T3 : FS > (& mut self ) -> (T1 , T2 , T3 ) {(self . v () , self . v () , self . v () ) } fn v4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 ) {(self . v () , self . v () , self . v () , self . v () ) } fn v5 < T1 : FS , T2 : FS , T3 : FS , T4 : FS , T5 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 , T5 ) {(self . v () , self . v () , self . v () , self . v () , self . v () ) } fn vec < T : FS > (& mut self , length : usize ) -> Vec < T > {(0 .. length ) . map (| _ | self . v () ) . collect () } fn vec2 < T1 : FS , T2 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 ) > {(0 .. length ) . map (| _ | self . v2 () ) . collect () } fn vec3 < T1 : FS , T2 : FS , T3 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 ) > {(0 .. length ) . map (| _ | self . v3 () ) . collect () } fn vec4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 , T4 ) > {(0 .. length ) . map (| _ | self . v4 () ) . collect () } fn chars (& mut self ) -> Vec < char > {self . v :: < String > () . chars () . collect () } fn split (& mut self , zero : u8 ) -> Vec < usize > {self . v :: < String > () . chars () . map (| c | (c as u8 - zero ) as usize ) . collect () } fn digits (& mut self ) -> Vec < usize > {self . split (b'0' ) } fn lowercase (& mut self ) -> Vec < usize > {self . split (b'a' ) } fn uppercase (& mut self ) -> Vec < usize > {self . split (b'A' ) } fn char_map (& mut self , h : usize ) -> Vec < Vec < char > > {(0 .. h ) . map (| _ | self . chars () ) . collect () } fn bool_map (& mut self , h : usize , ng : char ) -> Vec < Vec < bool > > {self . char_map (h ) . iter () . map (| v | v . iter () . map (| & c | c != ng ) . collect () ) . collect () } fn matrix < T : FS > (& mut self , h : usize , w : usize ) -> Vec < Vec < T > > {(0 .. h ) . map (| _ | self . vec (w ) ) . collect () } } pub struct ReaderFromStr {buf : VecDeque < String > , } impl ReaderTrait for ReaderFromStr {fn next (& mut self ) -> Option < String > {self . buf . pop_front () } } impl ReaderFromStr {pub fn new (src : & str ) -> Self {Self {buf : src . split_whitespace () . map (ToString :: to_string ) . collect :: < Vec < String > > () . into () , } } pub fn push (& mut self , src : & str ) {for s in src . split_whitespace () . map (ToString :: to_string ) {self . buf . push_back (s ) ; } } pub fn from_file (path : & str ) -> Result < Self , Box < dyn std :: error :: Error > > {Ok (Self :: new (& std :: fs :: read_to_string (path ) ? ) ) } } impl WriterTrait for ReaderFromStr {fn out < S : Display > (& mut self , s : S ) {self . push (& s . to_string () ) ; } fn flush (& mut self ) {} } # [derive (Clone , Debug , Default ) ] pub struct ReaderFromStdin {buf : VecDeque < String > , } impl ReaderTrait for ReaderFromStdin {fn next (& mut self ) -> Option < String > {while self . buf . is_empty () {let stdin = stdin () ; let mut reader = stdin . lock () ; let mut l = String :: new () ; reader . read_line (& mut l ) . unwrap () ; self . buf . append (& mut l . split_whitespace () . map (ToString :: to_string ) . collect () ) ; } self . buf . pop_front () } } }
# [rustfmt :: skip ] pub trait ToLR < T > {fn to_lr (& self ) -> (T , T ) ; }
# [rustfmt :: skip ] impl < R : RangeBounds < T > , T : Copy + BoundedAbove + BoundedBelow + One + Add < Output = T > > ToLR < T > for R {# [inline ] fn to_lr (& self ) -> (T , T ) {use Bound :: {Excluded , Included , Unbounded } ; let l = match self . start_bound () {Unbounded => T :: min_value () , Included (& s ) => s , Excluded (& s ) => s + T :: one () , } ; let r = match self . end_bound () {Unbounded => T :: max_value () , Included (& e ) => e + T :: one () , Excluded (& e ) => e , } ; (l , r ) } }
# [rustfmt :: skip ] pub use std :: {cmp :: {max , min , Ordering , Reverse } , collections :: {hash_map :: RandomState , BTreeMap , BTreeSet , BinaryHeap , VecDeque , } , convert :: Infallible , convert :: {TryFrom , TryInto } , fmt :: {Debug , Display , Formatter } , hash :: {Hash , BuildHasherDefault , Hasher } , io :: {stdin , stdout , BufRead , BufWriter , Read , Write , StdoutLock } , iter :: {repeat , Product , Sum } , marker :: PhantomData , mem :: swap , ops :: {Add , AddAssign , BitAnd , BitAndAssign , BitOr , BitOrAssign , BitXor , BitXorAssign , Bound , Deref , DerefMut , Div , DivAssign , Index , IndexMut , Mul , MulAssign , Neg , Not , Range , RangeBounds , Rem , RemAssign , Shl , ShlAssign , Shr , ShrAssign , Sub , SubAssign , } , str :: {from_utf8 , FromStr } , } ;
#[allow(unused_macros)]
macro_rules ! chmin {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_min = min ! ($ ($ cmps ) ,+ ) ; if $ base > cmp_min {$ base = cmp_min ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! min {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ b } else {$ a } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = min ! ($ ($ rest ) ,+ ) ; if $ a > b {b } else {$ a } } } ; }
#[allow(unused_macros)]
macro_rules ! chmax {($ base : expr , $ ($ cmps : expr ) ,+ $ (, ) * ) => {{let cmp_max = max ! ($ ($ cmps ) ,+ ) ; if $ base < cmp_max {$ base = cmp_max ; true } else {false } } } ; }
#[allow(unused_macros)]
macro_rules ! max {($ a : expr $ (, ) * ) => {{$ a } } ; ($ a : expr , $ b : expr $ (, ) * ) => {{if $ a > $ b {$ a } else {$ b } } } ; ($ a : expr , $ ($ rest : expr ) ,+ $ (, ) * ) => {{let b = max ! ($ ($ rest ) ,+ ) ; if $ a > b {$ a } else {b } } } ; }
pub use io_debug_impl::IODebug;
# [rustfmt :: skip ] mod io_debug_impl {use super :: {stdout , BufWriter , Display , ReaderFromStr , ReaderTrait , Write , WriterTrait } ; pub struct IODebug < F > {pub reader : ReaderFromStr , pub test_reader : ReaderFromStr , pub buf : String , stdout : bool , f : F , } impl < F : FnMut (& mut ReaderFromStr , & mut ReaderFromStr ) > WriterTrait for IODebug < F > {fn out < S : Display > (& mut self , s : S ) {self . buf . push_str (& s . to_string () ) ; } fn flush (& mut self ) {if self . stdout {let stdout = stdout () ; let mut writer = BufWriter :: new (stdout . lock () ) ; write ! (writer , "{}" , self . buf ) . expect ("Failed to write." ) ; let _ = writer . flush () ; } self . test_reader . push (& self . buf ) ; self . buf . clear () ; (self . f ) (& mut self . test_reader , & mut self . reader ) } } impl < F > ReaderTrait for IODebug < F > {fn next (& mut self ) -> Option < String > {self . reader . next () } } impl < F > IODebug < F > {pub fn new (str : & str , stdout : bool , f : F ) -> Self {Self {reader : ReaderFromStr :: new (str ) , test_reader : ReaderFromStr :: new ("" ) , buf : String :: new () , stdout , f , } } } }
pub use io_impl::IO;
# [rustfmt :: skip ] mod io_impl {use super :: {ReaderFromStdin , ReaderTrait , WriterToStdout , WriterTrait } ; # [derive (Clone , Debug , Default ) ] pub struct IO {reader : ReaderFromStdin , writer : WriterToStdout , } impl ReaderTrait for IO {fn next (& mut self ) -> Option < String > {self . reader . next () } } impl WriterTrait for IO {fn out < S : std :: fmt :: Display > (& mut self , s : S ) {self . writer . out (s ) } fn flush (& mut self ) {self . writer . flush () } } }
# [rustfmt :: skip ] pub use self :: fxhasher_impl :: {FxHashMap as HashMap , FxHashSet as HashSet } ;
# [rustfmt :: skip ] mod fxhasher_impl {use super :: {BitXor , BuildHasherDefault , Hasher , TryInto } ; use std :: collections :: {HashMap , HashSet } ; # [derive (Default ) ] pub struct FxHasher {hash : u64 , } type BuildHasher = BuildHasherDefault < FxHasher > ; pub type FxHashMap < K , V > = HashMap < K , V , BuildHasher > ; pub type FxHashSet < V > = HashSet < V , BuildHasher > ; const ROTATE : u32 = 5 ; const SEED : u64 = 0x51_7c_c1_b7_27_22_0a_95 ; impl Hasher for FxHasher {# [inline ] fn finish (& self ) -> u64 {self . hash as u64 } # [inline ] fn write (& mut self , mut bytes : & [u8 ] ) {while bytes . len () >= 8 {self . add_to_hash (u64 :: from_ne_bytes (bytes [.. 8 ] . try_into () . unwrap () ) ) ; bytes = & bytes [8 .. ] ; } while bytes . len () >= 4 {self . add_to_hash (u64 :: from (u32 :: from_ne_bytes (bytes [.. 4 ] . try_into () . unwrap () , ) ) ) ; bytes = & bytes [4 .. ] ; } while bytes . len () >= 2 {self . add_to_hash (u64 :: from (u16 :: from_ne_bytes (bytes [.. 2 ] . try_into () . unwrap () , ) ) ) ; } if let Some (& byte ) = bytes . first () {self . add_to_hash (u64 :: from (byte ) ) ; } } } impl FxHasher {# [inline ] pub fn add_to_hash (& mut self , i : u64 ) {self . hash = self . hash . rotate_left (ROTATE ) . bitxor (i ) . wrapping_mul (SEED ) ; } } }
#[allow(unused_macros)]
macro_rules ! dbg {($ ($ x : tt ) * ) => {{# [cfg (debug_assertions ) ] {std :: dbg ! ($ ($ x ) * ) } # [cfg (not (debug_assertions ) ) ] {($ ($ x ) * ) } } } }
# [rustfmt :: skip ] pub use algebra_traits :: {AbelianGroup , Associative , Band , BoundedAbove , BoundedBelow , Commutative , CommutativeMonoid , Group , Idempotent , Invertible , Magma , MapMonoid , Monoid , One , Pow , PrimitiveRoot , SemiGroup , Unital , Zero , TrailingZeros } ;
# [rustfmt :: skip ] mod algebra_traits {use super :: Debug ; pub trait Magma {type M : Clone + PartialEq + Debug ; fn op (x : & Self :: M , y : & Self :: M ) -> Self :: M ; } pub trait Associative {} pub trait Unital : Magma {fn unit () -> Self :: M ; } pub trait Commutative : Magma {} pub trait Invertible : Magma {fn inv (x : & Self :: M ) -> Self :: M ; } pub trait Idempotent : Magma {} pub trait SemiGroup : Magma + Associative {} pub trait Monoid : Magma + Associative + Unital {fn pow (& self , x : Self :: M , mut n : usize ) -> Self :: M {let mut res = Self :: unit () ; let mut base = x ; while n > 0 {if n & 1 == 1 {res = Self :: op (& res , & base ) ; } base = Self :: op (& base , & base ) ; n >>= 1 ; } res } } pub trait CommutativeMonoid : Magma + Associative + Unital + Commutative {} pub trait Group : Magma + Associative + Unital + Invertible {} pub trait AbelianGroup : Magma + Associative + Unital + Commutative + Invertible {} pub trait Band : Magma + Associative + Idempotent {} impl < M : Magma + Associative > SemiGroup for M {} impl < M : Magma + Associative + Unital > Monoid for M {} impl < M : Magma + Associative + Unital + Commutative > CommutativeMonoid for M {} impl < M : Magma + Associative + Unital + Invertible > Group for M {} impl < M : Magma + Associative + Unital + Commutative + Invertible > AbelianGroup for M {} impl < M : Magma + Associative + Idempotent > Band for M {} pub trait MapMonoid {type Mono : Monoid ; type Func : Monoid ; fn op (& self , x : & < Self :: Mono as Magma > :: M , y : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M {Self :: Mono :: op (x , y ) } fn unit () -> < Self :: Mono as Magma > :: M {Self :: Mono :: unit () } fn apply (& self , f : & < Self :: Func as Magma > :: M , value : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M ; fn identity_map () -> < Self :: Func as Magma > :: M {Self :: Func :: unit () } fn compose (& self , f : & < Self :: Func as Magma > :: M , g : & < Self :: Func as Magma > :: M , ) -> < Self :: Func as Magma > :: M {Self :: Func :: op (f , g ) } } pub trait Zero {fn zero () -> Self ; } pub trait One {fn one () -> Self ; } pub trait BoundedBelow {fn min_value () -> Self ; } pub trait BoundedAbove {fn max_value () -> Self ; } pub trait Pow {fn pow (self , exp : i64 ) -> Self ; } pub trait PrimitiveRoot {const DIVIDE_LIMIT : usize ; fn primitive_root () -> Self ; } pub trait TrailingZeros {fn trailing_zero (self ) -> Self ; } }
# [rustfmt :: skip ] pub trait Integral : 'static + Send + Sync + Copy + Ord + Display + Debug + Add < Output = Self > + Sub < Output = Self > + Mul < Output = Self > + Div < Output = Self > + Rem < Output = Self > + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + Product + BitOr < Output = Self > + BitAnd < Output = Self > + BitXor < Output = Self > + Not < Output = Self > + Shl < Output = Self > + Shr < Output = Self > + BitOrAssign + BitAndAssign + BitXorAssign + ShlAssign + ShrAssign + Zero + One + BoundedBelow + BoundedAbove + TrailingZeros {}
macro_rules ! impl_integral {($ ($ ty : ty ) ,* ) => {$ (impl Zero for $ ty {# [inline ] fn zero () -> Self {0 } } impl One for $ ty {# [inline ] fn one () -> Self {1 } } impl BoundedBelow for $ ty {# [inline ] fn min_value () -> Self {Self :: min_value () } } impl BoundedAbove for $ ty {# [inline ] fn max_value () -> Self {Self :: max_value () } } impl TrailingZeros for $ ty {# [inline ] fn trailing_zero (self ) -> Self {self . trailing_zeros () as $ ty } } impl Integral for $ ty {} ) * } ; }
impl_integral!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
# [rustfmt :: skip ] pub fn main () {std :: thread :: Builder :: new () . name ("extend stack size" . into () ) . stack_size (32 * 1024 * 1024 ) . spawn (move | | {let mut io = IO :: default () ; solve (& mut io ) ; io . flush () ; } ) . unwrap () . join () . unwrap () }

pub trait SliceBounds {
    type Item: Ord;
    fn lower_bound(&self, k: &Self::Item) -> usize;
    fn upper_bound(&self, k: &Self::Item) -> usize;
}
pub trait SliceBoundsBy {
    type Item;
    fn lower_bound_by<F: FnMut(&Self::Item) -> Ordering>(&self, f: F) -> usize;
    fn upper_bound_by<F: FnMut(&Self::Item) -> Ordering>(&self, f: F) -> usize;
}
pub trait SliceBoundsByKey {
    type Item;
    fn lower_bound_by_key<K: Ord, F: FnMut(&Self::Item) -> K>(&self, k: &K, f: F) -> usize;
    fn upper_bound_by_key<K: Ord, F: FnMut(&Self::Item) -> K>(&self, k: &K, f: F) -> usize;
}
impl<T: Ord> SliceBounds for [T] {
    type Item = T;
    fn lower_bound(&self, k: &T) -> usize {
        self.lower_bound_by(|p| p.cmp(k))
    }
    fn upper_bound(&self, k: &T) -> usize {
        self.upper_bound_by(|p| p.cmp(k))
    }
}
impl<T> SliceBoundsBy for [T] {
    type Item = T;
    fn lower_bound_by<F: FnMut(&T) -> Ordering>(&self, mut f: F) -> usize {
        self.binary_search_by(|p| f(p).then(Ordering::Greater))
            .unwrap_err()
    }
    fn upper_bound_by<F: FnMut(&T) -> Ordering>(&self, mut f: F) -> usize {
        self.binary_search_by(|p| f(p).then(Ordering::Less))
            .unwrap_err()
    }
}
impl<T> SliceBoundsByKey for [T] {
    type Item = T;
    fn lower_bound_by_key<K: Ord, F: FnMut(&T) -> K>(&self, k: &K, mut f: F) -> usize {
        self.lower_bound_by(|p| f(p).cmp(k))
    }
    fn upper_bound_by_key<K: Ord, F: FnMut(&T) -> K>(&self, k: &K, mut f: F) -> usize {
        self.upper_bound_by(|p| f(p).cmp(k))
    }
}
pub fn solve<IO: ReaderTrait + WriterTrait>(io: &mut IO) {
    let n: usize = io.v();
    let mut v = HashMap::default();
    let mut t = (1..=n).collect::<Vec<_>>();
    t.sort_by_key(|ti| {
        let mut t = *ti;
        let mut c = 0;
        while t & 1 == 0 {
            c += 1;
            t >>= 1;
        }
        c
    });
    let mut p = 1;
    let mut vec = Vec::new();
    for ti in t {
        let c = ti.trailing_zero();
        for i in 0..=1 << c {
            if i + ti > n {
                break;
            }
            v.entry(ti).or_insert(Vec::new()).push((ti + i, p));
            vec.push((ti, ti + i));
            p += 1;
        }
        for i in 1..1 << c {
            v.entry(ti).or_insert(Vec::new()).push((ti - i, p));
            vec.push((ti - i, ti));
            p += 1;
        }
        v.entry(ti).or_insert(Vec::new()).sort();
    }
    io.out(vec.len().ln());
    for &(l, r) in &vec {
        io.out(format!("{} {}", l, r).ln());
    }
    io.flush();
    for _ in 0..io.v::<usize>() {
        let (l, r) = io.v2::<usize, usize>();
        if l == r {
            let vv = v.get(&l).unwrap();
            let (a, p) = vv[vv.lower_bound_by_key(&r, |(a, _p)| *a)];
            // dbg!(a, v.get(&l), l);
            io.out(format!("{} {}", p, p).ln());
        } else {
            let mut k = 1;
            for b in 1..=11 {
                if l / (1 << b) == r / (1 << b) && l % (1 << b) != 0 && r % (1 << b) != 0 {
                    break;
                }
                k = 1 << b;
            }
            let t = r / k * k;
            let vv = v.get(&t).unwrap();
            let (aa, ap) = vv[vv.lower_bound_by_key(&l, |(a, _p)| *a)];
            let (ba, bp) = vv[vv.lower_bound_by_key(&r, |(a, _p)| *a)];
            // dbg!(format!("{} {} {}", aa, t, ba));
            io.out(format!("{} {}", ap, bp).ln());
        }
        io.flush();
    }
    dbg!(vec.len());
}

#[test]
fn test() {
    std::thread::Builder::new()
        .name("extend stack size".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            let n = 4000;
            let mut xorshift = XorShift::default();
            let mut m = None;
            let mut v = Vec::new();
            let mut q = 100000;
            let (mut l, mut r) = (0, 0);
            let mut io = IODebug::new(
                &format!("{}", n),
                false,
                |outer: &mut ReaderFromStr, inner: &mut ReaderFromStr| {
                    if m.is_none() {
                        m = Some(outer.v::<usize>());
                        v = outer.vec2::<usize, usize>(m.unwrap());
                        inner.out(q.ln());
                    } else {
                        let (a, b) = outer.v2::<usize, usize>();
                        let ((l1, r1), (l2, r2)) = (v[a - 1], v[b - 1]);

                        assert!(
                            l1 == l && r2 == r && r1 == l2,
                            "l:{} r:{} l1:{} r1:{} l2:{} r2:{}",
                            l,
                            r,
                            l1,
                            r1,
                            l2,
                            r2
                        );

                        // dbg!(l1, r1, l2, r2);
                    }
                    // 出題
                    if q > 0 {
                        l = xorshift.rand_range(1..=n) as usize;
                        r = xorshift.rand_range(l as i64..=n) as usize;
                        // println!("出題{}: {} {}", q, l, r);
                        inner.out(format!("{} {}", l, r).ln());
                        inner.flush();
                        q -= 1;
                    }
                },
            );
            solve(&mut io);
            io.flush();
        })
        .unwrap()
        .join()
        .unwrap()
}

#[derive(Clone, Debug)]
pub struct XorShift {
    seed: u64,
}
mod xor_shift_impl {
    use super::{RangeBounds, ToLR, XorShift};
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
