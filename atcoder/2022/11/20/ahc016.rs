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
            buf
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
# [rustfmt :: skip ] mod reader_impl {use super :: {stdin , BufRead , FromStr as FS , VecDeque , Display , WriterTrait } ; pub trait ReaderTrait {fn next (& mut self ) -> Option < String > ; fn v < T : FS > (& mut self ) -> T {let s = self . next () . expect ("Insufficient input." ) ; s . parse () . ok () . expect ("Failed to parse." ) } fn v2 < T1 : FS , T2 : FS > (& mut self ) -> (T1 , T2 ) {(self . v () , self . v () ) } fn v3 < T1 : FS , T2 : FS , T3 : FS > (& mut self ) -> (T1 , T2 , T3 ) {(self . v () , self . v () , self . v () ) } fn v4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 ) {(self . v () , self . v () , self . v () , self . v () ) } fn v5 < T1 : FS , T2 : FS , T3 : FS , T4 : FS , T5 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 , T5 ) {(self . v () , self . v () , self . v () , self . v () , self . v () ) } fn vec < T : FS > (& mut self , length : usize ) -> Vec < T > {(0 .. length ) . map (| _ | self . v () ) . collect () } fn vec2 < T1 : FS , T2 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 ) > {(0 .. length ) . map (| _ | self . v2 () ) . collect () } fn vec3 < T1 : FS , T2 : FS , T3 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 ) > {(0 .. length ) . map (| _ | self . v3 () ) . collect () } fn vec4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 , T4 ) > {(0 .. length ) . map (| _ | self . v4 () ) . collect () } fn chars (& mut self ) -> Vec < char > {self . v :: < String > () . chars () . collect () } fn split (& mut self , zero : u8 ) -> Vec < usize > {self . v :: < String > () . chars () . map (| c | (c as u8 - zero ) as usize ) . collect () } fn digits (& mut self ) -> Vec < usize > {self . split (b'0' ) } fn lowercase (& mut self ) -> Vec < usize > {self . split (b'a' ) } fn uppercase (& mut self ) -> Vec < usize > {self . split (b'A' ) } fn char_map (& mut self , h : usize ) -> Vec < Vec < char > > {(0 .. h ) . map (| _ | self . chars () ) . collect () } fn bool_map (& mut self , h : usize , ng : char ) -> Vec < Vec < bool > > {self . char_map (h ) . iter () . map (| v | v . iter () . map (| & c | c != ng ) . collect () ) . collect () } fn matrix < T : FS > (& mut self , h : usize , w : usize ) -> Vec < Vec < T > > {(0 .. h ) . map (| _ | self . vec (w ) ) . collect () } } pub struct ReaderFromStr {pub buf : VecDeque < String > , } impl ReaderTrait for ReaderFromStr {fn next (& mut self ) -> Option < String > {self . buf . pop_front () } } impl ReaderFromStr {pub fn new (src : & str ) -> Self {Self {buf : src . split_whitespace () . map (ToString :: to_string ) . collect :: < Vec < String > > () . into () , } } pub fn push (& mut self , src : & str ) {for s in src . split_whitespace () . map (ToString :: to_string ) {self . buf . push_back (s ) ; } } pub fn from_file (path : & str ) -> Result < Self , Box < dyn std :: error :: Error > > {Ok (Self :: new (& std :: fs :: read_to_string (path ) ? ) ) } } impl WriterTrait for ReaderFromStr {fn out < S : Display > (& mut self , s : S ) {self . push (& s . to_string () ) ; } fn flush (& mut self ) {} } # [derive (Clone , Debug , Default ) ] pub struct ReaderFromStdin {buf : VecDeque < String > , } impl ReaderTrait for ReaderFromStdin {fn next (& mut self ) -> Option < String > {while self . buf . is_empty () {let stdin = stdin () ; let mut reader = stdin . lock () ; let mut l = String :: new () ; reader . read_line (& mut l ) . unwrap () ; self . buf . append (& mut l . split_whitespace () . map (ToString :: to_string ) . collect () ) ; } self . buf . pop_front () } } }
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
# [rustfmt :: skip ] pub fn main () {std :: thread :: Builder :: new () . name ("extend stack size" . into () ) . stack_size (32 * 1024 * 1024 ) . spawn (move | | {let mut io = IO :: default () ; solve (& mut io, None ) ; io . flush () ; } ) . unwrap () . join () . unwrap () }

#[derive(Clone, Debug, Ord, PartialOrd, PartialEq, Eq)]
pub struct XorShift {
    seed: u64,
}
mod xor_shift_impl {
    use super::{RangeBounds, ToLR, XorShift};
    impl Default for XorShift {
        fn default() -> Self {
            let seed = 0xf0fb588ca2196dac;
            Self { seed }
        }
    }
    impl Iterator for XorShift {
        type Item = u64;
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
        pub fn rand(&mut self, m: u64) -> u64 {
            self.next().unwrap() % m
        }
        pub fn rand_range<R: RangeBounds<i64>>(&mut self, range: R) -> i64 {
            let (l, r) = range.to_lr();
            let k = self.next().unwrap() as i64;
            k.rem_euclid(r - l) + l
        }
        pub fn randf(&mut self) -> f64 {
            const UPPER_MASK: u64 = 0x3FF0000000000000;
            const LOWER_MASK: u64 = 0xFFFFFFFFFFFFF;
            f64::from_bits(UPPER_MASK | (self.next().unwrap() & LOWER_MASK)) - 1.0
        }

        pub fn shuffle<T>(&mut self, s: &mut [T]) {
            for i in 0..s.len() {
                s.swap(i, self.rand_range(i as i64..s.len() as i64) as usize);
            }
        }
    }
}

#[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct BitSet {
    bits: Vec<u64>,
    size: usize,
}
mod bitset_impl {
    use super::{
        BitAnd, BitOr, BitSet, BitXor, Debug, Display, Formatter, Index, Not, Shl, ShlAssign, Shr,
        ShrAssign,
    };
    impl BitSet {
        const BLOCK_LEN: usize = 1 << Self::BLOCK_LEN_LEN;
        const BLOCK_LEN_LEN: usize = 6;
        pub fn new(size: usize) -> Self {
            let bits = vec![0; (size + Self::BLOCK_LEN - 1) / Self::BLOCK_LEN];
            Self { bits, size }
        }
        pub fn set(&mut self, index: usize, b: bool) {
            assert!(index < self.size);
            if b {
                self.bits[index >> Self::BLOCK_LEN_LEN] |= 1 << (index & (Self::BLOCK_LEN - 1));
            } else {
                self.bits[index >> Self::BLOCK_LEN_LEN] &= !(1 << (index & (Self::BLOCK_LEN - 1)));
            }
        }
        pub fn count_ones(&self) -> u32 {
            self.bits.iter().map(|b| b.count_ones()).sum()
        }
        pub fn get_u64(&self) -> u64 {
            self.bits[0]
        }
        fn chomp(&mut self) {
            let r = self.size & (Self::BLOCK_LEN - 1);
            if r != 0 {
                let d = Self::BLOCK_LEN - r;
                if let Some(x) = self.bits.last_mut() {
                    *x = (*x << d) >> d;
                }
            }
        }
    }
    impl Debug for BitSet {
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            Display::fmt(self, f)
        }
    }
    impl Index<usize> for BitSet {
        type Output = bool;
        fn index(&self, index: usize) -> &bool {
            assert!(index < self.size);
            &[false, true][((self.bits[index >> Self::BLOCK_LEN_LEN]
                >> (index & (Self::BLOCK_LEN - 1)))
                & 1) as usize]
        }
    }
    impl BitAnd for BitSet {
        type Output = BitSet;
        fn bitand(self, rhs: Self) -> Self::Output {
            &self & &rhs
        }
    }
    impl BitAnd for &BitSet {
        type Output = BitSet;
        fn bitand(self, rhs: Self) -> Self::Output {
            assert_eq!(self.size, rhs.size);
            BitSet {
                bits: self
                    .bits
                    .iter()
                    .zip(rhs.bits.iter())
                    .map(|(l, r)| l & r)
                    .collect(),
                size: self.size,
            }
        }
    }
    impl BitOr for BitSet {
        type Output = Self;
        fn bitor(self, rhs: Self) -> Self::Output {
            assert_eq!(self.size, rhs.size);
            Self {
                bits: (0..self.bits.len())
                    .map(|i| self.bits[i] | rhs.bits[i])
                    .collect(),
                size: self.size,
            }
        }
    }
    impl BitXor for BitSet {
        type Output = Self;
        fn bitxor(self, rhs: Self) -> Self::Output {
            assert_eq!(self.size, rhs.size);
            Self {
                bits: (0..self.bits.len())
                    .map(|i| self.bits[i] ^ rhs.bits[i])
                    .collect(),
                size: self.size,
            }
        }
    }
    impl ShlAssign<usize> for BitSet {
        fn shl_assign(&mut self, rhs: usize) {
            if rhs >= self.size {
                self.bits.iter_mut().for_each(|b| *b = 0);
                return;
            }
            let block = rhs >> Self::BLOCK_LEN_LEN;
            let inner = rhs & (Self::BLOCK_LEN - 1);
            if inner == 0 {
                (block..self.bits.len())
                    .rev()
                    .for_each(|i| self.bits[i] = self.bits[i - block])
            } else {
                (block + 1..self.bits.len()).rev().for_each(|i| {
                    self.bits[i] = (self.bits[i - block] << inner)
                        | (self.bits[i - block - 1] >> (Self::BLOCK_LEN - inner))
                });
                self.bits[block] = self.bits[0] << inner;
            }
            self.bits[..block].iter_mut().for_each(|b| *b = 0);
            self.chomp();
        }
    }
    impl Shl<usize> for BitSet {
        type Output = Self;
        fn shl(mut self, rhs: usize) -> Self::Output {
            self <<= rhs;
            self
        }
    }
    impl ShrAssign<usize> for BitSet {
        fn shr_assign(&mut self, rhs: usize) {
            if rhs >= self.size {
                self.bits.iter_mut().for_each(|b| *b = 0);
                return;
            }
            let block = rhs >> Self::BLOCK_LEN_LEN;
            let inner = rhs & (Self::BLOCK_LEN - 1);
            let len = self.bits.len();
            if inner == 0 {
                (0..len - block).for_each(|i| self.bits[i] = self.bits[i + block])
            } else {
                (0..len - block - 1).for_each(|i| {
                    self.bits[i] = (self.bits[i + block] >> inner)
                        | (self.bits[i + block + 1] << (Self::BLOCK_LEN - inner))
                });
                self.bits[len - block - 1] = self.bits[len - 1] >> inner;
            }
            self.bits[len - block..].iter_mut().for_each(|b| *b = 0);
        }
    }
    impl Shr<usize> for BitSet {
        type Output = Self;
        fn shr(mut self, rhs: usize) -> Self::Output {
            self >>= rhs;
            self
        }
    }
    impl Not for BitSet {
        type Output = Self;
        fn not(self) -> Self::Output {
            Self {
                bits: self.bits.iter().map(|&i| i ^ std::u64::MAX).collect(),
                size: self.size,
            }
        }
    }
    impl Display for BitSet {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{}",
                (0..self.size)
                    .map(|i| usize::from(self[i]).to_string())
                    .collect::<String>()
            )
        }
    }
}
pub use split_of_natural_number_impl::SplitOfNumber;
mod split_of_natural_number_impl {
    #[derive(Clone, Debug)]
    pub struct SplitOfNumber(Option<Vec<usize>>);
    impl Iterator for SplitOfNumber {
        type Item = Vec<usize>;
        fn next(&mut self) -> Option<Vec<usize>> {
            let ret = self.0.clone();
            if let Some(v) = &mut self.0 {
                match v.iter().rposition(|&x| x != 1) {
                    None => self.0 = None,
                    Some(i) => {
                        let others = v.split_off(i);
                        let mut rest = others.iter().sum::<usize>();
                        let max = others[0] - 1;
                        while rest > 0 {
                            let next = rest.min(max);
                            v.push(next);
                            rest -= next;
                        }
                    }
                }
            } else {
                self.0 = None
            };
            ret
        }
    }
    impl From<usize> for SplitOfNumber {
        fn from(n: usize) -> Self {
            SplitOfNumber(Some(vec![n]))
        }
    }
    impl From<&[usize]> for SplitOfNumber {
        fn from(src: &[usize]) -> Self {
            SplitOfNumber(Some(src.to_vec()))
        }
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

pub fn solve<IO: ReaderTrait + WriterTrait>(io: &mut IO, n: Option<usize>) {
    let (m, e) = io.v2::<usize, f64>();
    let solver = Solver::build(m, (e * 100.0 + 0.5) as i64, n);
    let (n, graphs) = solver.graphs();
    io.out(n.ln());
    for g in graphs {
        io.out(g);
    }
    io.flush();
    for _ in 0..TEST_CASES {
        let h = io.digits();
        io.out(solver.query(&Selection::new(n, &h)).ln());
        io.flush();
    }
}
const TEST_CASES: i64 = 100;

#[derive(Clone, Debug)]
enum Solver {
    OneBlock(OneBlock),
    UnionStatWithError(UnionStatWithError),
    Random {
        graphs: Vec<Selection>,
        m: usize,
        n: usize,
    },
}

impl Solver {
    fn build(m: usize, e: i64, n: Option<usize>) -> Self {
        if e <= 2 {
            Self::build_union_stat_with_error(m, e)
        } else if (e >= 27 && m >= 78)
            || (e >= 28 && m >= 70)
            || (e >= 29 && m >= 60)
            || (e >= 30 && m >= 42)
            || (e >= 31 && m >= 30)
            || (e >= 32)
        {
            Self::random(m)
        } else {
            Self::build_one_block(m, e, n)
        }
    }
    fn random(m: usize) -> Self {
        // n=最小 0に寄せたバージョンと1に寄せたバージョン、半々のものを作ってヒット率を少し上げる
        let mut graphs = Vec::with_capacity(m);
        let n = 4;
        let len = n * (n - 1) / 2;
        for i in 0..m {
            if i < m / 3 {
                graphs.push(Selection::new(n, &vec![0; len]));
            } else if i >= 2 * m / 3 {
                graphs.push(Selection::new(n, &vec![1; len]));
            } else {
                graphs.push(Selection::new(n, &vec![0, 0, 0, 1, 1, 1]));
            }
        }
        Self::Random { m, n: 4, graphs }
    }
    fn build_union_stat_with_error(m: usize, e: i64) -> Self {
        Self::UnionStatWithError(UnionStatWithError::build(m, e))
    }
    fn build_one_block(m: usize, e: i64, n: Option<usize>) -> Self {
        Self::OneBlock(OneBlock::build(m, e, n))
    }
    fn graphs(&self) -> (usize, Vec<String>) {
        match self {
            Self::OneBlock(stat) => (stat.n, stat.graphs.iter().map(|g| g.to_string()).collect()),
            Self::UnionStatWithError(stat) => {
                (stat.n, stat.graphs.iter().map(|g| g.to_string()).collect())
            }
            Self::Random { graphs, n, .. } => (*n, graphs.iter().map(|g| g.to_string()).collect()),
        }
    }

    fn query(&self, h: &Selection) -> usize {
        match self {
            Self::Random { m, .. } => {
                let ones = h.count_ones();
                if ones < 2 {
                    0
                } else if ones < 6 {
                    m / 2
                } else {
                    m - 1
                }
            }
            Self::OneBlock(stat) => stat.query(h),
            Self::UnionStatWithError(stat) => stat.query(h),
        }
    }
}

#[derive(Clone, Debug)]
struct OneBlock {
    graphs: Vec<Selection>,
    graph_size: Vec<Vec<i64>>,
    m: usize,
    n: usize,
}

impl OneBlock {
    fn build(m: usize, e: i64, n: Option<usize>) -> Self {
        let mut graphs = Vec::new();
        let mut n = n.unwrap_or(max!(10 + e as usize, m));
        chmax!(n, m);
        chmin!(n, 100);
        let mut graph_size = vec![Vec::new(), Vec::new()];

        // 辺のあるクラスタ
        let s = m / 2;
        for i in 0..s {
            let c = (n + 1) / 2 + i;
            let mut g = Selection::empty(n);
            for i in 0..c {
                for j in i + 1..c {
                    g.set_true(i, j);
                }
            }
            for i in c..n {
                for j in i + 1..n {
                    g.set_true(i, j);
                }
            }
            graphs.push(g);
            graph_size[0].push(c as i64);
        }
        // 辺のないクラスタ
        let s = (m + 1) / 2;
        for i in 0..s {
            let c = (n + 1) / 2 + i;
            // dbg!(n, i, c);
            assert!(c <= n, "n:{} c:{} m:{}", n, c, m);
            let mut g = Selection::empty(n);
            for i in 0..c {
                for j in c..n {
                    g.set_true(i, j);
                    g.set_true(j, i);
                }
            }
            graphs.push(g);
            graph_size[1].push(c as i64);
        }
        // dbg!(&graph_size);
        assert_eq!(m, graphs.len());
        Self {
            n,
            m,
            graph_size,
            graphs,
        }
    }
    fn query(&self, h: &Selection) -> usize {
        let mut uf = UnionFind::new(self.n);
        for i in 0..self.n {
            for j in i + 1..self.n {
                let mut sim = 0;
                for k in 0..self.n {
                    if h.get(i, k) == h.get(j, k) {
                        sim += 1;
                    }
                }
                if sim >= self.n * 2 / 3 {
                    uf.unite(i, j);
                }
            }
        }

        let mut size = vec![0; self.n];
        for i in 0..self.n {
            size[uf.root(i)] = uf.size(i) as i64;
        }
        let mut max_size = 0;
        let mut max_root = 0;
        for i in 0..self.n {
            if chmax!(max_size, size[i]) {
                max_root = i;
            }
        }
        let v = (0..self.n)
            .filter(|i| uf.root(*i) == max_root)
            .collect::<Vec<_>>();

        let mut edge_cnt = 0;
        for i in 0..v.len() {
            for j in i + 1..v.len() {
                if h.get(v[i], v[j]) {
                    edge_cnt += 1;
                }
            }
        }
        let white = usize::from(edge_cnt < v.len() * (v.len() - 1) / 2 / 2);
        let size = v.len() as i64;

        let mut diff_min = 1 << 30;
        let mut l = 0;
        for i in 0..self.graph_size[white].len() {
            if chmin!(diff_min, (size - self.graph_size[white][i]).abs()) {
                l = i;
            }
        }
        // dbg!(white, size);
        max!(0, min!(self.m - 1, l + self.graph_size[0].len() * white))
    }
}

#[derive(Clone, Debug)]
struct UnionStatWithError {
    graphs: Vec<Selection>,
    map: HashMap<Selection, usize>,
    n: usize,
}
impl UnionStatWithError {
    fn build(m: usize, e: i64) -> Self {
        // let mut graph_pool = Vec::new();
        let n = if m + e as usize <= 11 {
            4
        } else if m + e as usize <= 34 {
            5
        } else {
            6
        };
        let len = n * (n - 1) / 2;
        let mut graph_pool = Vec::new();
        let mut parsed_map = HashMap::default();

        let mut graphs = Vec::new();

        let mut graph_map = HashMap::default();
        for p in 0..1 << len {
            let g = Selection::new(
                n,
                &(0..len).map(|i| (p >> i & 1) as usize).collect::<Vec<_>>(),
            );
            let parsed = g.canonical();
            if let Some(p) = parsed_map.get(&parsed) {
                graph_map.insert(g, *p);
            } else {
                parsed_map.insert(parsed.clone(), graph_pool.len());
                graph_map.insert(g.clone(), graph_pool.len());
                graph_pool.push(parsed);
            }
        }
        let k = graph_pool.len();
        let mut graph = Graph::new(k);
        for s in 0..k {
            let mut done = HashSet::default();
            for j in 0..len {
                let b = graph_pool[s].b[j];
                graph_pool[s].b.set(j, !b);

                if let Some(t) = graph_map.get(&graph_pool[s].canonical()) {
                    if done.insert(*t) {
                        graph.add_arc(s, *t, 1);
                    }
                }
                graph_pool[s].b.set(j, b);
            }
        }

        let mut dist = vec![1 << 30; k];
        let mut nearest = vec![0; k];
        for i in 0..m {
            let mut max_dist = 0;
            let mut s = 0;
            for i in 0..k {
                if chmax!(max_dist, dist[i]) {
                    s = i;
                }
            }
            assert!(max_dist > 0);
            // max_iをi番目のgraphとして採用
            graphs.push(graph_pool[s].clone());
            nearest[s] = i;
            dist[s] = 0;
            let mut q = vec![s];
            for _ in 0..3 {
                let mut next = Vec::new();
                while let Some(s) = q.pop() {
                    for (t, w) in graph.edges(s) {
                        if chmin!(dist[t], dist[s] + w) {
                            nearest[t] = i;
                            next.push(t);
                        }
                    }
                }
                q = next;
            }
            // dbg!(&s, &graphs, &nearest, &dist);
        }
        // dbg!(&graphs, &graph_pool, graphs.len(), graph_pool.len());
        let mut map = HashMap::default();
        for (i, g) in graph_pool.into_iter().enumerate() {
            map.insert(g, nearest[i]);
        }
        // dbg!(&map, &graphs);
        Self { graphs, map, n }
    }
    fn query(&self, h: &Selection) -> usize {
        let h = h.canonical();
        let mut min_diff = 1 << 30;
        let mut ret = 0;
        for (g, i) in &self.map {
            let mut c = 0;
            for i in 0..self.n * (self.n - 1) / 2 {
                if g.b[i] != h.b[i] {
                    c += 1;
                }
            }
            if chmin!(min_diff, c) {
                ret = *i;
            }
        }
        ret
    }
}

#[derive(Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
struct Selection {
    b: BitSet,
    n: usize,
}

impl Selection {
    fn empty(n: usize) -> Self {
        Self {
            n,
            b: BitSet::new(n * (n - 1) / 2),
        }
    }
    fn new(n: usize, src: &[usize]) -> Self {
        let mut ret = Self::empty(n);
        for i in 0..n * (n - 1) / 2 {
            if src[i] > 0 {
                ret.b.set(i, true)
            }
        }
        ret
    }
    #[inline]
    fn index(&self, mut i: usize, mut j: usize) -> usize {
        if i > j {
            swap(&mut i, &mut j)
        }
        (self.n - 1 + self.n - 1 - i + 1) * i / 2 + (j - i - 1)
    }
    #[inline]
    fn get(&self, i: usize, j: usize) -> bool {
        if i == j {
            return false;
        }
        self.b[self.index(i, j)]
    }
    #[inline]
    fn set_true(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        self.b.set(self.index(i, j), true)
    }
    #[inline]
    fn count_ones(&self) -> usize {
        self.b.count_ones() as usize
    }
}

impl Display for Selection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            (0..self.n * (self.n - 1) / 2)
                .map(|i| if self.b[i] { "1" } else { "0" })
                .join("")
                .ln()
        )
    }
}

impl Debug for Selection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n{}",
            (0..self.n)
                .map(|i| (0..self.n)
                    .map(|j| if self.get(i, j) { "1" } else { "0" })
                    .join("")
                    .ln())
                .join("")
                .ln()
        )
    }
}
impl Canonize for Selection {
    fn size(&self) -> usize {
        self.n
    }

    fn apply_morphism(&self, p: &[usize]) -> Self {
        let mut ret = Self::empty(self.n);
        for i in 0..self.n - 1 {
            for j in i + 1..self.n {
                if self.get(i, j) {
                    ret.set_true(p[i], p[j]);
                }
            }
        }
        ret
    }
    fn invariant_neighborhood(&self, i: usize) -> Vec<Vec<usize>> {
        vec![(0..self.n).rev().filter(|&j| self.get(i, j)).collect()]
    }
}

use std::collections::btree_map::Entry::{Occupied, Vacant};
use std::rc::Rc;
/// A partition of a set `{0..n-1}`.
/// Each part of the partition is refered by an index in `{0..k-1}`,
/// where `k` is the number of parts.
#[derive(Clone, Debug)]
pub struct Partition {
    /// Vector of contained elements
    elems: Vec<usize>,
    /// `elems[rev_elem[i]] = i`
    rev_elems: Vec<usize>,
    /// i is in part set_id[i]
    set_id: Vec<usize>,
    /// List of parts indexed by their id
    sets: Vec<Set>,
    /// mechanism to split sets:
    /// `sieve` contains a value for element (0 by default)
    /// when split is called, the partition is refined according to the values in `sieve`
    /// and `sieve` is reset.
    sieve: Vec<u64>,
    /// Contains the sets containing a `e` with `sieve[e] != 0`
    touched: Vec<usize>,
}

/// A part of a partition.
/// The part `s` is `elems[s.begin..s.end]`
/// The subset of `s` with non-zero `sieve` is `elems[s.begin..s.mid]`
#[derive(Clone, Debug, Copy)]
struct Set {
    begin: usize,
    end: usize,
    mid: usize,
}

impl Set {
    const fn len(&self) -> usize {
        self.end - self.begin
    }
}

impl Partition {
    /// Return the partition of `{0..n-1}` with one part.
    pub fn simple(size: usize) -> Self {
        Self::with_singletons(size, 0)
    }

    /// Return the partition of `{0}...{k}{k+1..n-1}` with k+1 parts.
    pub fn with_singletons(size: usize, k: usize) -> Self {
        assert!(k <= size);
        let mut sets = Vec::with_capacity(size);
        let num_parts = if k == size { k } else { k + 1 };
        let mut set_id = if size > 0 {
            vec![num_parts - 1; size]
        } else {
            Vec::new()
        };
        // Create the singletons
        for (i, set_i) in set_id.iter_mut().enumerate().take(k) {
            sets.push(Set {
                begin: i,
                mid: i,
                end: i + 1,
            });
            *set_i = i;
        }
        // Create the set with the rest, if it is non-empty
        if size > k {
            sets.push(Set {
                begin: k,
                mid: k,
                end: size,
            });
        }
        Self {
            elems: (0..size).collect(),
            rev_elems: (0..size).collect(),
            set_id,
            sets,
            sieve: vec![0; size],
            touched: Vec::with_capacity(size),
        }
    }

    /// Exchange the elements of indices `i1` and `i2` of a partition.
    #[inline]
    fn swap(&mut self, i1: usize, i2: usize) {
        if i1 != i2 {
            let e1 = self.elems[i1];
            let e2 = self.elems[i2];
            self.elems[i1] = e2;
            self.elems[i2] = e1;
            self.rev_elems[e1] = i2;
            self.rev_elems[e2] = i1;
        }
    }

    /// Add `x` to the sieve value of `e` and update the related administration
    pub fn sieve(&mut self, e: usize, x: u64) {
        if self.sieve[e] == 0 {
            let set = &mut self.sets[self.set_id[e]];
            if set.len() == 1 {
                // A part of size one cannot be split further: we ignore the call
                return;
            };
            if set.mid == set.begin {
                self.touched.push(self.set_id[e]);
            };
            // update the partition so that `e` is in `elems[s.begin..s.mid]`
            let new_pos = set.mid;
            set.mid += 1;
            self.swap(new_pos, self.rev_elems[e]);
        }
        self.sieve[e] += x;
    }

    /// Split the partitions according to the values in sieve
    /// Call callback on each partition created
    pub fn split<F>(&mut self, mut callback: F)
    where
        F: FnMut(usize),
    {
        self.touched.sort_unstable();
        for &s in &self.touched {
            let set = self.sets[s];
            let begin = set.begin;
            let end = set.mid;
            self.sets[s].mid = begin;
            let sieve = &self.sieve;

            self.elems[begin..end].sort_by_key(|e| sieve[*e]);

            let mut current_set = s;
            let mut current_key = self.sieve[self.elems[set.end - 1]];

            for i in (begin..end).rev() {
                let elem_i = self.elems[i];
                if self.sieve[elem_i] != current_key {
                    current_key = self.sieve[elem_i];
                    self.sets[current_set].begin = i + 1;
                    self.sets[current_set].mid = i + 1;
                    self.sets.push(Set {
                        begin,
                        mid: begin,
                        end: i + 1,
                    });
                    current_set = self.num_parts() - 1;
                    callback(current_set);
                }
                self.set_id[elem_i] = current_set;
                self.rev_elems[elem_i] = i;
                self.sieve[elem_i] = 0;
            }
        }
        self.touched.clear();
    }

    /// Separate elements with different keys.
    pub fn refine_by_value<F>(&mut self, key: &[u64], callback: F)
    where
        F: FnMut(usize),
    {
        for (i, &key) in key.iter().enumerate() {
            self.sieve(i, key);
        }
        self.split(callback);
    }

    fn parent_set(&self, s: usize) -> usize {
        self.set_id[self.elems[self.sets[s].end]]
    }

    /// Delete the last sets created until there are only nsets left
    pub fn undo(&mut self, nparts: usize) {
        for s in (nparts..self.num_parts()).rev() {
            let set = self.sets[s];
            let parent = self.parent_set(s);
            for e in &mut self.elems[set.begin..set.end] {
                self.set_id[*e] = parent;
            }
            self.sets[parent].begin = set.begin;
            self.sets[parent].mid = set.begin;
        }
        self.sets.truncate(nparts);
    }

    #[inline]
    /// Return the slice of the elements of the part `part`.
    pub fn part(&self, part: usize) -> &[usize] {
        &self.elems[self.sets[part].begin..self.sets[part].end]
    }

    #[inline]
    /// Number of parts in the partition.
    pub fn num_parts(&self) -> usize {
        self.sets.len()
    }

    /// Number of elememts in the partition.
    pub fn num_elems(&self) -> usize {
        self.elems.len()
    }

    #[inline]
    /// Return `true` if the partition contains only cells of size 1.
    pub fn is_discrete(&self) -> bool {
        self.elems.len() == self.sets.len()
    }

    /// Refine the partition such that `e` is in a cell of size 1.
    pub fn individualize(&mut self, e: usize) -> Option<usize> {
        let s = self.set_id[e];
        if self.sets[s].end - self.sets[s].begin >= 2 {
            let i = self.rev_elems[e];
            self.swap(i, self.sets[s].begin);
            let delimiter = self.sets[s].begin + 1;
            //
            let new_set = self.num_parts();
            self.set_id[e] = new_set;
            let new_begin = self.sets[s].begin;
            self.sets.push(Set {
                begin: new_begin,
                mid: new_begin,
                end: delimiter,
            });
            self.sets[s].begin = delimiter;
            self.sets[s].mid = delimiter;
            Some(new_set)
        } else {
            None
        }
    }

    /// If the partition conatains only cells of size 1, returns the bijection
    /// that map each element to the position of the corresponding cell.
    pub fn as_bijection(&self) -> Option<&[usize]> {
        if self.is_discrete() {
            Some(&self.rev_elems)
        } else {
            None
        }
    }

    /// Return the list of the cell in the order of the partition.
    pub const fn parts(&self) -> PartsIterator<'_> {
        PartsIterator {
            partition: self,
            pos: 0,
        }
    }

    /// Panic if the data structure does not correspond to a partition.
    fn check_consistent(&self) {
        let n = self.elems.len();
        assert_eq!(self.rev_elems.len(), n);
        for i in 0..n {
            assert_eq!(self.rev_elems[self.elems[i]], i);
            assert_eq!(self.elems[self.rev_elems[i]], i);
        }
        for (i, set) in self.sets.iter().enumerate() {
            assert!(set.begin < set.end);
            assert!(set.begin <= set.mid);
            assert!(set.mid <= set.end);
            for j in set.begin..set.end {
                assert_eq!(self.set_id[self.elems[j]], i);
            }
            for j in set.begin..set.mid {
                assert!(self.sieve[j] != 0);
            }
            for j in set.mid..set.end {
                assert_eq!(self.sieve[j], 0);
            }
        }
    }
}

pub struct PartsIterator<'a> {
    partition: &'a Partition,
    pos: usize,
}

impl<'a> Iterator for PartsIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(&e) = self.partition.elems.get(self.pos) {
            let s = self.partition.set_id[e];
            self.pos = self.partition.sets[s].end;
            Some(s)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n_parts = self.partition.num_parts();
        let lower = if self.pos < n_parts {
            n_parts - self.pos
        } else {
            0
        };
        (lower, Some(n_parts))
    }
}

impl Display for Partition {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        self.check_consistent();
        write!(f, "(")?;
        for i in 0..self.elems.len() {
            if i > 0 {
                if self.sets[self.set_id[self.elems[i]]].begin == i {
                    write!(f, ")(")?;
                } else {
                    write!(f, ",")?;
                }
            }
            write!(f, "{}", self.elems[i])?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

pub trait Canonize
where
    Self: Sized + Ord + Clone,
{
    fn size(&self) -> usize;

    fn apply_morphism(&self, p: &[usize]) -> Self;

    fn invariant_coloring(&self) -> Option<Vec<u64>> {
        None
    }
    fn invariant_neighborhood(&self, _u: usize) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn canonical(&self) -> Self {
        self.canonical_typed(0)
    }
    fn canonical_typed(&self, sigma: usize) -> Self {
        let partition = Partition::with_singletons(self.size(), sigma);
        canonical_constraint(self, partition)
    }

    #[inline]
    fn morphism_to_canonical(&self) -> Vec<usize> {
        self.morphism_to_canonical_typed(0)
    }
    fn morphism_to_canonical_typed(&self, sigma: usize) -> Vec<usize> {
        assert!(sigma <= self.size());
        let partition = Partition::with_singletons(self.size(), sigma);
        morphism_to_canonical_constraint(self, partition)
    }

    #[inline]
    fn automorphisms(&self) -> AutomorphismIterator<Self> {
        self.stabilizer(0)
    }

    #[inline]
    fn stabilizer(&self, sigma: usize) -> AutomorphismIterator<Self> {
        let mut partition = Partition::simple(self.size());
        for i in 0..sigma {
            let _ = partition.individualize(i);
        }
        AutomorphismIterator::new(self, partition)
    }
}

fn target_selector(part: &Partition) -> Option<usize> {
    let mut min = usize::max_value();
    let mut arg_min = None;
    for i in part.parts() {
        let length = part.part(i).len();
        if 2 <= length && (length < min) {
            min = length;
            arg_min = Some(i);
        }
    }
    arg_min
}

fn precompute_invariant<F>(g: &F) -> Vec<Vec<Vec<usize>>>
where
    F: Canonize,
{
    let n = g.size();
    let mut res = Vec::with_capacity(n);
    for i in 0..n {
        res.push(g.invariant_neighborhood(i));
    }
    res
}

fn refine(partition: &mut Partition, invariants: &[Vec<Vec<usize>>], new_part: Option<usize>) {
    if !partition.is_discrete() {
        let n = partition.num_elems();
        assert!(n >= 2);
        let invariant_size = invariants[0].len();
        debug_assert!(invariants.iter().all(|v| v.len() == invariant_size));
        // Stack contains the new created partitions
        let mut stack: Vec<_> = match new_part {
            Some(p) => vec![p],
            None => partition.parts().collect(),
        };
        // base
        let max_step = ((n + 1 - partition.num_parts()) as u64).pow(invariant_size as u32);
        let threshold = u64::max_value() / max_step; //
        let mut part_buffer = Vec::new();
        while !stack.is_empty() && !partition.is_discrete() {
            let mut weight = 1; // multiplicator to make the values in the sieve unique
            while let Some(part) = stack.pop() {
                part_buffer.clear();
                part_buffer.extend_from_slice(partition.part(part));
                let factor = (part_buffer.len() + 1) as u64;
                for i in 0..invariant_size {
                    weight *= factor;
                    // Compute sieve
                    for &u in &part_buffer {
                        for &v in &invariants[u][i] {
                            partition.sieve(v, weight);
                        }
                    }
                }
                if weight > threshold {
                    break;
                };
            }
            partition.split(|new| {
                stack.push(new);
            });
        }
    }
}

/// Return the first index on which `u` and `v` differ.
fn fca(u: &[usize], v: &[usize]) -> usize {
    let mut i = 0;
    while i < u.len() && i < v.len() && u[i] == v[i] {
        i += 1;
    }
    i
}

/// Node of the tree of the normalization process
#[derive(Clone, Debug)]
struct IsoTreeNode {
    nparts: usize,
    children: Vec<usize>,
    inv: Rc<Vec<Vec<Vec<usize>>>>,
}

impl IsoTreeNode {
    fn root<F: Canonize>(partition: &mut Partition, g: &F) -> Self {
        let inv = Rc::new(precompute_invariant(g));
        if let Some(coloring) = g.invariant_coloring() {
            partition.refine_by_value(&coloring, |_| {});
        }
        Self::new(partition, inv, None)
    }
    fn new(
        partition: &mut Partition,
        inv: Rc<Vec<Vec<Vec<usize>>>>,
        new_part: Option<usize>,
    ) -> Self {
        refine(partition, &inv, new_part);
        Self {
            children: match target_selector(partition) {
                Some(set) => partition.part(set).to_vec(),
                None => Vec::new(),
            },
            nparts: partition.num_parts(),
            inv,
        }
    }
    fn explore(&self, v: usize, pi: &mut Partition) -> Self {
        debug_assert!(self.is_restored(pi));
        let new_part = pi.individualize(v);
        Self::new(pi, self.inv.clone(), new_part)
    }
    // Should never be used
    fn dummy() -> Self {
        Self {
            children: Vec::new(),
            nparts: 1,
            inv: Rc::new(Vec::new()),
        }
    }
    fn restore(&self, partition: &mut Partition) {
        partition.undo(self.nparts);
    }
    fn is_restored(&self, partition: &Partition) -> bool {
        partition.num_parts() == self.nparts
    }
}

/// Normal form of `g` under the action of isomorphisms that
/// stabilize the parts of `partition`.
fn canonical_constraint<F>(g: &F, mut partition: Partition) -> F
where
    F: Canonize,
{
    // contains the images of `g` already computed associated to the path to the corresponding leaf
    let mut zeta: BTreeMap<F, Vec<usize>> = BTreeMap::new();
    let mut tree = Vec::new(); // A stack of IsoTreeNode
    let mut path = Vec::new(); // Current path as a vector of chosen vertices
    let mut node = IsoTreeNode::root(&mut partition, g);
    loop {
        // If we have a leaf, treat it
        if let Some(phi) = partition.as_bijection() {
            match zeta.entry(g.apply_morphism(phi)) {
                Occupied(entry) =>
                // We are in a branch isomorphic to a branch we explored
                {
                    let k = fca(entry.get(), &path) + 1;
                    tree.truncate(k);
                    path.truncate(k);
                }
                Vacant(entry) => {
                    let _ = entry.insert(path.clone());
                }
            }
        };
        // If there is a child, explore it
        if let Some(u) = node.children.pop() {
            let new_node = node.explore(u, &mut partition);
            tree.push(node);
            path.push(u);
            node = new_node;
        } else {
            match tree.pop() {
                Some(n) => {
                    node = n;
                    let _ = path.pop();
                    node.restore(&mut partition); // backtrack the partition
                }
                None => break,
            }
        };
    }
    let (g_max, _) = zeta.into_iter().next_back().unwrap(); // return the largest image found
    g_max
}

/// Iterator on the automorphisms of a combinatorial structure.
#[derive(Clone, Debug)]
pub struct AutomorphismIterator<F> {
    tree: Vec<IsoTreeNode>,
    node: IsoTreeNode,
    partition: Partition,
    g: F,
}

impl<F: Canonize> AutomorphismIterator<F> {
    /// Iterator on the automorphisms of `g` that preserve `partition`.
    fn new(g: &F, mut partition: Partition) -> Self {
        debug_assert!(g == &canonical_constraint(g, partition.clone()));
        Self {
            tree: vec![IsoTreeNode::root(&mut partition, g)],
            partition,
            node: IsoTreeNode::dummy(), // Dummy node that will be unstacked at the first iteration
            g: g.clone(),
        }
    }
}

impl<F: Canonize> Iterator for AutomorphismIterator<F> {
    type Item = Vec<usize>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(u) = self.node.children.pop() {
                let new_node = self.node.explore(u, &mut self.partition);
                let old_node = std::mem::replace(&mut self.node, new_node);
                self.tree.push(old_node);
            } else {
                match self.tree.pop() {
                    Some(n) => {
                        n.restore(&mut self.partition);
                        self.node = n;
                    }
                    None => return None,
                }
            }
            if let Some(phi) = self.partition.as_bijection() {
                if self.g.apply_morphism(phi) == self.g {
                    return Some(phi.to_vec());
                }
            };
        }
    }
}

/// Return a morphism `phi`
/// such that `g.apply_morphism(phi) = canonical_constraint(g, partition)`.
fn morphism_to_canonical_constraint<F>(g: &F, mut partition: Partition) -> Vec<usize>
where
    F: Canonize,
{
    // initialisation
    let mut tree = Vec::new();
    let mut node = IsoTreeNode::root(&mut partition, g);
    let mut max = None;
    let mut phimax = Vec::new();
    loop {
        if let Some(phi) = partition.as_bijection() {
            // If node is a leaf
            let phi_g = Some(g.apply_morphism(phi));
            if phi_g > max {
                max = phi_g;
                phimax = phi.to_vec();
            }
        };
        if let Some(u) = node.children.pop() {
            let new_node = node.explore(u, &mut partition);
            tree.push(node);
            node = new_node;
        } else {
            match tree.pop() {
                Some(n) => {
                    n.restore(&mut partition);
                    node = n;
                }
                None => break,
            }
        }
    }
    phimax
}

/// Tests
#[cfg(test)]
mod test {
    use super::*;

    fn group_test<M: RangeBounds<i64> + Clone, E: RangeBounds<i64> + Clone>(
        m_range: M,
        e_range: E,
        cases: u64,
    ) -> i64 {
        let mut total_score = 0;
        const OFFSET: u64 = 1;
        let mut me = (0..cases)
            .map(|i| {
                let seed = OFFSET + i;
                let mut xorshift = XorShift::with_seed(seed);
                (
                    xorshift.rand_range(m_range.clone()),
                    xorshift.rand_range(e_range.clone()),
                    xorshift,
                )
            })
            .collect::<Vec<_>>();
        me.sort_by_key(|(m, e, _)| (*e, *m));
        for (m, e, mut xorshift) in me {
            let (score, m, e, error, n, t) = score(m, e, false, &mut xorshift, None, |m, e| {
                Solver::build(m, e, None)
            });
            println!(
                "type:{:>12} score:{:>10} m:{:>3} e:{:>3} n:{:>3} len:{:>4} error:{:>3}",
                t,
                score,
                m,
                e,
                n,
                n * (n - 1) / 2,
                error
            );
            total_score += score;
        }
        total_score
    }

    // (score, m, e, error, n)
    fn score<F: FnMut(usize, i64) -> Solver>(
        m: i64,
        e: i64,
        stdout: bool,
        xorshift: &mut XorShift,
        mut n: Option<usize>,
        mut solve_builder: F,
    ) -> (i64, i64, i64, i64, usize, String) {
        let mut error = 0i64;
        let mut actual = None;
        let mut graph = Vec::new();
        let mut io = IODebug::new(
            &format!("{} {}", m, e as f64 * 0.01),
            stdout,
            |outer: &mut ReaderFromStr, inner: &mut ReaderFromStr| {
                if n.is_none() || graph.is_empty() {
                    n = Some(outer.v::<usize>());
                    for _ in 0..m {
                        graph.push(Selection::new(n.unwrap(), &outer.digits()));
                    }
                } else {
                    // 採点
                    let result = outer.v::<usize>();
                    if stdout {
                        println!("result:{} actual{}", result, actual.unwrap());
                    }
                    if actual != Some(result) {
                        error += 1;
                    }
                }
                // 出題
                {
                    let n = n.unwrap();
                    let select = xorshift.rand_range(0..m) as usize;
                    let g = {
                        // 出力グラフの生成
                        let mut nodes = (0..n).collect::<Vec<_>>();
                        xorshift.shuffle(&mut nodes);
                        let mut g = graph[select].apply_morphism(&nodes);
                        for i in 0..g.n * (g.n - 1) / 2 {
                            let b = (xorshift.rand(100) as i64) < e;
                            let r = g.b[i] ^ b;
                            g.b.set(i, r);
                        }
                        g
                    };
                    inner.out(g);
                    actual = Some(select);
                    inner.flush();
                }
            },
        );
        let solver_type = {
            // 解答
            let (m, e) = io.v2::<usize, f64>();
            let solver = solve_builder(m, (e * 100.0 + 0.5) as i64);
            let (n, graphs) = solver.graphs();
            io.out(n.ln());
            for g in graphs {
                io.out(g);
            }
            io.flush();
            for _ in 0..TEST_CASES {
                let h = io.digits();
                io.out(solver.query(&Selection::new(n, &h)).ln());
                io.flush();
            }
            match solver {
                Solver::OneBlock(_) => "oneblock  ",
                Solver::UnionStatWithError(_) => "union_nodes",
                Solver::Random { .. } => "randomized",
            }
        };
        (
            (1_000_000_000.0 * 0.9f64.powi(error as i32) / n.unwrap() as f64).round() as i64,
            m,
            e,
            error,
            n.unwrap(),
            solver_type.to_string(),
        )
    }
    #[test]
    fn zero_test() {
        let mut xorshift = XorShift::with_seed(1);
        // 最後のテストの結果: 8,846,153,858
        let mut total = 0;
        for m in 10..=100 {
            let (score, m, _e, error, n, _) = score(m, 0, false, &mut xorshift, None, |m, e| {
                Solver::build_union_stat_with_error(m, e)
            });
            total += score;
            println!("m: {}, n: {}, error:{} n_total: {}", m, n, error, score)
        }
        println!("non error test; score: {}", total * 50 / 91);
    }

    #[test]
    fn graph_test() {
        let graph = Selection::new(5, &vec![1, 0, 0, 1, 0, 0, 0, 1, 1, 1]); // 0-1, 0-4, 2-3, 2-4, 3-4
        println!("{:?}", graph);
        let mut c = 0;
        for i in 0..5 {
            for j in i + 1..5 {
                assert_eq!(c, graph.index(i, j));
                c += 1;
            }
        }
        let mut graph2 = Selection::empty(5);
        graph2.set_true(0, 1);
        graph2.set_true(0, 4);
        graph2.set_true(2, 3);
        graph2.set_true(2, 4);
        graph2.set_true(3, 4);
        dbg!(&graph2);
    }

    #[test]
    fn canonical_test() {
        let mut xorshift = XorShift::default();
        let mut c = 0;
        for _ in 0..100000 {
            let p = vec![
                xorshift.rand(u64::max_value()),
                xorshift.rand(u64::max_value()),
            ];
            let graph = Selection::new(
                16,
                &(0..120)
                    .map(|i| usize::from(p[i / 64] >> (i % 64) & 1 == 1))
                    .collect::<Vec<_>>(),
            );
            let can = graph.canonical();
            c += can.n;
        }
        println!("{}", c);
    }

    #[test]
    fn param() {
        for e in 18..=40 {
            let mut res = Vec::new();
            let mut n = 5;
            for m in 10..=100 {
                chmax!(n, m);
                let mut best_score = 0;
                let mut best_n = 0;
                for _ in 0..30 {
                    let mut r = Vec::new();
                    for t in 1..=15 {
                        let mut xorshift = XorShift::with_seed((m * e * t) as u64);
                        let (s, _m, _e, _error, _n, _) =
                            score(m, e, false, &mut xorshift, None, |m, e| {
                                Solver::build_one_block(m, e, Some(n as usize))
                            });
                        r.push(s);
                    }
                    r.sort();
                    if chmax!(best_score, r[15 / 2]) {
                        best_n = n;
                    }
                    if r[15 / 2] < best_n {
                        break;
                    }
                    n += 1;
                    if n > 100 {
                        break;
                    }
                }
                n = m;
                res.push(best_n);
                dbg!(e, m, best_n, best_score);
            }
            println!("vec![{}],", res.join(", "))
        }
    }

    #[test]
    fn sequence_test() {
        let mut sum = 0;
        for e in (26..=40) {
            for m in 10..=100 {
                let mut res = Vec::new();
                for t in 1..=3 {
                    let mut xorshift = XorShift::with_seed((m * e * t) as u64);
                    let (s, _m, _e, _ue, _n, _) =
                        score(m, e, false, &mut xorshift, None, |m, e| {
                            Solver::build_one_block(m, e, None)
                        });
                    res.push(s);
                }
                res.sort();
                let r = res[1];

                let mut res = Vec::new();
                for t in 1..=3 {
                    let mut xorshift = XorShift::with_seed((m * e * t) as u64);
                    let (s, _m, _e, _oe, _n, _) =
                        score(m, e, false, &mut xorshift, None, |m, e| {
                            Solver::build_union_stat_with_error(m, e)
                        });
                    res.push(s);
                }
                let t = res[1];

                let mut res = Vec::new();
                for u in 1..=3 {
                    let mut xorshift = XorShift::with_seed((m * e * u) as u64);
                    let (s, _m, _e, _oe, _n, _) =
                        score(m, e, false, &mut xorshift, None, |m, _e| Solver::random(m));
                    res.push(s);
                }
                let u = res[1];
                sum += r;
                let a = max!(t, r, u);
                println!(
                    "union:{:>5} oneblock: {:>5} random: {:>5} m:{:>3} e:{:>3}",
                    t * 10000 / a,
                    r * 10000 / a,
                    u * 10000 / a,
                    m,
                    e,
                );
            }
        }
        dbg!(sum);
        // n = 5        [src/ahc.rs:1071] sum = 843986256
    }

    #[test]
    fn debug_case_single_case() {
        let mut xorshift = XorShift::with_seed(1);
        let (m, e) = (13, 5);

        for _ in 0..1 {
            let (_, _) = (xorshift.rand_range(10..=100), xorshift.rand_range(0..=40));
            let (score, m, e, error, n, t) = score(m, e, true, &mut xorshift, None, |m, e| {
                Solver::build_one_block(m, e, None)
            });
            println!(
                "type:{} score:{:>10} m:{} e:{} n:{} len:{} error:{}",
                t,
                score,
                m,
                e,
                n,
                n * (n - 1) / 2,
                error
            );
        }
        123;
    }

    #[test]
    fn random_test() {
        // 最後のテストの結果: 414007560
        dbg!(
            "all",
            group_test(10..=100, 0..=40, 500) / super::TEST_CASES * 50 / 10
        );
    }
}
