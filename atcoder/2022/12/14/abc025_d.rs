pub use writer_impl::{AddLineTrait, BitsTrait, JoinTrait, WriterToStdout, WriterTrait};
# [rustfmt :: skip ] mod writer_impl {use super :: {stdout , BufWriter , Display , Integral , Write } ; pub trait WriterTrait {fn out < S : Display > (& mut self , s : S ) ; fn flush (& mut self ) ; } pub trait AddLineTrait {fn ln (& self ) -> String ; } impl < D : Display > AddLineTrait for D {fn ln (& self ) -> String {self . to_string () + "\n" } } pub trait JoinTrait {fn join (self , separator : & str ) -> String ; } impl < D : Display , I : IntoIterator < Item = D > > JoinTrait for I {fn join (self , separator : & str ) -> String {let mut buf = String :: new () ; self . into_iter () . fold ("" , | sep , arg | {buf . push_str (& format ! ("{}{}" , sep , arg ) ) ; separator } ) ; buf + "\n" } } pub trait BitsTrait {fn bits (self , length : Self ) -> String ; } impl < I : Integral > BitsTrait for I {fn bits (self , length : Self ) -> String {let mut buf = String :: new () ; let mut i = I :: zero () ; while i < length {buf . push_str (& format ! ("{}" , self >> i & I :: one () ) ) ; i += I :: one () ; } buf + "\n" } } # [derive (Clone , Debug , Default ) ] pub struct WriterToStdout {buf : String , } impl WriterTrait for WriterToStdout {fn out < S : Display > (& mut self , s : S ) {self . buf . push_str (& s . to_string () ) ; } fn flush (& mut self ) {let stdout = stdout () ; let mut writer = BufWriter :: new (stdout . lock () ) ; write ! (writer , "{}" , self . buf ) . expect ("Failed to write." ) ; let _ = writer . flush () ; self . buf . clear () ; } } }
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

pub trait Mod: Copy + Clone + Debug {
    const MOD: u32;
    const MOD_INV: u32;
    const R: u32;
    const R_POW2: u32;
}
#[derive(Copy, Clone, Eq, PartialEq, Default)]
pub struct ModInt<M: Mod>(u32, PhantomData<fn() -> M>);
mod mod_int_impl {
    use super::{
        Add, AddAssign, Debug, Display, Div, DivAssign, Formatter, FromStr, Mod, ModInt, Mul,
        MulAssign, Neg, One, PhantomData, Pow, Sub, SubAssign, Sum, Zero,
    };
    use std::num::ParseIntError;
    impl<M: Mod> Mod for ModInt<M> {
        const MOD: u32 = M::MOD;
        const MOD_INV: u32 = M::MOD_INV;
        const R_POW2: u32 = M::R_POW2;
        const R: u32 = M::R;
    }
    impl<M: Mod> ModInt<M> {
        #[inline]
        pub fn new(mut n: u32) -> Self {
            if n >= Self::MOD {
                n = n.rem_euclid(Self::MOD);
            }
            Self(Self::mrmul(n, Self::R_POW2), PhantomData)
        }
        pub fn comb(n: i64, mut r: i64) -> Self {
            assert!(0 <= r && r <= n);
            if r > n - r {
                r = n - r;
            }
            let (mut ret, mut rev) = (Self::one(), Self::one());
            for k in 0..r {
                ret *= n - k;
                rev *= r - k;
            }
            ret / rev
        }
        #[inline]
        pub fn mrmul(ar: u32, br: u32) -> u32 {
            let t: u64 = (ar as u64) * (br as u64);
            let (t, f) = ((t >> 32) as u32).overflowing_sub(
                ((((t as u32).wrapping_mul(Self::MOD_INV) as u128) * Self::MOD as u128) >> 32)
                    as u32,
            );
            if f {
                t.wrapping_add(Self::MOD)
            } else {
                t
            }
        }
        #[inline]
        pub fn reduce(self) -> u32 {
            let (t, f) = (((((self.0.wrapping_mul(Self::MOD_INV)) as u128) * (Self::MOD as u128))
                >> 32) as u32)
                .overflowing_neg();
            if f {
                t.wrapping_add(Self::MOD)
            } else {
                t
            }
        }
    }
    impl<M: Mod> Pow for ModInt<M> {
        #[inline]
        fn pow(mut self, mut e: i64) -> Self {
            debug_assert!(e > 0);
            let mut t = if e & 1 == 0 { M::R } else { self.0 };
            e >>= 1;
            while e != 0 {
                self.0 = Self::mrmul(self.0, self.0);
                if e & 1 != 0 {
                    t = Self::mrmul(t, self.0);
                }
                e >>= 1;
            }
            self.0 = t;
            self
        }
    }
    impl<M: Mod, Rhs: Into<Self>> Add<Rhs> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn add(mut self, rhs: Rhs) -> Self {
            self += rhs;
            self
        }
    }
    impl<M: Mod, Rhs: Into<Self>> AddAssign<Rhs> for ModInt<M> {
        #[inline]
        fn add_assign(&mut self, rhs: Rhs) {
            let rhs = rhs.into();
            self.0 = self.0 + rhs.0;
            if self.0 >= Self::MOD {
                self.0 -= Self::MOD
            }
        }
    }
    impl<M: Mod> Neg for ModInt<M> {
        type Output = Self;
        #[inline]
        fn neg(self) -> Self {
            Self::zero() - self
        }
    }
    impl<M: Mod, Rhs: Into<Self>> Sub<Rhs> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn sub(mut self, rhs: Rhs) -> Self {
            self -= rhs;
            self
        }
    }
    impl<M: Mod, Rhs: Into<Self>> SubAssign<Rhs> for ModInt<M> {
        #[inline]
        fn sub_assign(&mut self, rhs: Rhs) {
            let rhs = rhs.into();
            self.0 = if self.0 >= rhs.0 {
                self.0 - rhs.0
            } else {
                self.0 + Self::MOD - rhs.0
            }
        }
    }
    impl<M: Mod, Rhs: Into<Self>> Mul<Rhs> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn mul(mut self, rhs: Rhs) -> Self {
            self *= rhs.into();
            self
        }
    }
    impl<M: Mod, Rhs: Into<Self>> MulAssign<Rhs> for ModInt<M> {
        #[inline]
        fn mul_assign(&mut self, rhs: Rhs) {
            self.0 = Self::mrmul(self.0, rhs.into().0)
        }
    }
    impl<M: Mod, Rhs: Into<Self>> Div<Rhs> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn div(mut self, rhs: Rhs) -> Self {
            self /= rhs;
            self
        }
    }
    impl<M: Mod, Rhs: Into<Self>> DivAssign<Rhs> for ModInt<M> {
        #[inline]
        fn div_assign(&mut self, rhs: Rhs) {
            *self *= rhs.into().pow(Self::MOD as i64 - 2)
        }
    }
    impl<M: Mod> Display for ModInt<M> {
        #[inline]
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "{}", self.reduce())
        }
    }
    impl<M: Mod> Debug for ModInt<M> {
        #[inline]
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "{}", self.reduce())
        }
    }
    impl<M: Mod> Sum for ModInt<M> {
        #[inline]
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::zero(), |x, a| x + a)
        }
    }
    impl<M: Mod> FromStr for ModInt<M> {
        type Err = ParseIntError;
        #[inline]
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(Self::new(s.parse::<u32>()?))
        }
    }
    macro_rules ! impl_integral {($ ($ ty : ty ) ,* ) => {$ (impl < M : Mod > From <$ ty > for ModInt < M > {# [inline ] fn from (i : $ ty ) -> Self {Self :: new (i . rem_euclid (Self :: MOD as $ ty ) as u32 ) } } ) * } ; }
    impl_integral!(i32, i64, i128, isize, u32, u64, u128, usize);
    impl<M: Mod> From<ModInt<M>> for i64 {
        #[inline]
        fn from(m: ModInt<M>) -> Self {
            m.reduce() as i64
        }
    }
    impl<M: Mod> Zero for ModInt<M> {
        #[inline]
        fn zero() -> Self {
            Self(0, PhantomData)
        }
    }
    impl<M: Mod> One for ModInt<M> {
        #[inline]
        fn one() -> Self {
            Self(M::R, PhantomData)
        }
    }
}

pub use mod_1_000_000_007_impl::{mi, Mi, Mod1_000_000_007};
mod mod_1_000_000_007_impl {
    use super::{Mod, ModInt};
    pub fn mi(i: i64) -> Mi {
        Mi::from(i)
    }
    pub type Mi = ModInt<Mod1_000_000_007>;
    #[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
    pub struct Mod1_000_000_007;
    impl Mod for Mod1_000_000_007 {
        const MOD: u32 = 1_000_000_007;
        const MOD_INV: u32 = 2_068_349_879;
        const R: u32 = 294_967_268;
        const R_POW2: u32 = 582_344_008;
    }
}

pub fn solve<IO: ReaderTrait + WriterTrait>(io: &mut IO) {
    let v = io.vec::<usize>(25);
    let mut done = vec![None; 25];
    for i in 0..25 {
        if v[i] != 0 {
            done[v[i] - 1] = Some(i);
        }
    }
    let mut dp = vec![Mi::zero(); 1 << 25];
    dp[0] = Mi::one();
    for i in 0..25 {
        for stat in 0usize..1 << 25 {
            if (stat.count_ones() as usize) != i {
                continue;
            }
            if let Some(p) = done[i] {
                if check(p, stat) {
                    dp[stat | (1 << p)] = dp[stat | (1 << p)] + dp[stat];
                }
            } else {
                for p in 0..25 {
                    if check(p, stat) {
                        dp[stat | (1 << p)] = dp[stat | (1 << p)] + dp[stat];
                    }
                }
            }
        }
    }
    io.out(dp[dp.len() - 1].ln());
}

#[inline]
fn check(p: usize, stat: usize) -> bool {
    if stat >> p & 1 > 0 {
        return false;
    }
    let (x, y) = to_xy(p);
    if 0 < x && x < 4 && (exists(x - 1, y, stat) ^ exists(x + 1, y, stat)) {
        return false;
    }
    if 0 < y && y < 4 && (exists(x, y - 1, stat) ^ exists(x, y + 1, stat)) {
        return false;
    }
    true
}

#[inline]
fn exists(x: usize, y: usize, stat: usize) -> bool {
    let p = to_p(x, y);
    stat >> p & 1 > 0
}

#[inline]
fn to_xy(p: usize) -> (usize, usize) {
    (p % 5, p / 5)
}

#[inline]
fn to_p(x: usize, y: usize) -> usize {
    y * 5 + x
}

#[test]
fn test() {
    let test_suits = vec![
        "0 0 15 2 7
        0 0 16 1 22
        20 25 4 19 0
        3 23 9 18 10
        17 0 5 21 8
        ",
        "10 14 13 15 11
            16 0 17 0 18
            0 19 0 20 9
            21 12 22 0 23
            0 24 0 25 0
            ",
        "1 2 3 4 5
            6 7 8 9 10
            11 12 13 14 15
            16 17 18 19 20
            0 0 0 0 0
            ",
        "1 25 2 24 3
        23 4 22 5 21
        6 20 7 19 8
        18 9 17 10 16
        11 15 12 14 13
        ",
    ];
    for suit in test_suits {
        std::thread::Builder::new()
            .name("extend stack size".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(move || {
                let mut io = IODebug::new(
                    suit,
                    true,
                    |outer: &mut ReaderFromStr, inner: &mut ReaderFromStr| {
                        inner.out(outer.v::<String>())
                    },
                );
                solve(&mut io);
                io.flush();
            })
            .unwrap()
            .join()
            .unwrap()
    }
}
