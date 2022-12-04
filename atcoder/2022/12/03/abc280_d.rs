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

pub trait PollardRho {
    fn prime_factorize(&self) -> Vec<u64>;
}
#[allow(clippy::many_single_char_names)]
impl PollardRho for u64 {
    fn prime_factorize(&self) -> Vec<u64> {
        if self <= &1 {
            return Vec::new();
        }
        fn find_cycle_by_brent(n: u64) -> u64 {
            if n % 2 == 0 {
                return 2;
            }
            if n.is_prime() {
                return n;
            }
            let mul = MontgomeryReduction::new(n);
            const LIMIT: u64 = 256;
            for epoch in 1..LIMIT {
                let prng_next = |x| mul.add(mul.mrmul(x, x), epoch);
                let m = 1 << ((0u64.leading_zeros() - n.leading_zeros()) >> 3);
                let (mut y, mut r, mut q, mut g) = (2, 1, 1, 1);
                let (mut x, mut ys) = (0, 0);
                while g == 1 {
                    x = y;
                    for _ in 0..r {
                        y = prng_next(y);
                    }
                    let mut k = 0;
                    while k < r && g == 1 {
                        ys = y;
                        for _ in 0..min(m, r - k) {
                            y = prng_next(y);
                            q = mul.mrmul(q, max(x, y) - min(x, y));
                        }
                        g = Gcd::op(&q, &n);
                        k += m;
                    }
                    r <<= 1;
                }
                if g == n {
                    g = 1;
                    while g == 1 {
                        ys = prng_next(ys);
                        g = Gcd::op(&(max(x, ys) - min(x, ys)), &n);
                    }
                }
                if g < n {
                    return g;
                }
            }
            panic!("not found cycle.")
        }
        let mut ret = Vec::new();
        let mut t = *self;
        for &p in &[2, 3, 5, 7, 11, 13, 17] {
            if t.is_prime() || t <= p {
                break;
            }
            while t % p == 0 {
                ret.push(p);
                t /= p;
            }
        }
        if t == 1 {
            return ret;
        }
        let p = find_cycle_by_brent(t);
        if t == 1 || p == 1 {
            ret
        } else if p == t {
            ret.push(p);
            ret
        } else {
            ret.append(&mut p.prime_factorize());
            ret.append(&mut (t / p).prime_factorize());
            ret.sort_unstable();
            ret
        }
    }
}

#[derive(Clone, Debug)]
pub struct MontgomeryReduction {
    pub n: u64,
    pub n_inv: u64,
    pub nh: u64,
    pub r: u64,
    pub r_neg: u64,
    pub r_pow2: u64,
    pub d: u64,
    pub k: u32,
}
impl MontgomeryReduction {
    #[inline]
    pub const fn new(n: u64) -> Self {
        let mut n_inv = n;
        n_inv = n_inv.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(n_inv)));
        n_inv = n_inv.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(n_inv)));
        n_inv = n_inv.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(n_inv)));
        n_inv = n_inv.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(n_inv)));
        n_inv = n_inv.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(n_inv)));
        let nh = (n >> 1) + 1;
        let r = n.wrapping_neg() % n;
        let r_neg = n - r;
        let r_pow2 = ((n as u128).wrapping_neg() % (n as u128)) as u64;
        let k = (n - 1).trailing_zeros();
        let d = (n - 1) >> k;
        Self {
            n,
            n_inv,
            nh,
            r,
            r_neg,
            r_pow2,
            d,
            k,
        }
    }
    #[inline]
    pub fn add(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.n);
        debug_assert!(b < self.n);
        let (t, fa) = a.overflowing_add(b);
        let (u, fs) = t.overflowing_sub(self.n);
        if fa || !fs {
            u
        } else {
            t
        }
    }
    #[inline]
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.n);
        debug_assert!(b < self.n);
        let (t, f) = a.overflowing_sub(b);
        if f {
            t.wrapping_add(self.n)
        } else {
            t
        }
    }
    #[inline]
    pub fn generate(&self, a: u64) -> u64 {
        debug_assert!(a < self.n);
        self.mrmul(a, self.r_pow2)
    }
    #[inline]
    pub fn reduce(&self, ar: u64) -> u64 {
        debug_assert!(ar < self.n, "{} {}", self.n, ar);
        let (t, f) = (((((ar.wrapping_mul(self.n_inv)) as u128) * (self.n as u128)) >> 64) as u64)
            .overflowing_neg();
        if f {
            t.wrapping_add(self.n)
        } else {
            t
        }
    }
    #[inline]
    pub fn mrmul(&self, ar: u64, br: u64) -> u64 {
        debug_assert!(ar < self.n);
        debug_assert!(br < self.n);
        let t: u128 = (ar as u128) * (br as u128);
        let (t, f) = ((t >> 64) as u64).overflowing_sub(
            ((((t as u64).wrapping_mul(self.n_inv) as u128) * self.n as u128) >> 64) as u64,
        );
        if f {
            t.wrapping_add(self.n)
        } else {
            t
        }
    }
    #[inline]
    pub fn mul_prim(&self, a: u64, b: u64) -> u64 {
        self.reduce(self.mrmul(self.generate(a), self.generate(b)))
    }
    #[inline]
    pub fn pow(&self, a: u64, mut b: u64) -> u64 {
        debug_assert!(a < self.n);
        let mut ar = self.generate(a);
        let mut t = if b & 1 == 0 { self.r } else { ar };
        b >>= 1;
        while b != 0 {
            ar = self.mrmul(ar, ar);
            if b & 1 != 0 {
                t = self.mrmul(t, ar);
            }
            b >>= 1;
        }
        t
    }
}

pub trait MillerRabin {
    fn is_prime(&self) -> bool;
}
impl MillerRabin for u64 {
    fn is_prime(&self) -> bool {
        if *self < 2 || *self & 1 == 0 {
            return *self == 2;
        }
        let mont = MontgomeryReduction::new(*self);
        let is_composite = |mut checker: u64| -> bool {
            if checker >= *self {
                checker %= self;
            }
            if checker == 0 {
                return false;
            }
            let mut tr = mont.pow(checker, mont.d);
            if tr == mont.r || tr == mont.r_neg {
                return false;
            }
            (1..mont.k).all(|_| {
                tr = mont.mrmul(tr, tr);
                tr != mont.r_neg
            })
        };
        const MILLER_RABIN_BASES_32: [u64; 3] = [2, 7, 61];
        const MILLER_RABIN_BASES_64: [u64; 7] = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];
        if *self < 1 << 32 {
            MILLER_RABIN_BASES_32.iter()
        } else {
            MILLER_RABIN_BASES_64.iter()
        }
        .all(|&checker| !is_composite(checker))
    }
}

pub use gcd_impl::Gcd;
mod gcd_impl {
    use super::{
        swap, Associative, Commutative, Debug, Idempotent, Magma, PhantomData, TrailingZeros,
        Unital, Zero,
    };
    use std::ops::{BitOr, Shl, ShrAssign, SubAssign};
    #[derive(Clone, Debug, Default)]
    pub struct Gcd<S>(PhantomData<fn() -> S>);
    # [rustfmt :: skip ] pub trait GcdNeedTrait : Clone + Copy + Debug + PartialOrd + Zero + BitOr < Output = Self > + ShrAssign + Shl < Output = Self > + SubAssign + TrailingZeros {}
    # [rustfmt :: skip ] impl < S : Clone + Copy + Debug + PartialOrd + Zero + BitOr < Output = S > + ShrAssign + Shl < Output = S > + SubAssign + TrailingZeros > GcdNeedTrait for S {}
    impl<S: GcdNeedTrait> Magma for Gcd<S> {
        type M = S;
        #[inline]
        fn op(x: &S, y: &S) -> S {
            if x == &S::zero() {
                return *y;
            }
            if y == &S::zero() {
                return *x;
            }
            let (mut x, mut y) = (*x, *y);
            let s = (x | y).trailing_zero();
            x >>= x.trailing_zero();
            while {
                y >>= y.trailing_zero();
                if x > y {
                    swap(&mut x, &mut y);
                }
                y -= x;
                y > S::zero()
            } {}
            x << s
        }
    }
    impl<S: GcdNeedTrait> Associative for Gcd<S> {}
    impl<S: GcdNeedTrait> Unital for Gcd<S> {
        fn unit() -> S {
            S::zero()
        }
    }
    impl<S: GcdNeedTrait> Commutative for Gcd<S> {}
    impl<S: GcdNeedTrait> Idempotent for Gcd<S> {}
}

pub fn solve<IO: ReaderTrait + WriterTrait>(io: &mut IO) {
    let k: u64 = io.v();

    let f = k.prime_factorize();
    let mut cnt = HashMap::default();
    for fi in f {
        *cnt.entry(fi).or_insert(0) += 1;
    }
    let mut ans = 1;
    // c回k(kは素数)が現れる必要がある c<=40

    for (k, c) in cnt {
        let mut cnt = 0;
        for i in 1.. {
            let mut l = k * i;
            while l % k == 0 {
                l /= k;
                cnt += 1;
            }
            if cnt >= c {
                chmax!(ans, k * i);
                break;
            }
        }
    }

    io.out(ans.ln());
}

#[test]
fn test() {
    let test_suits = vec![
        "30
    ",
        "123456789011
    ",
        "280
    ",
        "256",
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
