# [rustfmt :: skip ] pub struct Writer < W : Write > {writer : BufWriter < W > , }
# [rustfmt :: skip ] impl < W : Write > Writer < W > {pub fn new (write : W ) -> Self {Self {writer : BufWriter :: new (write ) , } } pub fn ln < S : Display > (& mut self , s : S ) {writeln ! (self . writer , "{}" , s ) . expect ("Failed to write." ) } pub fn out < S : Display > (& mut self , s : S ) {write ! (self . writer , "{}" , s ) . expect ("Failed to write." ) } pub fn join < S : Display > (& mut self , v : & [S ] , separator : & str ) {v . iter () . fold ("" , | sep , arg | {write ! (self . writer , "{}{}" , sep , arg ) . expect ("Failed to write." ) ; separator } ) ; writeln ! (self . writer ) . expect ("Failed to write." ) ; } pub fn bits (& mut self , i : i64 , len : usize ) {(0 .. len ) . for_each (| b | write ! (self . writer , "{}" , i >> b & 1 ) . expect ("Failed to write." ) ) ; writeln ! (self . writer ) . expect ("Failed to write." ) } pub fn flush (& mut self ) {let _ = self . writer . flush () ; } }
# [rustfmt :: skip ] pub struct Reader < F > {init : F , buf : VecDeque < String > , }
# [rustfmt :: skip ] mod reader_impl {use super :: {BufRead , Reader , VecDeque , FromStr as FS } ; impl < R : BufRead , F : FnMut () -> R > Iterator for Reader < F > {type Item = String ; fn next (& mut self ) -> Option < String > {if self . buf . is_empty () {let mut reader = (self . init ) () ; let mut l = String :: new () ; reader . read_line (& mut l ) . unwrap () ; self . buf . append (& mut l . split_whitespace () . map (ToString :: to_string ) . collect () ) ; } self . buf . pop_front () } } impl < R : BufRead , F : FnMut () -> R > Reader < F > {pub fn new (init : F ) -> Self {let buf = VecDeque :: new () ; Reader {init , buf } } pub fn v < T : FS > (& mut self ) -> T {let s = self . next () . expect ("Insufficient input." ) ; s . parse () . ok () . expect ("Failed to parse." ) } pub fn v2 < T1 : FS , T2 : FS > (& mut self ) -> (T1 , T2 ) {(self . v () , self . v () ) } pub fn v3 < T1 : FS , T2 : FS , T3 : FS > (& mut self ) -> (T1 , T2 , T3 ) {(self . v () , self . v () , self . v () ) } pub fn v4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 ) {(self . v () , self . v () , self . v () , self . v () ) } pub fn v5 < T1 : FS , T2 : FS , T3 : FS , T4 : FS , T5 : FS > (& mut self ) -> (T1 , T2 , T3 , T4 , T5 ) {(self . v () , self . v () , self . v () , self . v () , self . v () ) } pub fn vec < T : FS > (& mut self , length : usize ) -> Vec < T > {(0 .. length ) . map (| _ | self . v () ) . collect () } pub fn vec2 < T1 : FS , T2 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 ) > {(0 .. length ) . map (| _ | self . v2 () ) . collect () } pub fn vec3 < T1 : FS , T2 : FS , T3 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 ) > {(0 .. length ) . map (| _ | self . v3 () ) . collect () } pub fn vec4 < T1 : FS , T2 : FS , T3 : FS , T4 : FS > (& mut self , length : usize ) -> Vec < (T1 , T2 , T3 , T4 ) > {(0 .. length ) . map (| _ | self . v4 () ) . collect () } pub fn chars (& mut self ) -> Vec < char > {self . v :: < String > () . chars () . collect () } fn split (& mut self , zero : u8 ) -> Vec < usize > {self . v :: < String > () . chars () . map (| c | (c as u8 - zero ) as usize ) . collect () } pub fn digits (& mut self ) -> Vec < usize > {self . split (b'0' ) } pub fn lowercase (& mut self ) -> Vec < usize > {self . split (b'a' ) } pub fn uppercase (& mut self ) -> Vec < usize > {self . split (b'A' ) } pub fn char_map (& mut self , h : usize ) -> Vec < Vec < char > > {(0 .. h ) . map (| _ | self . chars () ) . collect () } pub fn bool_map (& mut self , h : usize , ng : char ) -> Vec < Vec < bool > > {self . char_map (h ) . iter () . map (| v | v . iter () . map (| & c | c != ng ) . collect () ) . collect () } pub fn matrix < T : FS > (& mut self , h : usize , w : usize ) -> Vec < Vec < T > > {(0 .. h ) . map (| _ | self . vec (w ) ) . collect () } } }
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
# [rustfmt :: skip ] pub fn main () {let stdin = stdin () ; let stdout = stdout () ; std :: thread :: Builder :: new () . name ("extend stack size" . into () ) . stack_size (32 * 1024 * 1024 ) . spawn (move | | solve (Reader :: new (| | stdin . lock () ) , Writer :: new (stdout . lock () ) ) ) . unwrap () . join () . unwrap () }
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
    #[inline]
    pub fn generate(a: u32, r2: u32, m: u32, m_inv: u32) -> u32 {
        mrmul(a, r2, m, m_inv)
    }
    #[inline]
    pub fn mrmul(ar: u32, br: u32, m: u32, m_inv: u32) -> u32 {
        let t: u64 = (ar as u64) * (br as u64);
        let (t, f) = ((t >> 32) as u32)
            .overflowing_sub(((((t as u32).wrapping_mul(m_inv) as u128) * m as u128) >> 32) as u32);
        if f {
            t.wrapping_add(m)
        } else {
            t
        }
    }
    #[inline]
    pub fn reduce(ar: u32, m: u32, m_inv: u32) -> u32 {
        let (t, f) =
            (((((ar.wrapping_mul(m_inv)) as u128) * (m as u128)) >> 32) as u32).overflowing_neg();
        if f {
            t.wrapping_add(m)
        } else {
            t
        }
    }
    impl<M: Mod> ModInt<M> {
        #[inline]
        pub fn new(mut n: u32) -> Self {
            if n >= M::MOD {
                n = n.rem_euclid(M::MOD);
            }
            Self(generate(n, M::R_POW2, M::MOD, M::MOD_INV), PhantomData)
        }
        pub fn comb(n: i64, mut r: i64) -> Self {
            if r > n - r {
                r = n - r;
            }
            if r == 0 {
                return Self::one();
            }
            let (mut ret, mut rev) = (Self::one(), Self::one());
            for k in 0..r {
                ret *= n - k;
                rev *= r - k;
            }
            ret / rev
        }
        #[inline]
        pub fn get(self) -> u32 {
            reduce(self.0, M::MOD, M::MOD_INV)
        }
    }
    impl<M: Mod> Pow for ModInt<M> {
        #[inline]
        fn pow(mut self, mut e: i64) -> Self {
            debug_assert!(e > 0);
            let mut t = if e & 1 == 0 { M::R } else { self.0 };
            e >>= 1;
            while e != 0 {
                self.0 = mrmul(self.0, self.0, M::MOD, M::MOD_INV);
                if e & 1 != 0 {
                    t = mrmul(t, self.0, M::MOD, M::MOD_INV);
                }
                e >>= 1;
            }
            self.0 = t;
            self
        }
    }
    impl<M: Mod> Add<i64> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn add(self, rhs: i64) -> Self {
            self + ModInt::from(rhs)
        }
    }
    impl<M: Mod> Add<ModInt<M>> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn add(mut self, rhs: Self) -> Self {
            self += rhs;
            self
        }
    }
    impl<M: Mod> AddAssign<i64> for ModInt<M> {
        #[inline]
        fn add_assign(&mut self, rhs: i64) {
            *self = *self + rhs
        }
    }
    impl<M: Mod> AddAssign<ModInt<M>> for ModInt<M> {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            self.0 = self.0 + rhs.0;
            if self.0 >= M::MOD {
                self.0 -= M::MOD
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
    impl<M: Mod> Sub<i64> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: i64) -> Self {
            self - ModInt::from(rhs)
        }
    }
    impl<M: Mod> Sub<ModInt<M>> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn sub(mut self, rhs: Self) -> Self {
            self -= rhs;
            self
        }
    }
    impl<M: Mod> SubAssign<i64> for ModInt<M> {
        #[inline]
        fn sub_assign(&mut self, rhs: i64) {
            *self -= Self::from(rhs)
        }
    }
    impl<M: Mod> SubAssign<ModInt<M>> for ModInt<M> {
        #[inline]
        fn sub_assign(&mut self, rhs: Self) {
            self.0 = if self.0 >= rhs.0 {
                self.0 - rhs.0
            } else {
                self.0 + M::MOD - rhs.0
            }
        }
    }
    impl<M: Mod> Mul<i64> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn mul(mut self, rhs: i64) -> Self {
            self *= rhs;
            self
        }
    }
    impl<M: Mod> Mul<ModInt<M>> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn mul(mut self, rhs: Self) -> Self {
            self *= rhs;
            self
        }
    }
    impl<M: Mod> MulAssign<i64> for ModInt<M> {
        #[inline]
        fn mul_assign(&mut self, rhs: i64) {
            *self *= Self::from(rhs);
        }
    }
    impl<M: Mod> MulAssign<ModInt<M>> for ModInt<M> {
        #[inline]
        fn mul_assign(&mut self, rhs: Self) {
            self.0 = mrmul(self.0, rhs.0, M::MOD, M::MOD_INV)
        }
    }
    impl<M: Mod> Div<i64> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn div(mut self, rhs: i64) -> Self {
            self /= rhs;
            self
        }
    }
    impl<M: Mod> Div<ModInt<M>> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn div(mut self, rhs: Self) -> Self {
            self /= rhs;
            self
        }
    }
    impl<M: Mod> DivAssign<i64> for ModInt<M> {
        #[inline]
        fn div_assign(&mut self, rhs: i64) {
            *self /= Self::from(rhs)
        }
    }
    impl<M: Mod> DivAssign<ModInt<M>> for ModInt<M> {
        #[inline]
        fn div_assign(&mut self, rhs: Self) {
            *self *= rhs.pow((M::MOD - 2) as i64)
        }
    }
    impl<M: Mod> Display for ModInt<M> {
        #[inline]
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "{}", self.get())
        }
    }
    impl<M: Mod> Debug for ModInt<M> {
        #[inline]
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "{}", self.get())
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
    impl<M: Mod> From<i64> for ModInt<M> {
        #[inline]
        fn from(i: i64) -> Self {
            Self::new(i.rem_euclid(M::MOD as i64) as u32)
        }
    }
    impl<M: Mod> From<ModInt<M>> for i64 {
        #[inline]
        fn from(m: ModInt<M>) -> Self {
            m.get() as i64
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
pub use mod_998_244_353_impl::{mi, Mi, Mod998_244_353};
pub mod mod_998_244_353_impl {
    use super::{Mod, ModInt, Pow, PrimitiveRoot, Zero};
    pub fn mi(i: u32) -> Mi {
        Mi::new(i)
    }
    pub type Mi = ModInt<Mod998_244_353>;
    #[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
    pub struct Mod998_244_353;
    impl Mod for Mod998_244_353 {
        const MOD: u32 = 998_244_353;
        const MOD_INV: u32 = 3_296_722_945;
        const R: u32 = 301_989_884;
        const R_POW2: u32 = 932_051_910;
    }
    impl PrimitiveRoot for Mi {
        const DIVIDE_LIMIT: usize = 23;
        #[inline]
        fn primitive_root() -> Self {
            let exp = (Mi::zero() - 1) / Self::new(2).pow(23);
            Mi::pow(Self::new(3), exp.into())
        }
    }
}
pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let n: usize = reader.v();
    for _ in 0..n {
        let n = reader.v::<usize>();
        let a = reader.vec::<usize>(n);
        let mut b = a.clone();
        b.sort();
        let mut v = 0;
        for i in 0..n {
            if a[i] != b[i] && a[i] == 0 {
                v += 1;
            }
        }
        let p = mi(n as u32) * mi(n as u32 - 1) / 2;
        let mut ans = Mi::zero();
        for i in 1..=v {
            ans += p / mi(i * i);
        }
        writer.ln(ans);
    }
}
