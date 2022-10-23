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
# [rustfmt :: skip ] mod algebra_traits {use super :: Debug ; pub trait Magma {type M : Clone + PartialEq + Debug ; fn op (x : & Self :: M , y : & Self :: M ) -> Self :: M ; } pub trait Associative {} pub trait Unital : Magma {fn unit () -> Self :: M ; } pub trait Commutative : Magma {} pub trait Invertible : Magma {fn inv (x : & Self :: M ) -> Self :: M ; } pub trait Idempotent : Magma {} pub trait SemiGroup : Magma + Associative {} pub trait Monoid : Magma + Associative + Unital {fn pow (& self , x : Self :: M , mut n : usize ) -> Self :: M {let mut res = Self :: unit () ; let mut base = x ; while n > 0 {if n & 1 == 1 {res = Self :: op (& res , & base ) ; } base = Self :: op (& base , & base ) ; n >>= 1 ; } res } } pub trait CommutativeMonoid : Magma + Associative + Unital + Commutative {} pub trait Group : Magma + Associative + Unital + Invertible {} pub trait AbelianGroup : Magma + Associative + Unital + Commutative + Invertible {} pub trait Band : Magma + Associative + Idempotent {} impl < M : Magma + Associative > SemiGroup for M {} impl < M : Magma + Associative + Unital > Monoid for M {} impl < M : Magma + Associative + Unital + Commutative > CommutativeMonoid for M {} impl < M : Magma + Associative + Unital + Invertible > Group for M {} impl < M : Magma + Associative + Unital + Commutative + Invertible > AbelianGroup for M {} impl < M : Magma + Associative + Idempotent > Band for M {} pub trait MapMonoid {type Mono : Monoid ; type Func : Monoid ; fn op (& self , x : & < Self :: Mono as Magma > :: M , y : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M {Self :: Mono :: op (x , y ) } fn unit () -> < Self :: Mono as Magma > :: M {Self :: Mono :: unit () } fn apply (& self , f : & < Self :: Func as Magma > :: M , value : & < Self :: Mono as Magma > :: M , ) -> < Self :: Mono as Magma > :: M ; fn identity_map () -> < Self :: Func as Magma > :: M {Self :: Func :: unit () } fn compose (& self , f : & < Self :: Func as Magma > :: M , g : & < Self :: Func as Magma > :: M , ) -> < Self :: Func as Magma > :: M {Self :: Func :: op (f , g ) } } pub trait Zero {fn zero () -> Self ; } pub trait One {fn one () -> Self ; } pub trait BoundedBelow {fn min_value () -> Self ; } pub trait BoundedAbove {fn max_value () -> Self ; } pub trait Pow {fn pow (self , exp : i64 ) -> Self ; } pub trait PrimitiveRoot {fn primitive_root () -> Self ; } }
# [rustfmt :: skip ] pub trait Integral : 'static + Send + Sync + Copy + Ord + Display + Debug + Add < Output = Self > + Sub < Output = Self > + Mul < Output = Self > + Div < Output = Self > + Rem < Output = Self > + AddAssign + SubAssign + MulAssign + DivAssign + RemAssign + Sum + Product + BitOr < Output = Self > + BitAnd < Output = Self > + BitXor < Output = Self > + Not < Output = Self > + Shl < Output = Self > + Shr < Output = Self > + BitOrAssign + BitAndAssign + BitXorAssign + ShlAssign + ShrAssign + Zero + One + BoundedBelow + BoundedAbove {}
macro_rules ! impl_integral {($ ($ ty : ty ) ,* ) => {$ (impl Zero for $ ty {fn zero () -> Self {0 } } impl One for $ ty {fn one () -> Self {1 } } impl BoundedBelow for $ ty {fn min_value () -> Self {Self :: min_value () } } impl BoundedAbove for $ ty {fn max_value () -> Self {Self :: max_value () } } impl Integral for $ ty {} ) * } ; }
impl_integral!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
# [rustfmt :: skip ] pub fn main () {let stdin = stdin () ; let stdout = stdout () ; std :: thread :: Builder :: new () . name ("extend stack size" . into () ) . stack_size (32 * 1024 * 1024 ) . spawn (move | | solve (Reader :: new (| | stdin . lock () ) , Writer :: new (stdout . lock () ) ) ) . unwrap () . join () . unwrap () }

pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let n: i64 = reader.v();
    let m: u64 = reader.v();
    let p = m.prime_factorize();
    let mut ans = mi(1);
    let mut cur = 0;
    let mut cnt = 0;
    dbg!(&p, cnt);
    for pi in p {
        if pi == cur {
            cnt += 1;
        } else {
            ans *= ncr(cnt + n - 1, cnt);
            cnt = 1;
            cur = pi;
            dbg!(&ans);
        }
    }
    if cnt > 0 {
        ans *= ncr(cnt + n - 1, cnt);
    }
    writer.ln(ans);
}

fn ncr(n: i64, r: i64) -> Mi {
    let mut ret = mi(1);
    for i in 1..=r {
        ret *= n - r + i;
        ret /= i;
    }
    ret
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

pub trait Mod: Copy + Clone + Debug {
    fn get() -> mod_int_impl::InnerType;
    fn mont() -> MontgomeryReduction;
}
#[derive(Copy, Clone, Eq, PartialEq, Default)]
pub struct ModInt<M: Mod>(mod_int_impl::InnerType, PhantomData<fn() -> M>);
mod mod_int_impl {
    use super::{
        Add, AddAssign, Debug, Deref, DerefMut, Display, Div, DivAssign, Formatter, Mod, ModInt,
        Mul, MulAssign, Neg, One, PhantomData, Pow, Sub, SubAssign, Sum, Zero,
    };
    pub type InnerType = i64;
    impl<M: Mod> ModInt<M> {
        pub fn new(mut n: InnerType) -> Self {
            if n < 0 || n >= M::get() {
                n = n.rem_euclid(M::get());
            }
            Self(n, PhantomData)
        }
        pub fn comb(n: i64, mut r: i64) -> Self {
            if r > n - r {
                r = n - r;
            }
            if r == 0 {
                return Self::new(1);
            }
            let (mut ret, mut rev) = (Self::new(1), Self::new(1));
            for k in 0..r {
                ret *= n - k;
                rev *= r - k;
            }
            ret / rev
        }
        pub fn get(self) -> InnerType {
            self.0
        }
    }
    impl<M: Mod> Pow for ModInt<M> {
        fn pow(self, mut e: i64) -> Self {
            let m = e < 0;
            e = e.abs();
            let t = M::mont().reduce(M::mont().pow(self.0 as u64, e as u64));
            if m {
                Self::new(1) / t as i64
            } else {
                Self::new(t as i64)
            }
        }
    }
    impl<M: Mod> Add<i64> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn add(self, rhs: i64) -> Self {
            self + ModInt::new(rhs)
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
            self.0 = if self.0 + rhs.0 >= M::get() {
                self.0 + rhs.0 - M::get()
            } else {
                self.0 + rhs.0
            }
        }
    }
    impl<M: Mod> Neg for ModInt<M> {
        type Output = Self;
        #[inline]
        fn neg(self) -> Self {
            Self::new(-self.0)
        }
    }
    impl<M: Mod> Sub<i64> for ModInt<M> {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: i64) -> Self {
            self - ModInt::new(rhs)
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
            *self -= Self::new(rhs)
        }
    }
    impl<M: Mod> SubAssign<ModInt<M>> for ModInt<M> {
        #[inline]
        fn sub_assign(&mut self, rhs: Self) {
            self.0 = if self.0 >= rhs.0 {
                self.0 - rhs.0
            } else {
                self.0 + M::get() - rhs.0
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
        fn mul(self, rhs: Self) -> Self {
            self * rhs.0
        }
    }
    impl<M: Mod> MulAssign<i64> for ModInt<M> {
        #[inline]
        fn mul_assign(&mut self, rhs: i64) {
            *self *= Self::new(rhs);
        }
    }
    impl<M: Mod> MulAssign<ModInt<M>> for ModInt<M> {
        #[inline]
        fn mul_assign(&mut self, rhs: Self) {
            self.0 = M::mont().mul_prim(self.0 as u64, rhs.0 as u64) as i64
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
            *self /= Self::new(rhs)
        }
    }
    impl<M: Mod> DivAssign<ModInt<M>> for ModInt<M> {
        #[inline]
        fn div_assign(&mut self, rhs: Self) {
            *self *= rhs.pow(M::get() - 2)
        }
    }
    impl<M: Mod> Display for ModInt<M> {
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl<M: Mod> Debug for ModInt<M> {
        fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl<M: Mod> Deref for ModInt<M> {
        type Target = i64;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<M: Mod> DerefMut for ModInt<M> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }
    impl<M: Mod> Sum for ModInt<M> {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::new(0), |x, a| x + a)
        }
    }
    impl<M: Mod> From<i64> for ModInt<M> {
        fn from(i: i64) -> Self {
            Self::new(i)
        }
    }
    impl<M: Mod> From<ModInt<M>> for i64 {
        fn from(m: ModInt<M>) -> Self {
            m.0
        }
    }
    impl<M: Mod> Zero for ModInt<M> {
        fn zero() -> Self {
            Self::new(0)
        }
    }
    impl<M: Mod> One for ModInt<M> {
        fn one() -> Self {
            Self::new(1)
        }
    }
}

pub use mod_1_000_000_007_impl::{mi, Mi};
mod mod_1_000_000_007_impl {
    use super::{Mod, ModInt, MontgomeryReduction};
    pub fn mi(i: i64) -> Mi {
        Mi::new(i)
    }
    pub type Mi = ModInt<Mod1_000_000_007>;
    const MOD: i64 = 1_000_000_007;
    const MONTGOMERY: MontgomeryReduction = MontgomeryReduction::new(MOD as u64);
    #[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
    pub struct Mod1_000_000_007;
    impl Mod for Mod1_000_000_007 {
        fn get() -> i64 {
            MOD
        }
        fn mont() -> MontgomeryReduction {
            MONTGOMERY
        }
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
        let p = find_cycle_by_brent(*self);
        if &p == self {
            return vec![p];
        }
        let mut ret = p.prime_factorize();
        ret.append(&mut (*self / p).prime_factorize());
        ret.sort_unstable();
        ret
    }
}

#[derive(Clone, Debug, Default)]
pub struct Gcd<S>(PhantomData<fn() -> S>);
mod gcd_impl {
    use super::{
        swap, Associative, Commutative, Debug, Gcd, Idempotent, Magma, RemAssign, Unital, Zero,
    };
    impl<S: Clone + Debug + RemAssign + PartialOrd + Zero> Magma for Gcd<S> {
        type M = S;
        fn op(x: &S, y: &S) -> S {
            let (mut x, mut y) = (x.clone(), y.clone());
            if y > x {
                swap(&mut x, &mut y);
            }
            while y != S::zero() {
                x %= y.clone();
                swap(&mut x, &mut y);
            }
            x
        }
    }
    impl<S: Clone + Debug + RemAssign + PartialOrd + Zero> Associative for Gcd<S> {}
    impl<S: Clone + Debug + RemAssign + PartialOrd + Zero> Unital for Gcd<S> {
        fn unit() -> S {
            S::zero()
        }
    }
    impl<S: Clone + Debug + RemAssign + PartialOrd + Zero> Commutative for Gcd<S> {}
    impl<S: Clone + Debug + RemAssign + PartialOrd + Zero> Idempotent for Gcd<S> {}
}
