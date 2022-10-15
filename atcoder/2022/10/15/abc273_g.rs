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
pub trait Mod: Copy + Clone + Debug {
    fn get() -> i64;
}
#[derive(Copy, Clone, Eq, PartialEq, Default)]
pub struct ModInt<M: Mod>(i64, PhantomData<fn() -> M>);
mod mod_int_impl {
    use super::{
        Add, AddAssign, Debug, Deref, DerefMut, Display, Div, DivAssign, Formatter, Mod, ModInt,
        Mul, MulAssign, Neg, One, PhantomData, Pow, Sub, SubAssign, Sum, Zero,
    };
    impl<M: Mod> ModInt<M> {
        pub fn new(mut n: i64) -> Self {
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
        pub fn get(self) -> i64 {
            self.0
        }
    }
    impl<M: Mod> Pow for ModInt<M> {
        fn pow(mut self, mut e: i64) -> Self {
            let m = e < 0;
            e = e.abs();
            let mut result = Self::new(1);
            while e > 0 {
                if e & 1 == 1 {
                    result *= self.0;
                }
                e >>= 1;
                self *= self.0;
            }
            if m {
                Self::new(1) / result
            } else {
                result
            }
        }
    }
    impl<M: Mod> Add<i64> for ModInt<M> {
        type Output = Self;
        fn add(self, rhs: i64) -> Self {
            self + ModInt::new(rhs)
        }
    }
    impl<M: Mod> Add<ModInt<M>> for ModInt<M> {
        type Output = Self;
        fn add(mut self, rhs: Self) -> Self {
            self += rhs;
            self
        }
    }
    impl<M: Mod> AddAssign<i64> for ModInt<M> {
        fn add_assign(&mut self, rhs: i64) {
            *self = *self + rhs
        }
    }
    impl<M: Mod> AddAssign<ModInt<M>> for ModInt<M> {
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
        fn neg(self) -> Self {
            Self::new(-self.0)
        }
    }
    impl<M: Mod> Sub<i64> for ModInt<M> {
        type Output = Self;
        fn sub(self, rhs: i64) -> Self {
            self - ModInt::new(rhs)
        }
    }
    impl<M: Mod> Sub<ModInt<M>> for ModInt<M> {
        type Output = Self;
        fn sub(mut self, rhs: Self) -> Self {
            self -= rhs;
            self
        }
    }
    impl<M: Mod> SubAssign<i64> for ModInt<M> {
        fn sub_assign(&mut self, rhs: i64) {
            *self = *self - rhs
        }
    }
    impl<M: Mod> SubAssign<ModInt<M>> for ModInt<M> {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 = if self.0 >= rhs.0 {
                self.0 - rhs.0
            } else {
                self.0 - rhs.0 + M::get()
            }
        }
    }
    impl<M: Mod> Mul<i64> for ModInt<M> {
        type Output = Self;
        fn mul(self, rhs: i64) -> Self {
            ModInt::new(self.0 * (rhs % M::get()))
        }
    }
    impl<M: Mod> Mul<ModInt<M>> for ModInt<M> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            self * rhs.0
        }
    }
    impl<M: Mod> MulAssign<i64> for ModInt<M> {
        fn mul_assign(&mut self, rhs: i64) {
            *self = *self * rhs
        }
    }
    impl<M: Mod> MulAssign<ModInt<M>> for ModInt<M> {
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs
        }
    }
    impl<M: Mod> Div<i64> for ModInt<M> {
        type Output = Self;
        fn div(self, rhs: i64) -> Self {
            self * ModInt::new(rhs).pow(M::get() - 2)
        }
    }
    impl<M: Mod> Div<ModInt<M>> for ModInt<M> {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            self / rhs.0
        }
    }
    impl<M: Mod> DivAssign<i64> for ModInt<M> {
        fn div_assign(&mut self, rhs: i64) {
            *self = *self / rhs
        }
    }
    impl<M: Mod> DivAssign<ModInt<M>> for ModInt<M> {
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs
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

pub fn mi(i: i64) -> Mi {
    Mi::new(i)
}
pub type Mi = ModInt<Mod998244353>;
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Mod998244353;
impl Mod for Mod998244353 {
    fn get() -> i64 {
        998_244_353
    }
}
impl PrimitiveRoot for ModInt<Mod998244353> {
    fn primitive_root() -> Self {
        let exp = ModInt::new(Mod998244353::get() - 1) / Self::new(2).pow(23);
        Self::new(3).pow(exp.get())
    }
}
#[derive(Debug)]
pub struct ModValTable<M> {
    fact: Vec<M>,
    fact_inv: Vec<M>,
}
impl<M: Mod> ModValTable<ModInt<M>> {
    pub fn new(n: usize) -> Self {
        let mut fact = vec![ModInt::<M>::new(1); n + 1];
        let mut fact_inv = vec![ModInt::<M>::new(1); n + 1];
        let mut inv = vec![ModInt::<M>::new(1); n + 1];
        for i in 2..=n {
            fact[i] = fact[i - 1] * i as i64;
            inv[i] = inv[0] / i as i64;
            fact_inv[i] = fact_inv[i - 1] * inv[i];
        }
        Self { fact, fact_inv }
    }
    pub fn factorial(&self, n: i64) -> ModInt<M> {
        self.fact[n as usize]
    }
    pub fn permutation(&self, n: i64, r: i64) -> ModInt<M> {
        if n < r {
            0.into()
        } else {
            self.fact[n as usize] * self.fact_inv[(n - r) as usize]
        }
    }
    pub fn combination(&self, n: i64, r: i64) -> ModInt<M> {
        if n < r {
            0.into()
        } else {
            self.permutation(n, r) * self.fact_inv[r as usize]
        }
    }
    pub fn combinations_with_replacement(&self, n: i64, r: i64) -> ModInt<M> {
        if n < r {
            0.into()
        } else {
            self.fact[(n + r) as usize] * self.fact_inv[r as usize] * self.fact_inv[n as usize - 1]
        }
    }
    pub fn catalan_number(&self, n: i64) -> ModInt<M> {
        if n < 0 {
            0.into()
        } else {
            assert!(n as usize * 2 < self.fact.len());
            self.fact[n as usize * 2] * self.fact_inv[n as usize + 1] * self.fact_inv[n as usize]
        }
    }
    pub fn montmort_number(&self, n: usize) -> ModInt<M> {
        if n < 1 {
            0.into()
        } else {
            let mut ret = ModInt::<M>::new(0);
            for k in 0..=n {
                ret += if k & 1 == 0 {
                    self.fact_inv[k]
                } else {
                    -self.fact_inv[k]
                };
            }
            ret * self.fact[n]
        }
    }
}

pub fn solve<R: BufRead, W: Write, F: FnMut() -> R>(mut reader: Reader<F>, mut writer: Writer<W>) {
    let n: usize = reader.v();
    let mut r = reader.vec::<i64>(n);
    let c = reader.vec::<i64>(n);
    // let n = 1500;
    // let mut r = vec![2; 1500];
    // let c = vec![2; 1500];
    if r.iter().sum::<i64>() != c.iter().sum() {
        return writer.ln(0);
    }
    r.sort();
    r.reverse();
    let mut two = 0;
    let mut one = 0;
    for ci in c {
        if ci == 2 {
            two += 1;
        } else if ci == 1 {
            one += 1;
        }
    }
    let mvt = ModValTable::new(n);
    let mut dp: HashMap<(usize, i64, i64), Mi> = HashMap::default();
    writer.ln(f(0, &r, two, one, &mut dp, &mvt));
    // dbg!(&dp);
}

fn f(
    index: usize,
    r: &Vec<i64>,
    two: i64,
    one: i64,
    dp: &mut HashMap<(usize, i64, i64), Mi>,
    mvt: &ModValTable<Mi>,
) -> Mi {
    if let Some(r) = dp.get(&(index, two, one)) {
        return *r;
    }
    if index == r.len() {
        return mi(1);
    }
    let ret = match r[index] {
        0 => f(index + 1, r, two, one, dp, mvt),
        1 => {
            let mut k = mi(0);
            if two > 0 {
                k += f(index + 1, r, two - 1, one + 1, dp, mvt) * two;
            }
            if one > 0 {
                k += f(index + 1, r, two, one - 1, dp, mvt) * one;
            }
            k
        }
        2 => {
            let mut k = mi(0);
            // 2のところ2か所に1個ずつ置く
            if two > 1 {
                k += f(index + 1, r, two - 2, one + 2, dp, mvt) * mvt.combination(two, 2);
            }
            // 1のところ2箇所に1個ずつ置く
            if one > 1 {
                k += f(index + 1, r, two, one - 2, dp, mvt) * mvt.combination(one, 2);
            }
            // 2のところに2個置く
            if two > 0 {
                k += f(index + 1, r, two - 1, one, dp, mvt) * two;
            }
            // 1のところと2のところに1個ずつ置く
            if two > 0 && one > 0 {
                k += f(index + 1, r, two - 1, one, dp, mvt) * two * one;
            }
            k
        }
        _ => panic!(),
    };
    dp.insert((index, two, one), ret);
    ret
}
