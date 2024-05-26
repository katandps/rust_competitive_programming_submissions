// codesnip-guard: template
#[cfg_attr(any(),rustfmt::skip)]pub fn main(){std::thread::Builder::new().name("extend stack size".into()).stack_size(128*1024*1024).spawn(move||{let(stdin,stdout)=(stdin(),stdout());let(stdin_lock,stdout_lock)=(stdin.lock(),stdout.lock());solve(stdin_lock,stdout_lock);}).unwrap().join().unwrap()}
// codesnip-guard: algebra
#[cfg_attr(any(),rustfmt::skip)]pub use algebra_traits::{AbelianGroup,Associative,Band,BoundedAbove,BoundedBelow,Commutative,CommutativeMonoid,Group,Idempotent,Integral,Invertible,LeastSignificantBit,Magma,MapMonoid,Mapping,Monoid,MonoidMapping,One,Pow,PrimitiveRoot,SemiGroup,TrailingZeros,Unital,Zero};
#[cfg_attr(any(),rustfmt::skip)]mod algebra_traits{use super::{Add,AddAssign,BitAnd,BitAndAssign,BitOr,BitOrAssign,BitXor,BitXorAssign,Debug,Display,Div,DivAssign,Mul,MulAssign,Not,Product,Rem,RemAssign,Shl,ShlAssign,Shr,ShrAssign,Sub,SubAssign,Sum};#[doc=" # マグマ"]#[doc=" 二項演算: $M \\circ M \\to M$"]pub trait Magma{#[doc=" マグマを構成する集合$M$"]type M:Clone+PartialEq+Debug;#[doc=" マグマを構成する演算$op$"]fn op(&mut self,x:&Self::M,y:&Self::M)->Self::M;fn op_rev(&mut self,x:&Self::M,y:&Self::M)->Self::M{self.op(y,x)}}#[doc=" # 結合則"]#[doc=" $\\forall a,\\forall b, \\forall c \\in T, (a \\circ b) \\circ c = a \\circ (b \\circ c)$"]pub trait Associative:Magma{}#[doc=" # 単位的"]pub trait Unital:Magma{#[doc=" 単位元 identity element: $e$"]fn unit()->Self::M;}#[doc=" # 可換"]pub trait Commutative:Magma{}#[doc=" # 可逆的"]#[doc=" $\\exists e \\in T, \\forall a \\in T, \\exists b,c \\in T, b \\circ a = a \\circ c = e$"]pub trait Invertible:Magma{#[doc=" $a$ where $a \\circ x = e$"]fn inv(x:&Self::M)->Self::M;}#[doc=" # 冪等性"]pub trait Idempotent:Magma{}#[doc=" # 半群"]#[doc=" 1. 結合則"]pub trait SemiGroup:Magma+Associative{}#[doc=" # モノイド"]#[doc=" 1. 結合則"]#[doc=" 1. 単位元"]pub trait Monoid:Magma+Associative+Unital{#[doc=" $x^n = x\\circ\\cdots\\circ x$"]fn pow(&mut self,x:Self::M,mut n:usize)->Self::M{let mut res=Self::unit();let mut base=x;while n>0{if n&1==1{res=self.op(&res,&base);}base=self.op(&base,&base);n>>=1;}res}}#[doc=" # 可換モノイド"]pub trait CommutativeMonoid:Magma+Associative+Unital+Commutative{}#[doc=" # 群"]#[doc=" 1. 結合法則"]#[doc=" 1. 単位元"]#[doc=" 1. 逆元"]pub trait Group:Magma+Associative+Unital+Invertible{}#[doc=" # アーベル群"]pub trait AbelianGroup:Magma+Associative+Unital+Commutative+Invertible{}#[doc=" # Band"]#[doc=" 1. 結合法則"]#[doc=" 1. 冪等律"]pub trait Band:Magma+Associative+Idempotent{}impl<M:Magma+Associative>SemiGroup for M{}impl<M:Magma+Associative+Unital>Monoid for M{}impl<M:Magma+Associative+Unital+Commutative>CommutativeMonoid for M{}impl<M:Magma+Associative+Unital+Invertible>Group for M{}impl<M:Magma+Associative+Unital+Commutative+Invertible>AbelianGroup for M{}impl<M:Magma+Associative+Idempotent>Band for M{}#[doc=" # 写像"]pub trait Mapping{#[doc=" # 写像を表現する値"]type Mapping:Clone+Debug;#[doc=" # 始集合"]type Domain:Clone+Debug;#[doc=" # 終集合"]type Codomain:Clone+Debug;#[doc=" #"]fn apply(&mut self,map:&Self::Mapping,a:&Self::Domain)->Self::Codomain;}#[doc=" # 作用モノイド"]#[doc=" 作用で、その合成がモノイドをなすもの"]pub trait MonoidMapping:Monoid<M=<Self as Mapping>::Mapping>+Mapping{}impl<T:Monoid+Mapping+Magma<M=<T as Mapping>::Mapping>>MonoidMapping for T{}#[doc=" # 作用モノイド付きモノイド"]#[doc=" 作用が同一集合上の変換である必要がある"]pub trait MapMonoid{#[doc=" 値の合成"]type Mono:Monoid;#[doc=" 作用の合成"]type Map:MonoidMapping<Domain=<Self::Mono as Magma>::M,Codomain=<Self::Mono as Magma>::M>;fn monoid(&mut self)->&mut Self::Mono;fn map(&mut self)->&mut Self::Map;#[doc=" 値xと値yを併合する"]fn op(&mut self,x:&<Self::Mono as Magma>::M,y:&<Self::Mono as Magma>::M)-><Self::Mono as Magma>::M{self.monoid().op(x,y)}fn unit()-><Self::Mono as Magma>::M{Self::Mono::unit()}#[doc=" 作用fをvalueに作用させる"]fn apply(&mut self,f:&<Self::Map as Mapping>::Mapping,value:&<Self::Mono as Magma>::M)-><Self::Map as Mapping>::Codomain{self.map().apply(f,value)}#[doc=" 作用fの単位元"]fn identity_map()-><Self::Map as Magma>::M{Self::Map::unit()}#[doc=" composition:"]#[doc=" $h() = f(g())$"]fn compose(&mut self,f:&<Self::Map as Magma>::M,g:&<Self::Map as Magma>::M)-><Self::Map as Magma>::M{self.map().op(f,g)}}#[doc=" # 加算の単位元"]pub trait Zero{fn zero()->Self;}#[doc=" # 乗算の単位元"]pub trait One{fn one()->Self;}#[doc=" # 下に有界"]pub trait BoundedBelow{fn min_value()->Self;}#[doc=" # 上に有界"]pub trait BoundedAbove{fn max_value()->Self;}pub trait Pow{fn pow(self,exp:i64)->Self;}#[doc=" # 原始根の存在"]pub trait PrimitiveRoot{#[doc=" # $2^{DIVIDE_LIMIT}$乗根まで存在する"]const DIVIDE_LIMIT:usize;#[doc=" # 原始根"]fn primitive_root()->Self;}#[doc=" # 二進数表記したとき最後尾につく0の数"]pub trait TrailingZeros{fn trailing_zero(self)->Self;}#[doc=" # 最下位bit"]pub trait LeastSignificantBit{fn lsb(self)->Self;}pub trait Integral:'static+Send+Sync+Copy+Ord+Display+Debug+Add<Output=Self>+Sub<Output=Self>+Mul<Output=Self>+Div<Output=Self>+Rem<Output=Self>+AddAssign+SubAssign+MulAssign+DivAssign+RemAssign+Sum+Product+BitOr<Output=Self>+BitAnd<Output=Self>+BitXor<Output=Self>+Not<Output=Self>+Shl<Output=Self>+Shr<Output=Self>+BitOrAssign+BitAndAssign+BitXorAssign+ShlAssign+ShrAssign+Zero+One+BoundedBelow+BoundedAbove+TrailingZeros+LeastSignificantBit{}macro_rules!impl_integral{($($ty:ty),*)=>{$(impl Zero for$ty{#[inline]fn zero()->Self{0}}impl One for$ty{#[inline]fn one()->Self{1}}impl BoundedBelow for$ty{#[inline]fn min_value()->Self{Self::MIN}}impl BoundedAbove for$ty{#[inline]fn max_value()->Self{Self::MAX}}impl TrailingZeros for$ty{#[inline]fn trailing_zero(self)->Self{self.trailing_zeros()as$ty}}impl LeastSignificantBit for$ty{#[inline]fn lsb(self)->Self{if self==0{0}else{self&!(self-1)}}}impl Integral for$ty{})*};}impl_integral!(i8,i16,i32,i64,i128,isize,u8,u16,u32,u64,u128,usize);}
// codesnip-guard: chclamp
#[cfg_attr(any(),rustfmt::skip)]#[allow(unused_macros)]#[macro_export]macro_rules!chclamp{($base:expr,$lower_bound:expr,$upper_bound:expr)=>{chmin!($base,$upper_bound)||chmax!($base,$lower_bound)};}
// codesnip-guard: chmax
#[cfg_attr(any(),rustfmt::skip)]#[allow(unused_macros)]#[macro_export]macro_rules!chmax{($base:expr,$($cmps:expr),+$(,)*)=>{{let cmp_max=max!($($cmps),+);if$base<cmp_max{$base=cmp_max;true}else{false}}};}
// codesnip-guard: chmin
#[cfg_attr(any(),rustfmt::skip)]#[allow(unused_macros)]#[macro_export]macro_rules!chmin{($base:expr,$($cmps:expr),+$(,)*)=>{{let cmp_min=min!($($cmps),+);if$base>cmp_min{$base=cmp_min;true}else{false}}};}
// codesnip-guard: clamp
#[cfg_attr(any(),rustfmt::skip)]#[allow(unused_macros)]#[macro_export]macro_rules!clamp{($base:expr,$lower_bound:expr,$upper_bound:expr)=>{max!($lower_bound,min!($upper_bound,$base))};}
// codesnip-guard: dbg-macro
#[cfg_attr(any(),rustfmt::skip)]#[allow(unused_macros)]macro_rules!dbg{($($x:tt)*)=>{{#[cfg(debug_assertions)]{std::dbg!($($x)*)}#[cfg(not(debug_assertions))]{($($x)*)}}}}
// codesnip-guard: faster-hashmap
#[cfg_attr(any(),rustfmt::skip)]pub use self::fxhasher_impl::{FxHashMap as HashMap,FxHashSet as HashSet};
#[cfg_attr(any(),rustfmt::skip)]mod fxhasher_impl{use super::{BitXor,BuildHasherDefault,Hasher,TryInto};use std::collections::{HashMap,HashSet};#[derive(Default)]pub struct FxHasher{hash:u64}type BuildHasher=BuildHasherDefault<FxHasher>;pub type FxHashMap<K,V>=HashMap<K,V,BuildHasher>;pub type FxHashSet<V>=HashSet<V,BuildHasher>;const ROTATE:u32=5;const SEED:u64=0x51_7c_c1_b7_27_22_0a_95;impl Hasher for FxHasher{#[inline]fn finish(&self)->u64{self.hash}#[inline]fn write(&mut self,mut bytes:&[u8]){while bytes.len()>=8{self.add_to_hash(u64::from_ne_bytes(bytes[..8].try_into().unwrap()));bytes=&bytes[8..];}while bytes.len()>=4{self.add_to_hash(u64::from(u32::from_ne_bytes(bytes[..4].try_into().unwrap())));bytes=&bytes[4..];}while bytes.len()>=2{self.add_to_hash(u64::from(u16::from_ne_bytes(bytes[..2].try_into().unwrap())));}if let Some(&byte)=bytes.first(){self.add_to_hash(u64::from(byte));}}}impl FxHasher{#[inline]pub fn add_to_hash(&mut self,i:u64){self.hash=self.hash.rotate_left(ROTATE).bitxor(i).wrapping_mul(SEED);}}}
// codesnip-guard: float-value
#[cfg_attr(any(),rustfmt::skip)]pub use float_value_impl::{FValue,EPS};
#[cfg_attr(any(),rustfmt::skip)]mod float_value_impl{use super::FromStr;use super::{Add,Debug,Display,Div,Formatter,Mul,Neg,Ordering,Sub};pub const EPS:f64=0.000_000_001;#[doc=" # 浮動小数点数"]#[doc=" 誤差判定をうまく行うための構造体"]#[derive(Copy,Clone,Default)]pub struct FValue(pub f64);impl FValue{pub fn sqrt(&self)->Self{self.0 .sqrt().into()}pub fn abs(&self)->Self{self.0 .abs().into()}pub const fn eps()->Self{Self(EPS)}}impl PartialEq for FValue{fn eq(&self,other:&Self)->bool{(self.0-other.0).abs()<EPS}}impl Eq for FValue{}impl PartialOrd for FValue{fn partial_cmp(&self,other:&Self)->Option<Ordering>{Some(self.cmp(other))}}impl Ord for FValue{fn cmp(&self,other:&Self)->Ordering{self.0 .partial_cmp(&other.0).expect("something went wrong")}}impl From<i64>for FValue{fn from(value:i64)->Self{FValue(value as f64)}}impl FromStr for FValue{type Err=std::num::ParseFloatError;#[inline]fn from_str(s:&str)->Result<Self,Self::Err>{Ok(Self::from(s.parse::<f64>()?))}}impl From<f64>for FValue{fn from(value:f64)->Self{if value.is_nan(){panic!("Detected NaN.");}FValue(value)}}impl From<FValue>for f64{fn from(value:FValue)->Self{value.0}}impl Display for FValue{fn fmt(&self,f:&mut Formatter<'_>)->std::fmt::Result{write!(f,"{}",self.0)}}impl Debug for FValue{fn fmt(&self,f:&mut Formatter<'_>)->std::fmt::Result{write!(f,"{}",self.0)}}impl Add<FValue>for f64{type Output=FValue;fn add(self,rhs:FValue)->Self::Output{(self+rhs.0).into()}}impl<T:Into<f64>>Add<T>for FValue{type Output=Self;fn add(self,rhs:T)->Self::Output{(self.0+rhs.into()).into()}}impl Sub<FValue>for f64{type Output=FValue;fn sub(self,rhs:FValue)->Self::Output{(self-rhs.0).into()}}impl<T:Into<f64>>Sub<T>for FValue{type Output=Self;fn sub(self,rhs:T)->Self::Output{(self.0-rhs.into()).into()}}impl Mul<FValue>for f64{type Output=FValue;fn mul(self,rhs:FValue)->Self::Output{(self*rhs.0).into()}}impl<T:Into<f64>>Mul<T>for FValue{type Output=Self;fn mul(self,rhs:T)->Self::Output{(self.0*rhs.into()).into()}}impl Div<FValue>for f64{type Output=FValue;fn div(self,rhs:FValue)->Self::Output{(self/rhs.0).into()}}impl<T:Into<f64>>Div<T>for FValue{type Output=Self;fn div(self,rhs:T)->Self::Output{(self.0/rhs.into()).into()}}impl Neg for FValue{type Output=Self;fn neg(self)->Self::Output{(-self.0).into()}}}
// codesnip-guard: io-util
#[cfg_attr(any(),rustfmt::skip)]pub use io_impl::{ReadHelper,ReaderTrait};
#[cfg_attr(any(),rustfmt::skip)]mod io_impl{use std::collections::VecDeque;use std::io::{BufRead,Read};use std::str::FromStr as FS;pub trait ReaderTrait{fn next(&mut self)->Option<String>;fn v<T:FS>(&mut self)->T{let s=self.next().expect("Insufficient input.");s.parse().ok().expect("Failed to parse.")}fn v2<T1:FS,T2:FS>(&mut self)->(T1,T2){(self.v(),self.v())}fn v3<T1:FS,T2:FS,T3:FS>(&mut self)->(T1,T2,T3){(self.v(),self.v(),self.v())}fn v4<T1:FS,T2:FS,T3:FS,T4:FS>(&mut self)->(T1,T2,T3,T4){(self.v(),self.v(),self.v(),self.v())}fn v5<T1:FS,T2:FS,T3:FS,T4:FS,T5:FS>(&mut self)->(T1,T2,T3,T4,T5){(self.v(),self.v(),self.v(),self.v(),self.v())}fn vec<T:FS>(&mut self,length:usize)->Vec<T>{(0..length).map(|_|self.v()).collect()}fn vec2<T1:FS,T2:FS>(&mut self,length:usize)->Vec<(T1,T2)>{(0..length).map(|_|self.v2()).collect()}fn vec3<T1:FS,T2:FS,T3:FS>(&mut self,length:usize)->Vec<(T1,T2,T3)>{(0..length).map(|_|self.v3()).collect()}fn vec4<T1:FS,T2:FS,T3:FS,T4:FS>(&mut self,length:usize)->Vec<(T1,T2,T3,T4)>{(0..length).map(|_|self.v4()).collect()}fn chars(&mut self)->Vec<char>{self.v::<String>().chars().collect()}fn split(&mut self,zero:u8)->Vec<usize>{self.v::<String>().chars().map(|c|(c as u8-zero)as usize).collect()}#[doc=" 英小文字からなる文字列の入力を $'0' = 0$ となる数値の配列で得る"]fn digits(&mut self)->Vec<usize>{self.split(b'0')}#[doc=" 英小文字からなる文字列の入力を $'a' = 1$ となる数値の配列で得る"]fn lowercase(&mut self)->Vec<usize>{self.split(b'a'-1)}#[doc=" 英大文字からなる文字列の入力を $'A' = 1$ となる数値の配列で得る"]fn uppercase(&mut self)->Vec<usize>{self.split(b'A'-1)}#[doc=" 改行された文字列の入力を2次元配列とみなし、charの2次元Vecとして得る"]fn char_map(&mut self,h:usize)->Vec<Vec<char>>{(0..h).map(|_|self.chars()).collect()}#[doc=" charの2次元配列からboolのmapを作る ngで指定した壁のみfalseとなる"]fn bool_map(&mut self,h:usize,ng:char)->Vec<Vec<bool>>{self.char_map(h).iter().map(|v|v.iter().map(|&c|c!=ng).collect()).collect()}#[doc=" 空白区切りで $h*w$ 個の要素を行列として取得する"]fn matrix<T:FS>(&mut self,h:usize,w:usize)->Vec<Vec<T>>{(0..h).map(|_|self.vec(w)).collect()}}pub struct ReadHelper<'a>{read:Box<dyn BufRead+'a>,pub buf:VecDeque<String>}impl<'a>ReadHelper<'a>{pub fn new(read:impl Read+'a)->ReadHelper<'a>{Self{read:Box::new(std::io::BufReader::new(read)),buf:VecDeque::new()}}}impl<'a>ReaderTrait for ReadHelper<'a>{fn next(&mut self)->Option<String>{let mut cnt=0;while self.buf.is_empty()&&cnt<100{let mut s=String::new();if let Ok(_l)=self.read.read_line(&mut s){self.buf.append(&mut s.split_ascii_whitespace().map(ToString::to_string).collect());}cnt+=1;}self.buf.pop_front()}}}
// codesnip-guard: max
#[cfg_attr(any(),rustfmt::skip)]#[allow(unused_macros)]#[macro_export]macro_rules!max{($a:expr$(,)*)=>{{$a}};($a:expr,$b:expr$(,)*)=>{{let(ar,br)=($a,$b);if ar>br{ar}else{br}}};($a:expr,$($rest:expr),+$(,)*)=>{{let b=max!($($rest),+);let ar=$a;if ar>b{ar}else{b}}};}
// codesnip-guard: min
#[cfg_attr(any(),rustfmt::skip)]#[allow(unused_macros)]#[macro_export]macro_rules!min{($a:expr$(,)*)=>{{$a}};($a:expr,$b:expr$(,)*)=>{{let(ar,br)=($a,$b);if ar>br{br}else{ar}}};($a:expr,$($rest:expr),+$(,)*)=>{{let b=min!($($rest),+);let ar=$a;if ar>b{b}else{ar}}};}
// codesnip-guard: prelude
#[cfg_attr(any(),rustfmt::skip)]pub use std::{cmp::{max,min,Ordering,Reverse},collections::{hash_map::RandomState,BTreeMap,BTreeSet,BinaryHeap,VecDeque},convert::Infallible,convert::{TryFrom,TryInto},default::Default,fmt::{Debug,Display,Formatter},hash::{BuildHasherDefault,Hash,Hasher},io::{stdin,stdout,BufRead,BufWriter,Read,StdoutLock,Write},iter::{repeat,FromIterator,Product,Sum},marker::PhantomData,mem::swap,ops::{Add,AddAssign,BitAnd,BitAndAssign,BitOr,BitOrAssign,BitXor,BitXorAssign,Bound,Deref,DerefMut,Div,DivAssign,Index,IndexMut,Mul,MulAssign,Neg,Not,Range,RangeBounds,RangeInclusive,Rem,RemAssign,Shl,ShlAssign,Shr,ShrAssign,Sub,SubAssign},str::{from_utf8,FromStr}};
// codesnip-guard: range-traits
#[cfg_attr(any(),rustfmt::skip)]pub use range_traits_impl::{PointUpdate,RangeProduct,RangeProductMut,RangeUpdate,ToBounds};
#[cfg_attr(any(),rustfmt::skip)]mod range_traits_impl{use super::{Add,Bound,BoundedAbove,BoundedBelow,Magma,One,RangeBounds};pub trait ToBounds<T>{fn lr(&self)->(T,T);}impl<R:RangeBounds<T>+Clone,T:Copy+BoundedAbove+BoundedBelow+One+Add<Output=T>>ToBounds<T>for R{#[inline]fn lr(&self)->(T,T){use Bound::{Excluded,Included,Unbounded};let l=match self.start_bound(){Unbounded=>T::min_value(),Included(&s)=>s,Excluded(&s)=>s+T::one(),};let r=match self.end_bound(){Unbounded=>T::max_value(),Included(&e)=>e+T::one(),Excluded(&e)=>e,};(l,r)}}#[doc=" # 二項演算の総積クエリを提供する"]#[doc=" 遅延評価などを持つデータ構造は、&mut selfを要求するRangeProductMutを使用する"]pub trait RangeProduct<I>{type Magma:Magma;fn product<R:ToBounds<I>>(&self,range:R)-><Self::Magma as Magma>::M;}pub trait RangeProductMut<I>{type Magma:Magma;fn product<R:ToBounds<I>>(&mut self,range:R)-><Self::Magma as Magma>::M;}impl<T:RangeProduct<I>,I>RangeProductMut<I>for T{type Magma=T::Magma;fn product<R:ToBounds<I>>(&mut self,range:R)-><Self::Magma as Magma>::M{<Self as RangeProduct<I>>::product(self,range)}}#[doc=" # 値の更新"]#[doc=" indexで指定した値をfで更新する"]pub trait PointUpdate<T>{fn update_at(&mut self,index:usize,f:T);}#[doc=" # 区間の更新"]#[doc=" rangeで指定した値をfで更新する"]pub trait RangeUpdate<I,T>{fn update_range<R:ToBounds<I>>(&mut self,range:R,f:T);}}
// codesnip-guard: string-util
#[cfg_attr(any(),rustfmt::skip)]pub use string_util_impl::{AddLineTrait,BitsTrait,JoinTrait,YesTrait};
#[cfg_attr(any(),rustfmt::skip)]mod string_util_impl{use super::{Display,Integral};pub trait AddLineTrait{fn line(&self)->String;}impl<D:Display>AddLineTrait for D{fn line(&self)->String{self.to_string()+"\n"}}pub trait JoinTrait{#[doc=" # separatorで結合する"]fn join(self,separator:&str)->String;}impl<D:Display,I:IntoIterator<Item=D>>JoinTrait for I{fn join(self,separator:&str)->String{let mut buf=String::new();self.into_iter().fold("",|sep,arg|{buf.push_str(&format!("{}{}",sep,arg));separator});buf}}pub trait BitsTrait{fn bits(self,length:Self)->String;}impl<I:Integral>BitsTrait for I{fn bits(self,length:Self)->String{let mut buf=String::new();let mut i=I::zero();while i<length{buf.push_str(&format!("{}",self>>i&I::one()));i+=I::one();}buf+"\n"}}pub trait YesTrait{fn yes(self)->String;fn no(self)->String;}impl YesTrait for bool{#[inline]fn yes(self)->String{if self{"Yes"}else{"No"}.to_string()}#[inline]fn no(self)->String{if self{"No"}else{"Yes"}.to_string()}}}
// codesnip-guard: union-find-tree
#[cfg_attr(any(),rustfmt::skip)]#[derive(Clone,Default)]pub struct UnionFind{parent:Vec<usize>,rank:Vec<usize>,size:Vec<usize>}
#[cfg_attr(any(),rustfmt::skip)]impl UnionFind{#[doc=" # 初期化"]#[doc=" 1-indexedで$n$まで初期化される"]pub fn new(n:usize)->Self{let parent=(0..n+1).collect::<Vec<_>>();let rank=vec![0;n+1];let size=vec![1;n+1];Self{parent,rank,size}}pub fn resize(&mut self,n:usize){while self.parent.len()<n{self.parent.push(self.parent.len());self.rank.push(0);self.size.push(1);}}pub fn root(&mut self,x:usize)->usize{if self.parent[x]==x{x}else{self.parent[x]=self.root(self.parent[x]);self.parent[x]}}pub fn rank(&self,x:usize)->usize{self.rank[x]}pub fn size(&mut self,x:usize)->usize{let root=self.root(x);self.size[root]}pub fn same(&mut self,x:usize,y:usize)->bool{self.root(x)==self.root(y)}#[doc=" # 併合する"]#[doc=" ## 返り値"]#[doc=" 新たに併合したときtrue 何もしなかった場合はfalse"]pub fn unite(&mut self,x:usize,y:usize)->bool{let mut x=self.root(x);let mut y=self.root(y);if x==y{return false;}if self.rank(x)<self.rank(y){swap(&mut x,&mut y);}if self.rank(x)==self.rank(y){self.rank[x]+=1;}self.parent[y]=x;self.size[x]+=self.size[y];true}}
// codesnip-guard: weighted-union-find
#[cfg_attr(any(),rustfmt::skip)]#[derive(Clone,Debug)]pub struct WeightedUnionFind<A:AbelianGroup>{parent:Vec<usize>,rank:Vec<usize>,weight_diff:Vec<A::M>,abelian_group:A}
#[cfg_attr(any(),rustfmt::skip)]impl<A:AbelianGroup>WeightedUnionFind<A>{pub fn new(n:usize,abelian_group:A)->Self{let parent=(0..n+1).collect::<Vec<_>>();let rank=vec![0;n+1];let weight_diff=vec![A::unit();n+1];Self{parent,rank,weight_diff,abelian_group}}pub fn root(&mut self,x:usize)->usize{if self.parent[x]==x{x}else{let r=self.root(self.parent[x]);self.weight_diff[x]=self.abelian_group.op(&self.weight_diff[x],&self.weight_diff[self.parent[x]]);self.parent[x]=r;self.parent[x]}}pub fn rank(&self,x:usize)->usize{self.rank[x]}pub fn same(&mut self,x:usize,y:usize)->bool{self.root(x)==self.root(y)}pub fn weight(&mut self,x:usize)->A::M{self.root(x);self.weight_diff[x].clone()}#[doc=" xとyがすでに併合されているとき、併合せずfalseを返す"]pub fn unite(&mut self,x:usize,y:usize,mut weight:A::M)->bool{let(wx,wy)=(self.weight(x),self.weight(y));weight=self.abelian_group.op(&weight,&wx);weight=self.abelian_group.op(&weight,&A::inv(&wy));let(mut x,mut y)=(self.root(x),self.root(y));if x==y{return false;}if self.rank(x)<self.rank(y){swap(&mut x,&mut y);weight=A::inv(&weight);}if self.rank(x)==self.rank(y){self.rank[x]+=1;}self.parent[y]=x;self.weight_diff[y]=weight;true}pub fn diff(&mut self,x:usize,y:usize)->A::M{assert_eq!(self.root(x),self.root(y));let(wx,wy)=(self.weight(x),self.weight(y));self.abelian_group.op(&wy,&A::inv(&wx))}}
// codesnip-guard: addition
#[cfg_attr(any(),rustfmt::skip)]#[derive(Clone,Debug,Default)]pub struct Addition<Lhs,Rhs=Lhs>(PhantomData<fn()->(Lhs,Rhs)>);
#[cfg_attr(any(),rustfmt::skip)]mod addition_impl{use super::{Add,Addition,Associative,Commutative,Debug,Invertible,Magma,Mapping,Neg,Unital,Zero};impl<S:Clone+Debug+Add<Output=S>+PartialEq>Magma for Addition<S>{type M=S;fn op(&mut self,x:&S,y:&S)->S{x.clone()+y.clone()}}impl<S:Clone+Debug+Add<Output=S>+PartialEq>Associative for Addition<S>{}impl<S:Clone+Debug+Add<Output=S>+PartialEq+Zero>Unital for Addition<S>{fn unit()->S{S::zero()}}impl<S:Clone+Debug+Add<Output=S>+PartialEq>Commutative for Addition<S>{}impl<S:Clone+Debug+Add<Output=S>+PartialEq+Neg<Output=S>>Invertible for Addition<S>{fn inv(x:&S)->S{x.clone().neg()}}impl<S:Clone+Debug+PartialEq,T:Clone+Debug+Add<S,Output=T>>Mapping for Addition<S,T>{type Mapping=S;type Domain=T;type Codomain=T;fn apply(&mut self,map:&S,value:&T)->T{value.clone()+map.clone()}}}
// codesnip-guard: bit-combination
#[cfg_attr(any(),rustfmt::skip)]pub fn bit_combination(n:usize,k:usize)->impl Iterator<Item=u64>{assert!(n<64);assert!(k<=n);std::iter::successors(Some(!(!0u64<<k)),move|&v|{if k==0{return None;}let x=v&v.wrapping_neg();let y=v+x;let z=(v&!y)>>(x.trailing_zeros()+1);Some(y|z)}).take_while(move|&v|v<(1<<n))}

// codesnip-guard: zzz-solver
pub fn solve(read: impl Read, mut write: impl Write) {
    let mut reader = ReadHelper::new(read);
    let (n, m) = reader.v2::<usize, usize>();
    let abc = reader.vec3::<usize, usize, i64>(m);
    let mut uf = WeightedUnionFind::<Addition<i64>>::new(n, Addition::default());
    for (a, b, c) in abc {
        let (a, b) = (a - 1, b - 1);
        uf.unite(a, b, -c);
    }
    let mut pieces = Vec::new();
    let mut roots = (0..n).map(|i| uf.root(i)).collect::<Vec<_>>();
    roots.sort();
    roots.dedup();
    for root in roots {
        let mut t = Vec::new();
        for i in 0..n {
            if uf.same(i, root) {
                t.push((uf.weight(i), i));
            }
        }
        t.sort();
        let mut offset = -1 << 60;
        let mut p: usize = 0;
        for (diff, _i) in &t {
            if offset == -1 << 60 {
                offset = *diff
            }
            p |= 1 << (diff - offset)
        }
        assert_ne!(p, 0);
        pieces.push((p, t));
    }
    let p = pieces.len();
    let mut pred = vec![vec![false; 1 << n]; p + 1];
    pred[0][0] = true;
    let mut total_bits = 0;
    for i in 0..p {
        let (pi, _) = pieces[i];
        for from in bit_combination(n, total_bits) {
            if !pred[i][from as usize] {
                continue;
            }
            let mut pi = pi;
            while pi < 1 << n {
                if from as usize & pi == 0 {
                    pred[i + 1][from as usize | pi] = true;
                }
                pi <<= 1;
            }
        }
        total_bits += pi.count_ones() as usize;
    }
    let mut succ = vec![vec![false; 1 << n]; p + 1];
    succ[p][0] = true;
    let mut total_bits = 0;
    for i in (0..p).rev() {
        let (pi, _) = pieces[i];
        for from in bit_combination(n, total_bits) {
            if !succ[i + 1][from as usize] {
                continue;
            }
            let mut pi = pi;
            while pi < 1 << n {
                if from as usize & pi == 0 {
                    succ[i][from as usize | pi] = true;
                }
                pi <<= 1;
            }
        }
        total_bits += pi.count_ones() as usize;
    }
    // dbg!(&pieces);
    let mut ans = vec![-1; n];
    let mut total_bits = 0;
    for unused in 0..p {
        // unused な開始位置
        let mut exist = -1;
        let mut twice_exist = false;
        let (pp, pi) = &pieces[unused];
        'index: for s in 0..n {
            let t = pp << s;
            if t >= 1 << n {
                break;
            }
            // dbg!(t.bits(p));
            for pre in bit_combination(n, total_bits) {
                if !pred[unused][pre as usize] || t & pre as usize > 0 {
                    continue;
                }
                let expect = (1 << n) - 1 - t - pre as usize;
                // dbg!(pre.bits(n as u64), expect.bits(n), succ[unused + 1][expect]);
                if succ[unused + 1][expect] {
                    if exist == -1 {
                        exist = s as i64;
                        continue 'index;
                    } else {
                        twice_exist = true;
                        break 'index;
                    }
                }
            }
        }
        total_bits += pp.count_ones() as usize;
        assert_ne!(exist, -1, "{unused}");
        if !twice_exist {
            let mut offset = -1 << 60;
            for (diff, i) in pi {
                if offset == -1 << 60 {
                    offset = *diff;
                }
                ans[*i] = diff - offset + exist + 1;
            }
        }
    }
    writeln!(write, "{}", ans.join(" ")).unwrap()
}