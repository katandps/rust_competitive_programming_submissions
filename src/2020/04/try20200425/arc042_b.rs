#[allow(dead_code)]
fn main() {
    let (x, y) = i::i2();
    let n = i::u();
    let xy = i::iv2(n);

    let n1 = Line::new(
        Point::new(xy[0].0 as f64, xy[0].1 as f64),
        Point::new(xy[n - 1].0 as f64, xy[n - 1].1 as f64),
    );
    let p = Point::new(x as f64, y as f64);
    let mut ans = n1.distance(p);

    for i in 1..n {
        let l = Line::new(
            Point::new(xy[i].0 as f64, xy[i].1 as f64),
            Point::new(xy[i - 1].0 as f64, xy[i - 1].1 as f64),
        );
        let d = l.distance(p);
        if ans - d > 0.0 {
            ans = d
        };
    }
    println!("{}", ans);
}

#[allow(unused_imports)]
use geometric::*;

#[allow(dead_code)]
mod geometric {
    use std::f64;
    use std::fmt;
    use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

    #[derive(Copy, Clone)]
    pub struct Point {
        pub x: f64,
        pub y: f64,
    }

    #[derive(Copy, Clone)]
    pub struct Line {
        pub p1: Point,
        pub p2: Point,
    }

    impl Point {
        pub fn new(x: f64, y: f64) -> Point {
            Point { x: x, y: y }
        }

        /// 原点を軸にradian回転させる
        pub fn rot(self, radian: f64) -> Point {
            Point::new(
                radian.cos() * self.x - radian.sin() * self.y,
                radian.sin() * self.x + radian.cos() * self.y,
            )
        }

        /// 原点を軸にpi/2回転させる
        pub fn rot90(self) -> Point {
            Point::new(-self.y, self.x)
        }

        /// x軸に対して反転
        pub fn conj(self) -> Point {
            Point::new(self.x, -self.y)
        }

        /// 外積を求める
        pub fn cross(p: Self, q: Self) -> f64 {
            p.x * q.y - p.y * q.x
        }

        /// 内積を求める
        pub fn dot(p: Self, q: Self) -> f64 {
            p.x * q.x + p.y * p.y
        }

        /// ノルムを求める
        pub fn norm(self) -> f64 {
            Self::dot(self, self)
        }

        /// 大きさを求める
        pub fn abs(self) -> f64 {
            self.norm().sqrt()
        }

        /// 外心を求める
        pub fn circumcenter(p: Self, q: Self, r: Self) -> Option<Point> {
            let pq = Line::new((p + q) / 2.0, (p + q) / 2.0 + (p - q).rot90());
            let qr = Line::new((q + r) / 2.0, (q + r) / 2.0 + (q - r).rot90());
            Line::cross_points(pq, qr)
        }
    }

    impl Add<Point> for Point {
        type Output = Point;
        fn add(self, rhs: Point) -> Point {
            Point::new(self.x + rhs.x, self.y + rhs.y)
        }
    }

    impl AddAssign<Point> for Point {
        fn add_assign(&mut self, other: Point) {
            *self = *self + other;
        }
    }

    impl Sub<Point> for Point {
        type Output = Point;
        fn sub(self, rhs: Point) -> Point {
            Point::new(self.x - rhs.x, self.y - rhs.y)
        }
    }

    impl SubAssign<Point> for Point {
        fn sub_assign(&mut self, other: Point) {
            *self = *self - other;
        }
    }

    impl Mul<f64> for Point {
        type Output = Point;
        fn mul(self, rhs: f64) -> Point {
            Point::new(self.x * rhs, self.y * rhs)
        }
    }

    impl MulAssign<f64> for Point {
        fn mul_assign(&mut self, other: f64) {
            *self = *self * other;
        }
    }

    impl Div<f64> for Point {
        type Output = Point;
        fn div(self, rhs: f64) -> Point {
            Point::new(self.x / rhs, self.y / rhs)
        }
    }

    impl DivAssign<f64> for Point {
        fn div_assign(&mut self, other: f64) {
            *self = *self / other;
        }
    }

    impl fmt::Display for Point {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "x:{} y:{}", self.x, self.y)
        }
    }

    impl Line {
        pub fn new(p: Point, q: Point) -> Line {
            Line { p1: p, p2: q }
        }

        pub fn cross(l: &Self, m: &Self) -> f64 {
            Point::cross(m.p2 - m.p1, l.p2 - l.p1)
        }

        /// 交点を求める
        pub fn cross_points(l: Self, m: Self) -> Option<Point> {
            let d = Self::cross(&l, &m);
            if d.abs() < f64::EPSILON {
                None
            } else {
                Some(l.p1 + (l.p2 - l.p1) * Point::cross(m.p2 - m.p1, m.p2 - l.p1) / d)
            }
        }

        pub fn distance(self, p: Point) -> f64 {
            if self.p1.x == self.p2.x {
                return (p.x - self.p1.x).abs();
            }
            if self.p1.y == self.p2.y {
                return (p.y - self.p1.y).abs();
            }
            let l = Line::new(p, p + (self.p2 - self.p1).rot90());
            match Self::cross_points(self, l) {
                Some(cp) => (p - cp).abs(),
                None => 0.0,
            }
        }
    }

    impl fmt::Display for Line {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{} - {}", self.p1, self.p2)
        }
    }
}

#[allow(unused_imports)]
use std::cmp::*;
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet, VecDeque};
#[allow(unused_imports)]
use std::io::*;
#[allow(unused_imports)]
use std::num::*;
#[allow(unused_imports)]
use std::str::*;

#[allow(dead_code)]
mod i {
    use super::*;

    pub fn read<T: FromStr>() -> T {
        stdin()
            .bytes()
            .map(|c| c.unwrap() as char)
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| !c.is_whitespace())
            .collect::<String>()
            .parse::<T>()
            .ok()
            .unwrap()
    }

    pub fn str() -> String {
        read()
    }

    pub fn s() -> Vec<char> {
        str().chars().collect()
    }

    pub fn i() -> i64 {
        read()
    }

    pub fn i2() -> (i64, i64) {
        (read(), read())
    }

    pub fn u() -> usize {
        read()
    }

    pub fn u2() -> (usize, usize) {
        (read(), read())
    }

    pub fn u3() -> (usize, usize, usize) {
        (read(), read(), read())
    }

    pub fn f() -> f64 {
        read()
    }

    pub fn c() -> char {
        read::<String>().pop().unwrap()
    }

    pub fn iv(n: usize) -> Vec<i64> {
        (0..n).map(|_| i()).collect()
    }

    pub fn iv2(n: usize) -> Vec<(i64, i64)> {
        (0..n).map(|_| iv(2)).map(|a| (a[0], a[1])).collect()
    }

    pub fn uv(n: usize) -> Vec<usize> {
        (0..n).map(|_| u()).collect()
    }

    pub fn uv2(n: usize) -> Vec<(usize, usize)> {
        (0..n).map(|_| uv(2)).map(|a| (a[0], a[1])).collect()
    }

    pub fn uv3(n: usize) -> Vec<(usize, usize, usize)> {
        (0..n).map(|_| uv(3)).map(|a| (a[0], a[1], a[2])).collect()
    }

    pub fn fv(n: usize) -> Vec<f64> {
        (0..n).map(|_| f()).collect()
    }

    pub fn cmap(h: usize) -> Vec<Vec<char>> {
        (0..h).map(|_| s()).collect()
    }
}
