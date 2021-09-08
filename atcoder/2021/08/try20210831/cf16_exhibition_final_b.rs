
#[allow(unused_imports)]
use geometric::*;

#[allow(dead_code)]
mod geometric {
    use std::f64;
    use std::fmt;
    use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

    #[derive(Copy, Clone, PartialEq)]
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
            Point { x, y }
        }

        /// 偏角を求める(radian)
        /// 原点だった場合はNone
        pub fn declination(&self) -> Option<f64> {
            use std::f64::consts::PI;
            if self.x == 0.0 {
                if self.y == 0.0 {
                    None
                } else if self.y > 0.0 {
                    Some(PI / 2.0)
                } else {
                    Some(PI * 3.0 / 2.0)
                }
            } else {
                Some(
                    (libm::atan(self.y / self.x) + if self.x > 0.0 { 0.0 } else { PI })
                        .rem_euclid(PI * 2.0),
                )
            }
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
    }

    #[derive(Copy, Clone)]
    pub struct Triangle {
        p1: Point,
        p2: Point,
        p3: Point,
    }

    impl Triangle {
        pub fn new(p1: Point, p2: Point, p3: Point) -> Triangle {
            Triangle { p1, p2, p3 }
        }

        /// 内心を求める
        pub fn inner_center(&self) -> Option<Point> {
            let line = Line::new(self.p1, self.p2);
            if line.distance(self.p3) > 0.0 {
                Some((self.p1 + self.p2 + self.p3) / 3.0)
            } else {
                None
            }
        }

        ///内接円の半径
        pub fn inner_circle_radius(&self) -> f64 {
            let a = (self.p1 - self.p2).abs();
            let b = (self.p2 - self.p3).abs();
            let c = (self.p3 - self.p1).abs();
            let s = self.area();
            2.0 * s / (a + b + c)
        }

        /// 面積を求める
        pub fn area(&self) -> f64 {
            let a = self.p2 - self.p1;
            let b = self.p3 - self.p1;
            (a.x * b.y - a.y * b.x).abs() / 2.0
        }

        /// 外心を求める
        pub fn circumcenter(&self) -> Option<Point> {
            let p1p2 = Line::new(
                (self.p1 + self.p2) / 2.0,
                (self.p1 + self.p2) / 2.0 + (self.p1 - self.p2).rot90(),
            );
            let p2p3 = Line::new(
                (self.p2 + self.p3) / 2.0,
                (self.p2 + self.p3) / 2.0 + (self.p2 - self.p3).rot90(),
            );
            Line::cross_points(p1p2, p2p3)
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

    impl fmt::Debug for Point {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Point: (x: {}, y: {})", self.x, self.y)
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

        pub fn cross_points_as_segment(l: Self, m: Self) -> Option<Point> {
            let p = Self::cross_points(l, m);
            match p {
                Some(p) => {
                    if (p - l.p1).abs() + (l.p2 - p).abs() - (l.p2 - l.p1).abs() < f64::EPSILON
                        && (p - m.p1).abs() + (m.p2 - p).abs() - (m.p2 - m.p1).abs() < f64::EPSILON
                    {
                        Some(p)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }

        /// xを与えたときのyの値を求める
        pub fn y(self, x: f64) -> Option<f64> {
            if (self.p1.x - self.p2.x).abs() < f64::EPSILON {
                None
            } else {
                Some(
                    self.p1.y + (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x) * (x - self.p1.x),
                )
            }
        }

        /// yを与えたときのxの値を求める
        pub fn x(self, y: f64) -> Option<f64> {
            if (self.p1.y - self.p2.y).abs() < f64::EPSILON {
                None
            } else {
                Some(
                    self.p1.x + (self.p2.x - self.p1.x) / (self.p2.y - self.p1.y) * (y - self.p1.y),
                )
            }
        }

        /// 直線と点の距離
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
