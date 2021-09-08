use std::cmp::*;
use std::io::*;
use std::num::*;
use std::str::*;

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

    pub fn u() -> usize {
        read()
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

    pub fn uv(n: usize) -> Vec<usize> {
        (0..n).map(|_| u()).collect()
    }

    pub fn fv(n: usize) -> Vec<f64> {
        (0..n).map(|_| f()).collect()
    }

    pub fn cmap(h: usize) -> Vec<Vec<char>> {
        (0..h).map(|_| s()).collect()
    }
}

fn main() {
    let t = i::u();
    let n = i::u();
    let a = i::uv(n);
    let m = i::u();
    let b = i::uv(m);

    let mut aindex = n - 1;
    let mut bindex = m - 1;
    loop {
        if a[aindex] > b[bindex] {
            if aindex == 0 {
                println!("{}", "no");
                //   dbg!(aindex, bindex);
                return;
            }
            aindex -= 1;
            continue;
        } else {
            if a[aindex] + t < b[bindex] {
                println!("{}", "no");
                //   dbg!(aindex, bindex);
                return;
            }
            if bindex == 0 {
                break;
            }
            if aindex == 0 {
                println!("{}", "no");
                //   dbg!(aindex, bindex);
                return;
            }
            aindex -= 1;
            bindex -= 1;
        }
    }
    println!("{}", "yes");
}
