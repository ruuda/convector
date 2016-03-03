//! This module contains utility functions.

/// Returns the n-th bit of x (zero-based, least significant first).
#[inline(always)]
fn z_bit(x: u16, n: usize) -> u16 {
    (x >> n) & 1
}

/// Removes every odd bit (counting from the least significant bit).
#[inline(always)]
fn z_pack(x: u16) -> u16 {
    // Make a tree of ors to avoid data dependencies.
    let a = (z_bit(x, 0) << 0) | (z_bit(x, 2) << 1);
    let b = (z_bit(x, 4) << 2) | (z_bit(x, 6) << 3);
    let c = (z_bit(x, 8) << 4) | (z_bit(x, 10) << 5);
    let d = (z_bit(x, 12) << 6) | (z_bit(x, 14) << 7);
    (a | b) | (c | d)
}

/// Given an index, returns the x and y coordinate in a z-order curve.
///
/// This is sometimes called a Morton curve.
pub fn z_order(i: u16) -> (u16, u16) {
    let x_inter = i & 0b_0101_0101_0101_0101;
    let y_inter = i & 0b_1010_1010_1010_1010;
    let x = z_pack(x_inter);
    let y = z_pack(y_inter >> 1);
    (x, y)
}

#[test]
fn verify_z_order() {
   let coords: Vec<(u16, u16)> = (0..16).map(|i| z_order(i)).collect();
   let expected = [(0u16, 0), (1, 0), (0, 1), (1, 1),
                   (2, 0), (3, 0), (2, 1), (3, 1),
                   (0, 2), (1, 2), (0, 3), (1, 3),
                   (2, 2), (3, 2), (2, 3), (3, 3)];
   assert_eq!(&coords[..], &expected);
}
