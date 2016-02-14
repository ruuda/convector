//! A simple way to keep track of statistics.

/// Keeps track of the min, median, and max of a variable.
///
/// The number of values stored is bounded.
pub struct Stats<T: Copy + Ord> {
    values: Vec<T>
}

impl<T: Copy + Ord> Stats<T> {
    pub fn new() -> Stats<T> {
        Stats {
            values: Vec::with_capacity(128),
        }
    }

    pub fn insert(&mut self, value: T) {
        // Make room if there is none. Removing one extreme value below the median and one above
        // does not affect the median, so we can discard values without affecting the median.
        // However, when the median shifts, these values could have been imporant, and the result
        // is incorrect. For a stable value, the median will not shift by much, so it is best to
        // remove the most extreme values. On the other hand, the min and max are interesting to
        // know, so remove the values after the min and before the max.
        if self.values.len() == self.values.capacity() {
            debug_assert!(self.values.len() >= 4);
            let len = self.values.len();
            self.values.remove(len - 2);
            self.values.remove(2);
            // TODO: Instead of removing, I could average to be more fair.
        }

        let idx = match self.values.binary_search(&value) {
            Ok(i) => i,
            Err(i) => i,
        };

        self.values.insert(idx, value);
    }

    /// Returns the median of the stored values.
    ///
    /// Panics if no values are present.
    pub fn median(&self) -> T {
        // This is not correct for an even number of values, but as the number
        // of values grows bigger this difference becomes smaller.
        self.values[self.values.len() / 2]
    }

    /// Returns the minimum of the stored values.
    ///
    /// Panics if no values are present.
    pub fn min(&self) -> T {
        self.values[0]
    }

    /// Returns the maximum of the stored values.
    ///
    /// Panics if no values are present.
    pub fn max(&self) -> T {
        self.values[self.values.len() - 1]
    }
}
