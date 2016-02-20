//! A simple way to keep track of statistics.

use time::Duration;

/// Keeps track of the min, median, and max of a variable.
///
/// The number of values stored is bounded.
pub struct Stats {
    values: Vec<u32>
}

impl Stats {
    pub fn new() -> Stats {
        Stats {
            values: Vec::with_capacity(128),
        }
    }

    pub fn insert(&mut self, value: u32) {
        // Make room if there is none. Removing one extreme value below the
        // median and one above does not affect the median, so we can discard
        // values without affecting the median. However, when the median
        // shifts, these values could have been imporant, and the result is
        // incorrect. For a stable value, the median will not shift by much, so
        // it is best to remove the most extreme values. On the other hand, the
        // min and max are interesting to know, so merge the values after the
        // min and before the max.
        if self.values.len() == self.values.capacity() {
            debug_assert!(self.values.len() >= 4);
            let len = self.values.len();
            // Merge the two values after the min and the two values before the
            // max.
            let avg_high = (self.values[len - 3] + self.values[len - 2]) / 2;
            let avg_low = (self.values[1] + self.values[2]) / 2;
            self.values[len - 3] = avg_high;
            self.values[2] = avg_low;
            self.values.remove(len - 2);
            self.values.remove(1);
        }

        let idx = match self.values.binary_search(&value) {
            Ok(i) => i,
            Err(i) => i,
        };

        self.values.insert(idx, value);
    }

    /// Inserts the duration rounded to microseconds.
    pub fn insert_time_us(&mut self, duration: Duration) {
        let ns = duration.num_nanoseconds().unwrap();
        let us = (ns + 500) / 1000;
        self.insert(us as u32);
    }

    /// Returns the median of the stored values.
    ///
    /// Panics if no values are present.
    pub fn median(&self) -> u32 {
        // This is not correct for an even number of values, but as the number
        // of values grows bigger this difference becomes smaller.
        self.values[self.values.len() / 2]
    }

    /// Returns the minimum of the stored values.
    ///
    /// Panics if no values are present.
    pub fn min(&self) -> u32 {
        self.values[0]
    }

    /// Returns the maximum of the stored values.
    ///
    /// Panics if no values are present.
    pub fn max(&self) -> u32 {
        self.values[self.values.len() - 1]
    }
}

/// A collection of global stats that the app keeps track of.
pub struct GlobalStats {
    /// Texture upload time in microseconds.
    pub tex_upload_us: Stats,
    /// Draw and wait for vsync time in microseconds.
    pub draw_vsync_us: Stats,
    /// Total time of rendering and drawing a frame.
    pub frame_us: Stats,
}

impl GlobalStats {
    pub fn new() -> GlobalStats {
        GlobalStats {
            tex_upload_us: Stats::new(),
            draw_vsync_us: Stats::new(),
            frame_us: Stats::new(),
        }
    }

    pub fn print(&self) {
        println!("");
        println!("texture upload: median {} us, min {} us",
                 self.tex_upload_us.median(),
                 self.tex_upload_us.min());
        println!("draw and vsync: median {} us, min {} us",
                 self.draw_vsync_us.median(),
                 self.draw_vsync_us.min());
        println!("frame time: median {} us, min {} us -> {:0.1} fps",
                 self.frame_us.median(),
                 self.frame_us.min(),
                 1.0 / (self.frame_us.median() as f32 * 1e-6));
    }
}
