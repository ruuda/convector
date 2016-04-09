//! This mod writes trace logs that can be inspected with chrome://tracing.
//! It is intended as a debugging tool, so I can see what all the cores are
//! doing; how work is scheduled among CPUs and what is blocking.
//!
//! Note: this mod is not related to ray tracing, sorry for the name.

// TODO: Integrate this with stats.

use std::collections::VecDeque;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use thread_id;
use time::{Duration, PreciseTime};

struct TraceEvent {
    start: PreciseTime,
    end: PreciseTime,
    description: &'static str,
    frame: u32,
    id: u32,
    tid: u64,
}

pub struct ScopedTraceEvent {
    start: PreciseTime,
    description: &'static str,
    frame: u32,
    id: u32,
    log: Arc<Mutex<TraceLogImpl>>,
    handled: bool,
}

struct TraceLogImpl {
    events: VecDeque<TraceEvent>,
    limit: usize,
}

pub struct TraceLog {
    log: Arc<Mutex<TraceLogImpl>>,
    epoch: PreciseTime,
    frame_number: u32,
}

impl ScopedTraceEvent {
    /// Records the event in the trace log and returns its duration.
    pub fn take_duration(mut self) -> Duration {
        let end = PreciseTime::now();
        self.add_to_trace(end);
        self.start.to(end)
    }

    fn add_to_trace(&mut self, now: PreciseTime) {
        let event = TraceEvent {
            start: self.start,
            end: now,
            description: self.description,
            frame: self.frame,
            id: self.id,
            tid: thread_id::get() as u64,
        };
        let mut trace_log_impl = self.log.lock().unwrap();
        if trace_log_impl.events.len() == trace_log_impl.limit {
            trace_log_impl.events.pop_front();
        }
        trace_log_impl.events.push_back(event);
        self.handled = true;
    }
}

impl Drop for ScopedTraceEvent {
    fn drop(&mut self) {
        if !self.handled {
            let end = PreciseTime::now();
            self.add_to_trace(end);
        }
    }
}

impl TraceLog {
    pub fn with_limit(limit: usize) -> TraceLog {
        let trace_log_impl = TraceLogImpl {
            events: VecDeque::with_capacity(limit),
            limit: limit,
        };
        TraceLog {
            log: Arc::new(Mutex::new(trace_log_impl)),
            epoch:  PreciseTime::now(),
            frame_number: 0,
        }
    }

    /// Increments the frame number and returns the current frame number.
    pub fn inc_frame_number(&mut self) -> u32 {
        self.frame_number += 1;
        self.frame_number
    }

    /// Starts a new trace event. When the returned value goes out of scope, it
    /// is added to the log with the correct end time.
    pub fn scoped(&self, description: &'static str, id: u32) -> ScopedTraceEvent {
        ScopedTraceEvent {
            start: PreciseTime::now(),
            description: description,
            frame: self.frame_number,
            id: id,
            log: self.log.clone(),
            handled: false,
        }
    }

    /// Writes the trace as a json string in the trace log format that can be
    /// read by Chrome’s trace viewer (chrome://tracing).
    pub fn export<W: io::Write>(&self, output: &mut W) -> io::Result<()> {
        try!(write!(output, "{{\"traceEvents\":["));
        let mut is_first = true;
        for event in self.log.lock().unwrap().events.iter() {
            if !is_first {
                try!(write!(output, ","));
            }
            let ts = self.epoch.to(event.start).num_microseconds().unwrap();
            let dur = event.start.to(event.end).num_microseconds().unwrap();
            try!(write!(output, "{{\"name\":\"{0}\",\
                                   \"cat\":\"\",\
                                   \"ph\":\"X\",\
                                   \"ts\":{1},\
                                   \"dur\":{2},\
                                   \"pid\":0,\
                                   \"tid\":{3},\
                                   \"args\":{{\
                                   \"frame\":{4},\
                                   \"id\":{5}}}}}",
                                event.description, ts, dur, event.tid,
                                event.frame, event.id));
            is_first = false;
        }
        write!(output, "],\"displayTimeUnit\":\"ms\"}}")
    }

    /// Writes the trace to a json file.
    pub fn export_to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = try!(File::create(path));
        self.export(&mut file)
    }
}
