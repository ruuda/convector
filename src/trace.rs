//! This mod writes trace logs that can be inspected with chrome://tracing.
//! It is intended as a debugging tool, so I can see what all the cores are
//! doing; how work is scheduled among CPUs and what is blocking.

use std::collections::VecDeque;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};
use thread_id;
use time::PreciseTime;

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
}

struct TraceLogImpl {
    events: VecDeque<TraceEvent>,
    limit: usize,
}

pub struct TraceLog {
    log: Arc<Mutex<TraceLogImpl>>,
    epoch: PreciseTime,
}

impl Drop for ScopedTraceEvent {
    fn drop(&mut self) {
        let event = TraceEvent {
            start: self.start,
            end: PreciseTime::now(),
            description: self.description,
            frame: self.frame,
            id: self.id,
            tid: thread_id::get(),
        };
        let mut trace_log_impl = self.log.lock().unwrap();
        if trace_log_impl.events.len() == trace_log_impl.limit {
            trace_log_impl.events.pop_front();
        }
        trace_log_impl.events.push_back(event);
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
        }
    }

    /// Starts a new trace event. When the returned value goes out of scope, it
    /// is added to the log with the correct end time.
    pub fn scoped(&self, description: &'static str, frame: u32, id: u32) -> ScopedTraceEvent {
        ScopedTraceEvent {
            start: PreciseTime::now(),
            description: description,
            frame: frame,
            id: id,
            log: self.log.clone(),
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
