//! This module is responsible for scheduling work across multiple cores.

use num_cpus;
use std::thread;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, Sender, channel};

type Work = Box<FnOnce() + Send>;

struct Scheduler {
    worker_signals: Vec<Sender<()>>,
    work: Arc<Mutex<Vec<Work>>>,
}

impl Scheduler {
    pub fn new() -> Scheduler {
        Scheduler::with_concurrency(num_cpus::get())
    }

    pub fn with_concurrency(concurrency: usize) -> Scheduler {
        let mut worker_signals = Vec::with_capacity(concurrency);
        let work = Arc::new(Mutex::new(Vec::new()));
        for _ in 0 .. concurrency {
            let (signal_tx, signal_rx) = channel();
            let work_ref = work.clone();
            thread::spawn(move || Scheduler::run_worker(signal_rx, work_ref));
            worker_signals.push(signal_tx);
        }
        Scheduler {
            worker_signals: worker_signals,
            work: work,
        }
    }

    fn run_worker(signal_rx: Receiver<()>, work: Arc<Mutex<Vec<Work>>>) {
        // Block until the sender signals, or stop if the sender has quit.
        while let Ok(()) = signal_rx.recv() {
            // Keep executing fork from the queue until it is empty.
            while let Some(task) = work.lock().unwrap().pop() {
                task();
            }
        }
    }
}
