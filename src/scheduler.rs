//! This module is responsible for scheduling work across multiple cores.
//! It is a thread pool with a task queue and support for waiting for
//! completion.

use num_cpus;
use std::thread;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, Sender, channel};

pub type Work = Box<FnMut() + Send>;
pub type WorkRef<'a> = &'a (FnMut() + Send);

pub struct Scheduler {
    worker_signals: Vec<Sender<Sender<()>>>,
    work_queue: Arc<Mutex<Vec<Work>>>,
}

pub struct WaitToken {
    worker_done_signals: Vec<Receiver<()>>,
}

struct Batch<'a> {
    work_queue: Arc<Mutex<Vec<WorkRef<'a>>>>,
    done_signal: Sender<()>,
}

impl Scheduler {
    pub fn new() -> Scheduler {
        Scheduler::with_concurrency(num_cpus::get())
    }

    pub fn with_concurrency(concurrency: usize) -> Scheduler {
        let mut worker_signals = Vec::with_capacity(concurrency);
        let work_queue = Arc::new(Mutex::new(Vec::new()));
        for _ in 0..concurrency {
            let (signal_tx, signal_rx) = channel();
            let work_queue_ref = work_queue.clone();
            thread::spawn(move || Scheduler::run_worker(signal_rx, work_queue_ref));
            worker_signals.push(signal_tx);
        }
        Scheduler {
            worker_signals: worker_signals,
            work_queue: work_queue,
        }
    }

    fn run_worker(batch_rx: Receiver<Batch<'a>>) {
        // Block until there is work to do, or until the sender hangs up.
        while let Ok(batch) = batch_rx.recv() {
            while let Some(mut task) = batch.work_queue.lock().unwrap().pop() {
                task();
            }
            // Signal that this worker has completed.
            batch.done_signal.send(());
        }
    }

    /// Executes all of the work in worker threads. Returns immediately.
    ///
    /// This leaves `work` empty.
    pub fn do_work_async<'a>(&mut self, work: &mut Vec<WorkRef<'a>>) -> WaitToken {
        // First put the work into the queue.
        self.work_queue.lock().unwrap().append(work);

        // Then wake up the workers. Send them the sending end of a channel
        // through which they should signal that they are done.
        let mut worker_done_signals = Vec::with_capacity(self.worker_signals.len());
        for sender in &self.worker_signals {
            let (signal_tx, signal_rx) = channel();
            sender.send(signal_tx);
            worker_done_signals.push(signal_rx);
        }

        WaitToken {
            worker_done_signals: worker_done_signals,
        }
    }
}

impl WaitToken {
    /// A wait token that waits for nothing.
    pub fn empty() -> WaitToken {
        WaitToken {
            worker_done_signals: Vec::new(),
        }
    }

    pub fn wait(self) {
        // Wait for one signal from every worker.
        for done_signal in self.worker_done_signals {
            done_signal.recv().ok().expect("worker thread quit unexpectedly");
        }
    }
}
