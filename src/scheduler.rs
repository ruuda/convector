//! This module is responsible for scheduling work across multiple cores.

use num_cpus;
use std::thread;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, Sender, channel};

type Work = Box<FnMut() + Send>;

struct Scheduler {
    worker_signals: Vec<Sender<Sender<()>>>,
    work_queue: Arc<Mutex<Vec<Work>>>,
}

struct WaitToken {
    worker_done_signals: Vec<Receiver<()>>,
}

impl Scheduler {
    pub fn new() -> Scheduler {
        Scheduler::with_concurrency(num_cpus::get())
    }

    pub fn with_concurrency(concurrency: usize) -> Scheduler {
        let mut worker_signals = Vec::with_capacity(concurrency);
        let work_queue = Arc::new(Mutex::new(Vec::new()));
        for _ in 0 .. concurrency {
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

    fn run_worker(signal_rx: Receiver<Sender<()>>,
                  work_queue: Arc<Mutex<Vec<Work>>>) {
        // Block until the sender signals, or stop if the sender has quit.
        while let Ok(done_sender) = signal_rx.recv() {
            // Keep executing fork from the queue until it is empty.
            while let Some(mut task) = work_queue.lock().unwrap().pop() {
                task();
            }
            // Signal that this worker has completed.
            done_sender.send(());
        }
    }

    /// Executes all of the work in worker threads. Returns immediately.
    fn do_work_async(&mut self, mut work: Vec<Work>) -> WaitToken {
        // First put the work into the queue.
        let mut locked_queue = self.work_queue.lock().unwrap();
        // TODO: Is there a method for this?
        for w in work.drain(..) {
            locked_queue.push(w);
        }

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
    pub fn wait(self) {
        // Wait for one signal from every worker.
        for done_signal in self.worker_done_signals {
            done_signal.recv().ok().expect("worker thread quit unexpectedly");
        }
    }
}
