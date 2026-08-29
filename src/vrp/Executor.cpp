#include "vrp/Executor.hpp"

#include <algorithm>
#include <cassert>

namespace vrp {

std::pair<std::size_t, std::size_t> chunkRange(std::size_t n, std::size_t parts,
                                               std::size_t index) noexcept {
    assert(parts >= 1 && "chunkRange requires at least one part");
    assert(index < parts && "chunkRange requires index < parts");
    const std::size_t base = n / parts;
    const std::size_t remainder = n % parts;
    const std::size_t begin = index * base + std::min(index, remainder);
    const std::size_t end = begin + base + (index < remainder ? 1u : 0u);
    return {begin, end};
}

void SerialExecutor::parallelFor(std::size_t n, const ParallelBody& body) {
    if (n == 0) {
        return;
    }
    body(0, n);
}

ThreadPoolExecutor::ThreadPoolExecutor(std::size_t threads) : threads_(threads) {
    assert(threads >= 1 && "ThreadPoolExecutor requires at least one thread");
    workers_.reserve(threads_ - 1);
    for (std::size_t i = 1; i < threads_; ++i) {
        workers_.emplace_back([this, i](std::stop_token stop) { workerLoop(stop, i); });
    }
}

ThreadPoolExecutor::~ThreadPoolExecutor() {
    for (auto& worker : workers_) {
        worker.request_stop();
    }
    startSignal_.notify_all();
    // std::jthread destructors join.
}

void ThreadPoolExecutor::parallelFor(std::size_t n, const ParallelBody& body) {
    if (n == 0) {
        return;
    }
    {
        const std::lock_guard lock(mutex_);
        body_ = &body;
        itemCount_ = n;
        outstanding_ = threads_ - 1;
        ++epoch_;
    }
    startSignal_.notify_all();

    // The workers are already in flight and are reading body_, so the pool has to
    // be drained on every exit path -- including the one where the caller's own
    // chunk throws (Task 7 allocates its per-chunk scratch buffers inside exactly
    // this body, so bad_alloc is reachable). Unwinding without draining would
    // leave body_ pointing at a std::function the unwinding then destroys, and
    // would leave outstanding_ non-zero for the next call to overwrite, so the
    // stale workers' decrements would underflow it and that caller would block
    // forever.
    try {
        const auto [begin, end] = chunkRange(n, threads_, 0);
        if (begin < end) {
            body(begin, end);
        }
    } catch (...) {
        waitForWorkers();
        throw;
    }
    waitForWorkers();
}

void ThreadPoolExecutor::waitForWorkers() noexcept {
    std::unique_lock lock(mutex_);
    doneSignal_.wait(lock, [this] { return outstanding_ == 0; });
    body_ = nullptr;
}

void ThreadPoolExecutor::workerLoop(std::stop_token stop, std::size_t index) {
    std::uint64_t seenEpoch = 0;
    for (;;) {
        std::unique_lock lock(mutex_);
        // wait returns the final value of the predicate, so false means the stop
        // was requested with no work pending. Testing stop_requested() separately
        // would instead discard an epoch published just before the stop.
        const auto hasWork = [this, seenEpoch] { return epoch_ != seenEpoch; };
        if (!startSignal_.wait(lock, stop, hasWork)) {
            return;
        }
        seenEpoch = epoch_;
        const std::size_t n = itemCount_;
        const ParallelBody* body = body_;
        lock.unlock();

        const auto [begin, end] = chunkRange(n, threads_, index);
        if (begin < end) {
            (*body)(begin, end);
        }

        lock.lock();
        const bool last = (--outstanding_ == 0);
        lock.unlock();
        if (last) {
            doneSignal_.notify_one();
        }
    }
}

std::unique_ptr<Executor> makeExecutor(std::size_t threads) {
    if (threads <= 1) {
        return std::make_unique<SerialExecutor>();
    }
    return std::make_unique<ThreadPoolExecutor>(threads);
}

}  // namespace vrp
