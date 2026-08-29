#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace vrp {

// Receives a half-open range [begin, end), never a single index, so callers can
// hoist scratch allocations out of the per-item loop.
using ParallelBody = std::function<void(std::size_t begin, std::size_t end)>;

// Splits [0, n) into `parts` contiguous, disjoint chunks that together cover
// every index exactly once, and returns chunk `index` as a half-open range.
// Deterministic: the same (n, parts, index) always yields the same range, which
// is what lets a threaded run reproduce a serial one.
//
// Preconditions: parts >= 1 and index < parts. Violating either is undefined
// behaviour -- parts == 0 divides by zero -- so both are asserted in debug
// builds. Callers inside this library always satisfy them by construction.
std::pair<std::size_t, std::size_t> chunkRange(std::size_t n, std::size_t parts,
                                               std::size_t index) noexcept;

class Executor {
public:
    virtual ~Executor() = default;
    virtual std::size_t threadCount() const noexcept = 0;

    // Splits [0, n) across the executor's threads and invokes `body` once per
    // non-empty chunk, returning only after every chunk has run to completion.
    // n == 0 invokes `body` zero times.
    //
    // Usage contract, binding on every implementation:
    //  - Not reentrant: `body` must not call parallelFor on this executor.
    //  - Not concurrency-safe: a single owning thread issues the calls; two
    //    threads must not drive one executor at the same time.
    //  - The executor must not be destroyed while a call is in flight.
    //  - If `body` throws on the calling thread the exception propagates, but
    //    the executor is drained first and stays reusable.
    //  - A throw from a worker chunk, by contrast, terminates the process. It
    //    escapes the worker's thread function, and nothing here catches or
    //    forwards it, so there is no path back to the caller. This is not a
    //    theoretical edge: a body that allocates per-chunk scratch buffers can
    //    raise bad_alloc on a worker exactly as easily as on the caller. A body
    //    that must survive failure has to catch it inside the chunk itself.
    virtual void parallelFor(std::size_t n, const ParallelBody& body) = 0;
};

class SerialExecutor final : public Executor {
public:
    std::size_t threadCount() const noexcept override { return 1; }
    void parallelFor(std::size_t n, const ParallelBody& body) override;
};

// Persistent workers; the calling thread runs chunk 0 rather than idling.
class ThreadPoolExecutor final : public Executor {
public:
    // Requires threads >= 1 (asserted in debug builds; a Release build with
    // threads == 0 raises std::length_error out of the worker reserve, which is
    // a precondition violation, not a supported mode). threads == 1 is legal and
    // degenerates to running every chunk on the calling thread with no workers.
    explicit ThreadPoolExecutor(std::size_t threads);
    ~ThreadPoolExecutor() override;

    ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
    ThreadPoolExecutor& operator=(const ThreadPoolExecutor&) = delete;

    std::size_t threadCount() const noexcept override { return threads_; }
    void parallelFor(std::size_t n, const ParallelBody& body) override;

private:
    void workerLoop(std::stop_token stop, std::size_t index);

    // Blocks until every worker has finished the current epoch, then releases
    // body_. noexcept because it also runs while an exception from the caller's
    // own chunk is propagating: failing to drain there is unrecoverable.
    void waitForWorkers() noexcept;

    // const so the invariant is compiler-enforced: workers read threads_ without
    // synchronisation, which is only safe because it is never written after the
    // constructor publishes it to them.
    const std::size_t threads_;
    std::mutex mutex_;
    std::condition_variable_any startSignal_;
    std::condition_variable doneSignal_;
    const ParallelBody* body_ = nullptr;
    std::size_t itemCount_ = 0;
    std::uint64_t epoch_ = 0;
    std::size_t outstanding_ = 0;

    // MUST remain the last member. doneSignal_.notify_one() is issued after the
    // lock is released, so a worker can be preempted between its decrement and
    // that notify while the caller has already returned and the owner has begun
    // destroying the executor. That is safe only because members are destroyed
    // in reverse declaration order: ~vector<jthread> requests stop and joins
    // every worker BEFORE doneSignal_, startSignal_ and mutex_ are destroyed.
    // Moving this member earlier is silent undefined behaviour, not a warning.
    std::vector<std::jthread> workers_;
};

// Requires threads >= 1. The `--threads 0` sentinel is resolved in the CLI, so
// the core library carries no notion of "auto".
std::unique_ptr<Executor> makeExecutor(std::size_t threads);

}  // namespace vrp
