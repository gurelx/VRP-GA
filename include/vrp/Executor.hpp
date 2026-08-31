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

// Receives a half-open range [begin, end), so callers can hoist scratch
// allocations out of the per-item loop.
using ParallelBody = std::function<void(std::size_t begin, std::size_t end)>;

// Splits [0, n) into `parts` disjoint chunks covering every index exactly once
// and returns chunk `index`. Deterministic in (n, parts, index), which is what
// lets a threaded run reproduce a serial one.
// Preconditions: parts >= 1 and index < parts.
std::pair<std::size_t, std::size_t> chunkRange(std::size_t n, std::size_t parts,
                                               std::size_t index) noexcept;

class Executor {
public:
    virtual ~Executor() = default;
    virtual std::size_t threadCount() const noexcept = 0;

    // Splits [0, n) across the executor's threads, invoking `body` once per
    // non-empty chunk and returning once every chunk has completed; n == 0
    // invokes `body` zero times. Not reentrant and not concurrency-safe: one
    // owning thread issues the calls, and the executor must outlive them. A
    // throw from the caller's own chunk propagates after the pool is drained;
    // a throw from a worker chunk terminates the process.
    virtual void parallelFor(std::size_t n, const ParallelBody& body) = 0;
};

class SerialExecutor final : public Executor {
public:
    std::size_t threadCount() const noexcept override { return 1; }
    void parallelFor(std::size_t n, const ParallelBody& body) override;
};

class ThreadPoolExecutor final : public Executor {
public:
    // Requires threads >= 1; threads == 1 runs every chunk on the calling
    // thread with no workers.
    explicit ThreadPoolExecutor(std::size_t threads);
    ~ThreadPoolExecutor() override;

    ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
    ThreadPoolExecutor& operator=(const ThreadPoolExecutor&) = delete;

    std::size_t threadCount() const noexcept override { return threads_; }
    void parallelFor(std::size_t n, const ParallelBody& body) override;

private:
    void workerLoop(std::stop_token stop, std::size_t index);

    // Drains the current epoch and releases body_. noexcept because it also
    // runs while an exception from the caller's own chunk is propagating.
    void waitForWorkers() noexcept;

    // const: workers read threads_ without synchronisation, which is safe only
    // because it is never written after the constructor publishes it.
    const std::size_t threads_;
    std::mutex mutex_;
    std::condition_variable_any startSignal_;
    std::condition_variable doneSignal_;
    const ParallelBody* body_ = nullptr;
    std::size_t itemCount_ = 0;
    std::uint64_t epoch_ = 0;
    std::size_t outstanding_ = 0;

    // MUST remain the last member: a worker can still be between its decrement
    // and its notify when the owner starts destroying the executor, and only
    // reverse declaration order joins the workers before doneSignal_,
    // startSignal_ and mutex_ die. Moving it earlier is silent UB.
    std::vector<std::jthread> workers_;
};

// Requires threads >= 1; the `--threads 0` sentinel is resolved in the CLI.
std::unique_ptr<Executor> makeExecutor(std::size_t threads);

}  // namespace vrp
