
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace fulei
{

    template <typename T>
    class ThreadSafeQueue
    {

    public:
        ThreadSafeQueue() = default;
        ThreadSafeQueue(const ThreadSafeQueue &) = delete;
        ThreadSafeQueue &operator=(const ThreadSafeQueue &) = delete;
        ThreadSafeQueue(ThreadSafeQueue &&) = delete;
        ThreadSafeQueue &operator=(ThreadSafeQueue &&) = delete;

        /**
         * 插入一个元素
         */
        void push(const T &item)
        {
            std::lock_guard<std::mutex> guard(mtx_);
            task_queue_.push(item);
            not_empty_.notify_one();
        }

        /**
         * 获取一组元素
         *  max_batch_size: 最大返回数量 (为0则不做限制, 有多少返回多少)
         *  timeout_ms: 超时时间 (为0则一直阻塞,直到有数据返回)
         * 达到最大容量或者超时才会触发返回结果,否则处于阻塞状态
         */
        std::vector<T> pop(int max_batch_size = 0, int timeout_ms = 0)
        {
            std::vector<T> results;
            std::unique_lock<std::mutex> lock(mtx_);
            // 1. 直接先取数据
            auto start_time = std::chrono::steady_clock::now();
            while (!task_queue_.empty() && (max_batch_size == 0 || results.size() < max_batch_size))
            {
                results.emplace_back(std::move(task_queue_.front()));
                task_queue_.pop();
            }
            if (timeout_ms == 0 && !results.empty())
            {
                return results;
            }

            // 2. 数据没有达到预期或者数据没有(上面已经对没有的情况判断了)
            while (max_batch_size == 0 || results.size() < max_batch_size)
            {
                if (timeout_ms)
                {
                    auto duration = std::chrono::steady_clock::now() - start_time;
                    auto wait_time = std::chrono::milliseconds(timeout_ms) - std::chrono::duration_cast<std::chrono::milliseconds>(duration);
                    // 等待时间计算
                    if (wait_time.count() <= 0)
                    {
                        break;
                    }
                    not_empty_.wait_for(lock, wait_time, [this]
                                        { return !this->task_queue_.empty(); });
                }
                else
                {
                    not_empty_.wait(lock, [this]
                                    { return !this->task_queue_.empty(); });
                }
                while (!task_queue_.empty() && (max_batch_size == 0 || results.size() < max_batch_size))
                {
                    results.emplace_back(std::move(task_queue_.front()));
                    task_queue_.pop();
                }
                if (timeout_ms == 0)
                {
                    break;
                }
            }
            return results;
        }

    private:
        std::queue<T> task_queue_;
        std::mutex mtx_;
        std::condition_variable not_empty_;
    };

}
