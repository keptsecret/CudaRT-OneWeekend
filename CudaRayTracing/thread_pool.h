#pragma once

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <vector>
#include <iostream>

class thread_pool {
public:
	thread_pool() :
			num_workers(std::thread::hardware_concurrency()), jobs_completed(0) {
	}

	thread_pool(unsigned int nw) :
			num_workers(nw), jobs_completed(0) {
	}

	void start();
	void stop();

	template<class F, class... Args>
	auto queue_job(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

	bool busy();
	unsigned int num_jobs_queued() const { return jobs.size(); };
	unsigned int num_jobs_completed() const { return jobs_completed; }
	void reset_num_jobs_completed() { jobs_completed = 0; }

private:
	void thread_loop();

	bool should_terminate = false;
	unsigned int num_workers;
	unsigned int jobs_completed;

	std::mutex queue_mutex;
	std::mutex count_mutex;
	std::condition_variable mutex_condition;
	std::vector<std::thread> threads;
	std::queue<std::function<void()>> jobs;
};

inline void thread_pool::start() {
	threads.resize(num_workers);
	std::cout << "Starting " << num_workers << " workers..." << std::endl;
	for (unsigned int i = 0; i < num_workers; i++) {
		threads.emplace_back([this] { this->thread_loop(); });
	}
}

inline void thread_pool::stop() {
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		should_terminate = true;
	}

	mutex_condition.notify_all();
	for (std::thread& active_thread : threads) {
		try {
			if (active_thread.joinable())
				active_thread.join();
		} catch (const std::system_error& e) {
			std::cerr << e.code() << ": " << e.what() << ".\n";
			std::cerr << "Thread may have already terminated.\n";
		}
	}
}

template <class F, class... Args>
inline auto thread_pool::queue_job(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
	using return_type =	typename std::result_of<F(Args...)>::type;

	auto job = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
	std::future<return_type> result = job->get_future();
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		jobs.emplace([job]() { (*job)(); });
	}
	mutex_condition.notify_one();
	return result;
}

inline bool thread_pool::busy() {
	bool pool_busy = true;
	{
		std::unique_lock<std::mutex> lock(queue_mutex);
		pool_busy = !jobs.empty();
	}
	return pool_busy;
}

inline void thread_pool::thread_loop() {
	while (true) {
		std::function<void()> job;
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			mutex_condition.wait(lock, [this] {
				return should_terminate || !jobs.empty();
			});
			if (should_terminate && jobs.empty())
				return;
			job = std::move(jobs.front());
			jobs.pop();
		}

		job();
		{
			std::unique_lock<std::mutex> lock(count_mutex);
			jobs_completed++;
		}
	}
}
