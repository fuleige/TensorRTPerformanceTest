
#include <iostream>
#include "utils/ThreadSafeQueue.h"
#include <thread>

int main() {
    
    fulei::ThreadSafeQueue<int> queue;

    std::thread t([&](){
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        for(int i=0; i<128; i++) {
            queue.push(i);
        }
    });

    for(int i=0; i< 10; i++) {
        auto data = queue.pop(0, 0);
        std::cout << data.size() << std::endl;
    }
    return 0;
}