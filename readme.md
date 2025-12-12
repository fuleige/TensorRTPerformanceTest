
### 库安装

```bash
sudo apt-get install tensorrt-dev
sudo apt-get install tensorrt-libs
```

### 推理性能加速细节

#### 多线程多流推理

- cuda本身是利用多条流来实现多任务高效并发 (tensorRT当然也符合这种情况)
- 为了编程的简易性, 不同线程独有的数据利用thread_local进行维护实现
- 每个线程必须独有的数据
    - cuda流 (cudaStream_t)
    - 执行上下文 (nvinfer1::IExecutionContext)
    - 预分配的输入输出的数据指针 (包括主机和设备上的数据指针)

#### 内存预分配重复利用

在主机上进行测试, 考虑如下场景, yolo的输入数据:
- 输入数据大小批大小为64, 输入图像是(640,640,3), 数据类型为float32
- 内存块大小为: $bacth\_data\_memory\_size=64\times640\times640\times3\times4=300MB$

```C++
int batch_data_memory_size = 64 * 640 * 640 * 4;

// 在Linux机器上,这条指令几乎不耗什么时间,因为它没有真正申请内存
// 只有第一次触发读写才真正申请内存
char *data = new char[batch_data_memory_size];

// 可以用 memset 或者 memcpy 来触发写操作
// 第一次写, 实测耗时约 192ms
memset(data, 0, batch_data_memory_size);

// 再次执行 memset 或者 memcpy 来触发写操作
// 之后的耗时约 18ms
memset(data, 0, batch_data_memory_size);
```

> 重复触发每次从操作系统申请并且第一次写,会浪费很多不必要的时间

#### 主机分配锁页内存

> 使用 `cudaMallocHost` 而不是 `malloc` 分配主机内存,主要用于 `cudaMemcpy` 这种方法传入主机内存时使用

- cudaMallocHost 不是分配出的GPU上的显存,而是主机上的内存,一种特殊的"锁页"内存,操作系统不会将它移走,它对应的内存物理地址是不变的
- 然而它更快的原因,是因为GPU是通过 DMA来进行数据传输一些机制问题
    - DMA必须知道数据的物理地址, 操作系统一般对应用程序分配是"虚拟地址",且操作系统可能会将内存块置换到磁盘中
    - 为了防止操作系统这种问题, cuda驱动也自己会临时分配一块锁页内存,先讲数据拷贝到这块临时内存,再讲这个临时锁页内存拷贝到GPU显存中
    - 而我们申请内存时直接使用 `cudaMallocHost`直接分配锁页内存, 减少了一次内存拷贝时间

