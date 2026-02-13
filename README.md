# ⚡ GPU Puzzles: Solutions & Notes

Huge thanks to the author of [GPU Puzzles](https://github.com/srush/GPU-Puzzles). Unlike [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles), which focuses on one-line algorithmic logic, **GPU Puzzles** feels much more like a hands-on CUDA programming exercise. The Colab environment is convenient, and the visualization tools showing data flow **make it intuitive to understand** the CUDA Memory Model.

非常感谢 [GPU Puzzles](https://github.com/srush/GPU-Puzzles) 的作者。与侧重“一行代码逻辑解谜”的 [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) 不同，**GPU Puzzles** 更偏向 Hands-on 的 CUDA 编程实战。Colab 环境非常方便，特别是那个指示数据流向的可视化图表，让理解 **CUDA 内存模型**（尤其是 Global 与 Shared Memory 的交互）变得非常直观。

---

## 💡 Core Concepts / 核心体会

I highly recommend reading **PMPP** (*Programming Massively Parallel Processors*) alongside these puzzles to grasp the underlying hardware concepts. Initially, I was confused: *Why does each thread handle only one point? Shouldn't we use loops and strides?*

建议配合 **PMPP** (*Programming Massively Parallel Processors*) 阅读以获得概念性的支持。起初我很疑惑：“为什么每个 Thread 只处理一个点，而不用循环 Stride？”

The book explains that **GPU threads are extremely lightweight**. Unlike CPU threads, creating thousands of them is cheap. Apart from the warps actually executing on the SM, the queued threads have virtually no overhead. It’s less about "looping" and more about "massive parallelism."

书中解释道，**GPU 线程极其廉价 (Lightweight)**。与 CPU 线程不同，创建成千上万个 GPU 线程成本很低。除了正在 SM 上执行的 Warp，排队中的线程几乎没有额外开销。GPU 的设计哲学不是“循环处理”，而是“海量线程并行”。

---

## 📝 Key Takeaways / 关键笔记

### Puzzle 11: Conv 1D (Memory Load Balancing)
The key insight here is **Load Balancing** when moving data to Shared Memory. By carefully distributing the reading tasks among threads, we reduced the maximum global reads per thread (e.g., from 3 to 2). This optimization effectively helps **hide memory latency**.

这里的关键在于搬运数据到 Shared Memory 时的**负载均衡**。通过合理分配读取任务，我们将每个线程的最大 Global Read 次数降低（例如从 3 次降到 2 次）。这个优化能有效帮助**掩盖内存延迟**。

### Puzzle 12: Prefix Sum (Blelloch Algorithm)
This involves the **Blelloch Scan** algorithm. A great way to understand the `Downsweep` phase is to view a node's value as **"the prefix sum of the range preceding this node's jurisdiction."** By processing layer by layer down to the leaves, we ensure every operation maintains this property. Once the leaf nodes are reached, the prefix sum for every position naturally emerges.

涉及 **Blelloch 算法**。理解 `Downsweep` 过程的一个直观视角是：把当前 Node 的值理解为**“该 Node 管辖范围之前的 Prefix Sum”**。通过逐层处理直到叶子节点，每一次操作都在维护这个性质，当所有叶子节点处理完毕后，自然就得到了每个位置的正确 Prefix Sum。

### Puzzle 13: Axis Sum (Grid Mapping Strategy)
In this puzzle, `blockIdx.y` is mapped to the batch dimension. This reflects a strategy where `Global_Y` represents the flattened outer dimensions (e.g., `Batch * Seq` in a `(Batch, Seq, Hidden)` shape). By setting `Global_Y = blockIdx.y`, the Grid Y-axis handles independent rows naturally.

本题将 `blockIdx.y` 映射到了 Batch 维度。这反映了一种策略：用 `Global_Y` 来表示 Flatten 后的前置维度（例如 `(Batch, Seq, Hidden)` 中的 `Batch * Seq`）。通过 `Global_Y = blockIdx.y`，让 Grid 的 Y 轴自然地处理相互独立的行。
