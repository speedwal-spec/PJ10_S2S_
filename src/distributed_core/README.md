# 面向多云异构算力的星型异步联邦调度引擎 (Star-Federated Engine)

## 1. 设计概述
本模块 (`distributed_core`) 旨在解决大规模消融实验中单机算力受限的问题。通过构建一个**星型拓扑 (Star Topology)** 的分布式网络，系统将本地主控节点 (Master) 与分布在 Kaggle、Hugging Face Spaces、ModelScope 等平台的异构计算节点 (Workers) 连接起来，实现任务的并联调度与结果的自动化汇总。

## 2. 核心架构组件

### 2.1 StarDispatcher (星型主控节点)
*   **职责**：系统的“大脑”。负责解析消融实验矩阵，维护全局任务队列，并监控各云端节点的健康状态。
*   **关键特性**：
    *   **负载均衡**：根据各 Worker 上报的显存与算力规格，动态分配实验任务。
    *   **Local Fallback**：在检测到环境一致性要求极高时，自动降级为本地流水线执行。

### 2.2 HeterogeneousCloudWorker (异构边缘节点)
*   **职责**：系统的“手脚”。每个 Worker 代表一个云端的计算实例。
*   **异构适配**：通过抽象类设计，抹平了不同云平台（如 GPU 型号、操作系统）的差异。
*   **日志流式传输**：预留了 `stream_logs` 接口，支持将远程训练日志实时回传至本地终端。

### 2.3 TokenPoolManager (凭证安全池)
*   **职责**：管理多平台的 API Key。
*   **安全机制**：
    *   **自动轮换 (Rotation)**：防止单一 Token 因高频调用触发平台风控。
    *   **探活机制**：在任务下发前验证 Token 的有效性。

### 2.4 Message Queue (异步消息总线)
*   **TaskQueue**：采用生产者-消费者模式，解耦任务分发与执行过程。
*   **ResultCollector**：负责接收各节点回传的 ROUGE 指标与模型权重路径，并生成全局总结报告。

## 3. 数据流与工作流程

1.  **任务注册**：用户通过 `dispatch_ablation_matrix` 传入一组超参数配置。
2.  **封装下发**：Master 将配置封装为 `TaskPacket`，并根据当前负载选择最优的 Worker 节点。
3.  **异步执行**：Worker 节点接收任务，在云端容器内启动 `train.py`。
4.  **结果回传**：训练结束后，Worker 调用 `ResultCollector` 将评测结果上传至共享存储。
5.  **全局汇总**：Master 节点更新全局进度条，并在所有任务完成后生成对比图表。

## 4. 当前状态与未来规划

*   **当前状态**：**Framework Defined (框架已定义)**。目前模块处于演示模式，核心逻辑已通过 `Local Pipeline` 实现闭环，确保了课程实验数据的绝对一致性。
*   **下一步计划**：
    *   接入 `gRPC` 实现跨节点的低延迟通信。
    *   引入 `Ray` 或 `Celery` 框架处理大规模并发任务。
    *   实现基于 Kubernetes 的 Worker 节点自动扩缩容。

## 5. 模块结构说明

| 文件 | 核心类/功能 |
| :--- | :--- |
| `cloud_dispatcher.py` | `StarDispatcher`, `HeterogeneousCloudWorker` |
| `message_queue.py` | `TaskQueue`, `ResultCollector`, `TaskPacket` |

---
*注：本模块设计遵循高内聚、低耦合原则，预留了完整的 RPC 接口，可随时从“本地降级模式”切换至“真实联邦模式”。*
