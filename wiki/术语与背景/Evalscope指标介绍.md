# Evalscope指标介绍

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| Time taken for tests (s) | 一次压测从发起到结束的总耗时。 | 越短越好，但要结合并发规模和请求数一起看。 |
| Number of concurrency | 并发请求数。 | 越高表示同时在飞的请求越多。 |
| Request rate (req/s) | 客户端目标请求速率。该实验为无限速率模式，通常记为 -1。 | 这里只是压测配置，不是模型实际能力。 |
| Total requests | 本次压测发送的总请求数。 | 应与并发和实验设置一致。 |
| Succeed requests | 成功返回的请求数。 | 越接近总请求数越好。 |
| Failed requests | 失败请求数。 | 理想值为 0。 |
| Output token throughput (tok/s) | 仅统计输出 token 的吞吐率。 | 越高表示 decode 吞吐越强。 |
| Total token throughput (tok/s) | 输入 token 与输出 token 合计的吞吐率。 | 越高越好，更接近端到端 token 搬运能力。 |
| Request throughput (req/s) | 单位时间内完成的请求数。 | 越高越好。 |
| Average latency (s) | 单请求端到端平均延迟。 | 越低越好。 |
| Average time to first token (s) | 平均首 token 时延，等价于 prefill + 首次返回等待。 | 越低越好，对交互体验影响很大。 |
| Average time per output token (s) | 平均每个输出 token 的耗时。 | 越低越好，反映 decode 阶段速度。 |
| Average inter-token latency (s) | 平均相邻两个输出 token 之间的间隔。 | 通常与 TPOT 接近，越低越好。 |
| Average input tokens per request | 单请求平均输入 token 数。 | 该实验中用于核对负载是否一致。 |
| Average output tokens per request | 单请求平均输出 token 数。 | 该实验中用于核对负载是否一致。 |
| P99 latency (s) | 端到端延迟的 P99。 | 越低越好，用于观察尾延迟。 |
| P99 TTFT (s) | 首 token 时延的 P99。 | 越低越好，反映最慢那批请求的 prefill 体验。 |
| P99 TPOT (s) | 每输出 token 耗时的 P99。 | 越低越好，反映 decode 阶段的尾部抖动。 |
