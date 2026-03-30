Zen架构的L3虽然逻辑上是一个整体，但区别于传统组相联缓存的L3，地址A的数据不会被固定映射在某个特定的L3区块。
根据PPR文档(PPR-1-p131），Zen 5架构的L3是non‑inclusive的，与之相对应的是inclusive：规定高级别缓存（如 L3）必须包含低级别缓存（如 L2 和 L1）中的所有数据。如果数据存在于 L1 中，它也必然备份在 L2 和 L3 中。即L3 Cache不一定包含L1/L2中的数据。
[AMD ZEN Architecture PDF](https://www.scribd.com/document/394144659/AMD-ZEN-Architecture-pdf)提到L3为`victim cache`，因此从其他CCX的L3或内存加载数据到本地CCX时，会先加载到L1/L2 Cache。L3只会存放从本地CCX的L2 Cache中驱逐出的数据，不会跨越CCX边界直接缓存其他CCX核心L2驱逐的数据。
当数据缓存在其他CCX的L3时，会从远程CCX的L3经过IF链路加载到本地的L1/L2缓存。若数据不在任何L3中，则通过内存控制器从内存读取，也是先加载到本地L1/L2。