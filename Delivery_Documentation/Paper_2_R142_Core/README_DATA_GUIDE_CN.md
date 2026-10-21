# 两篇论文的材料与数据对应说明（2026-10-18）

## 两篇的身份
第一篇是 R51 的固定目标模型后的决策保留、复核、修复和重算；第二篇是 R142 的更新选择、任务要求、验证及已迁入的人工纠正/后续学习。每篇自身都有正文和 SI。此前“单篇整合版”仍保留原档，但不是本次两篇材料的两个阅读视图。

## 状态
| 材料 | 当前状态 |
|---|---|
| 第一篇历史正文与 SI 的 LaTeX | 已恢复全部 751 / 5,480 行原始可读文本 |
| 第一篇 2026-10-15 SA 原始 ZIP 和 12 个原图 | 导出失败 / 未恢复；第一篇未完成全稿编译 |
| 第一篇第三稿迁入模块 | 已保存，但尚未应用到完整当前第一篇；5 页 PDF 仅为模块阅读稿 |
| 第二篇完整原生 LaTeX、图表和 PDF | 已保存；合并入口本轮重新编译通过，逐页文本保持相同 |
| R142 原始研究包 | 312,100,206 字节原包已取回；本次数据包保留原研究输出/代码/协议/已保存权重 |
| 七个原数据运输卷 | 全部取回，按第一/第二篇主要用途分装，原目录及文件字节不变 |
| 小型神经记录、五个种子检查点 | 原记录包及两份大体积检查点包均已取回 |
| 后续30选择、24轮及严格匹配审计文件 | 仍有明确缺件，见第二篇 README；没有用新计算补造 |

## 对应关系
| 论文中的内容 | 数据路径（相对所有数据包共同解压的父目录） | 解释 |
|---|---|---|
| R51 已验证决策、模型更新与工作流记录 | `Research_Data/Data_Code/benchmarks/` | 保存的任务、数据及代码；不承诺等同于所有原图的专用绘图源 |
| R51 依赖成本、独立/联合求解与规模检查 | `Research_Data/Data_Code/choice_and_scale/` | 既有合成/受控设置原样标注；不作为企业实测成本 |
| R51 配对修复、模型及人类复核记录 | `Research_Data/Additional_Evidence/evidence/` | 共享历史证据，保留原分母和来源 |
| R142 候选选择与审计 | `Research_Data/R142_RESEARCH_REVISION/results/experimental/`、`results/control_analysis/` | 保留所有可取得批次和逐条结果，不按有利与否过滤 |
| R142 兼容训练/修复对照 | `Research_Data/R142_RESEARCH_REVISION/results/compatibility/` | 含已保存适配器权重与完整对照记录 |
| R142 数学检查、协议和分析程序 | `Research_Data/R142_RESEARCH_REVISION/results/theory_simulations/`、`reproducibility/` | 旧计算/程序；本轮未执行新仿真 |
| 固定特征、反馈复用、跨代拟合 | `Research_Data/Data_Code/experiments/`、`continued_learning/` | 原来源、权重、样本身份和输出数组保留 |
| PRM800K 原始/派生与其他证据 | `Research_Data/Data_Code/validation_extensions/` | 真实原始数据与受控设计分别保留，不把合成条件写成观测事实 |
| 五种子神经实验 | `Research_Data/Neural_Mathematics/Neural_Mathematics_Bridge_20261007/` | 小型记录与检查点包按同一路径合并 |
| 当前稿队列计算 | `Research_Data/Integrated_Human_Evidence/queue_simulation_data/` | 已保存的受控队列输出和原程序 |
| 其他固定特征核查/证明支持 | `Research_Data/Human_Evidence_Audit/`、`Additional_Human_Evidence/` | 部分历史组件需要原主树；只保证列出的文件已交付，不能把本轮打包称为全部研究复现 |

## 解压和使用
两篇的数据 ZIP 都从 `Research_Data/` 起放置。选择同一个父目录解压，目录会合并、不会把一份文件变成一项新实验。原七卷的全部文件已保留在这套按论文分装的数据中。通用 `Data_Code/reproduce.py` 是全量研究入口，使用它需要把两篇所有分装数据放在一起；仅查看某一结果可进入上表相应目录。

两个神经检查点原包例外：解压位置为上述父目录下的 `Research_Data/Neural_Mathematics/`。它们原包内已经包含 `Neural_Mathematics_Bridge_20261007/` 根目录。

每个包都有本包文件清单/校验信息；总文件索引注明文件来自哪个原始 ZIP。核验是对字节、源依赖和 LaTeX 构建的核对，不是重新训练、仿真、全量数学审计或原创性认证。检查点属于原研究的模型资产，不含字体二进制文件。

## 不做的替换
不把一篇的正文和 SI 当作两篇；不把 5 页模块稿当作第一篇全文；不把恢复的历史 `.tex` 冒充已恢复的 SA 原始 ZIP；不从图反推原始数据；不把缺失的30次/24轮记录换成新跑结果；不将两篇共用的历史样本重复算成独立证据。
