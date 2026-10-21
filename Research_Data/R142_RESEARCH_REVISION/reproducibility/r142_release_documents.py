"""Write a current, truthful response and author-facing assessment."""
from pathlib import Path
import re,json,csv
ROOT=Path(__file__).resolve().parents[1];MAN=ROOT/'manuscript'
def main():
    response=(MAN/'R141_RESPONSE_TO_COMMENTS.tex').read_text().replace('R141','R142')
    response=response.replace('Extended Data 1--4','Extended Data 1--10')
    response=response.replace('One frozen R142 protocol identifies the candidate count, grids, floors, families, costs and algorithmic sequence.','The original frozen selection protocol and the separately frozen R142 compatibility protocol identify their candidate sets, costs and algorithmic sequences.')
    response=response.replace('R142 repeats complete incumbent training, candidate construction, calibration, selection and audit in four blocks for each of two model families.', 'The R141 selection experiment retained in R142 includes complete incumbent training, candidate construction, calibration, selection and audit in four blocks for each model family.')
    a=response.index('The Introduction and Supplement identify MUSCLE',response.index('A15:'));b=response.index(r'\subsection*{A16:',a)
    response=response[:a]+r'''R142 implements MUSCLE's published student-error mask and KL objective on every one of the eight frozen full-update candidates, with a paired extra cross-entropy epoch from the same initialization. All sixteen trained adapters, token outputs, checker results, costs and hashes are recovered. The loss is faithfully specified; the adapter rank, target layers, duration and tasks follow our resource setting rather than reproduce the original paper's benchmark. The audit pools were previously inspected. This closes a specific missing learning-method comparison, while comparisons with every neighbouring method and independent cross-domain confirmation remain open.

\textbf{Location:} Algorithm 4; Figure 5d--f; Extended Data 9; Supplement: compatibility comparison.

\textbf{Status:} Published-loss comparison completed under disclosed settings; broad superiority is not established.

'''+response[b:]
    latest=r'''\section*{The five latest requests}
\subsection*{R1: The manuscript still seems distant from Nature or Science.}
We agree that the remaining distance is substantive. The question is important, but many mathematical components are applications of established results and the strongest current generated-output evidence uses two small-model families and previously studied BFCL pools. The revision defines the contribution and its limitations directly. It does not promise editorial review or acceptance, or present additional simulations as independent deployment evidence.

\textbf{Action:} a separate author-facing assessment distinguishes scientific importance, established mathematics, original observations, executable procedures and the evidence still needed for a broad claim. The title and Discussion connect task requirements to evidence and deployment work.

\subsection*{R2: Methods contains prose but no algorithms.}
Methods now contains four numbered, linked procedures with inputs, outputs, steps and failure cases: fixed-grid update selection; robust repair with a checked dual witness; exact finite-record evaluation and minimum-cut work planning; and the published compatibility-loss experiment with a matched continuation control. Executable Python accompanies them. Fixed-price minimum cut is not claimed to solve uncertain adaptive testing, training and repeated adoption jointly.

\textbf{Verification:} 24,562 exhaustive bounded-tape cases, 600 minimum-cut comparisons with exhaustive optimization, and 360 synthetic workflow replays all satisfy their stated equality checks. Proofs, assumptions and complexity are given in the Supplement.

\subsection*{R3: Expand Extended Data and distinguish lines and bars.}
Extended Data grows from four displays to ten while retaining all sixteen earlier panels. The additions show every candidate--strength outcome, declarations versus all losses, calibration and benefit sensitivity, observation/token/time costs, all eight compatibility comparisons, and deployment checks. Bars use colours and hatches; lines use colours, styles and markers. Main Figures 1 and 5 gain controlled-execution and training-comparison panels. Vector PDF/SVG and high-resolution PNG files are supplied.

\subsection*{R4: Link every result and display to its appendix support and back.}
Main Table 1 provides a result-to-proof and implementation map. Every supplementary section and formal result has a return link. Supplementary tables and main/Extended Data captions link to their relevant results and supporting protocols. A complete supporting index and machine-readable link map accompany the files. The combined PDF uses internal links; standalone Main and Supplement use relative links when kept together. A PDF link audit checks destination resolution, including cross-document links.

\subsection*{R5: Do the substantive work and judge research quality.}
The revision adds sixteen actually trained compatibility/control adapters across all eight batches, evaluates all 800 old-record appearances rather than only incumbent successes, and supplies all paired transitions. Mean protected-family loss is 4.69\% with MUSCLE loss, 7.09\% with ordinary continuation and 8.92\% for the jointly reevaluated full update. All three meet the experimental new-benefit floors in all eight batches. The comparison is supplementary because the audit pools were previously used, and the original selection experiment remains a distinct record.

A targeted same-weight check also reproduces one discrepancy: complete-pool evaluation loses 1/72 protected Qwen tasks, protected-only batches lose 3/72, and repeating the complete pool returns to 1/72 with identical tokens. The code, weights, device and evaluator are fixed. This establishes batch-composition sensitivity for that case, not a universal floating-point mechanism or an explanation of every historical mismatch.

\textbf{Remaining scientific boundary:} a broadly predictive mechanism, independent new task distributions, stronger comparisons under matched total resources, and consequential multi-release or organizational outcomes are still required for the intended foundational claim. This revision makes the work more complete and testable; it does not establish that it exceeds comparable Nature or Science papers.

'''
    response=response.replace(r'\section*{Daniel}',latest+r'\section*{Daniel}')
    response=response.replace('faithful comparisons with every neighbouring learning algorithm, independent semantic annotation', 'comparisons with every neighbouring learning algorithm, independent semantic annotation')
    (MAN/'R142_RESPONSE_TO_COMMENTS.tex').write_text(response)
    letter=(MAN/'R141_COVER_LETTER_DRAFT.tex').read_text().replace('R141','R142').replace('Evaluating language-model updates under competing task requirements','Evaluating language-model updates under task and deployment constraints')
    letter=letter.replace('The manuscript connects three parts of this decision.', 'The manuscript connects task requirements, evidence and deployment work.')
    letter=letter.replace('These findings identify a mismatch between controlling false selective assurances and minimizing total task loss, rather than claiming a universal safe-update algorithm.', 'A supplementary eight-batch comparison implements the published MUSCLE loss with a paired continuation control, and a same-weight control identifies sensitivity to evaluation batch composition. Fixed-record stopping and a classical minimum-cut reduction make a restricted deployment-work problem executable. These results identify where assurances and operational outcomes differ, with explicit conditions for each claim.')
    (MAN/'R142_COVER_LETTER_DRAFT.tex').write_text(letter)
    report='''# R142 修改说明与 Nature/Science 投稿判断

我的判断：问题重要，也适合严肃研究；R142 比上一版更完整、可复现、容易审查，但现有证据仍不足以支持“已经达到 Nature/Science 主刊所需的开创性和广泛影响力”。我不建议把这轮增加的公式、算法和图本身当作已经跨过这一门槛的依据，也无法可靠估计送审或录取概率。

本轮读入的 42 页 PDF 与保留的历史部署工作稿逐字节相同，题为 *Generative AI model changes need not determine deployment work*。它并非 R141。新稿以 R141 已核验的任务结果为基础，接回历史稿中的部署、证据复用和共享工作问题；历史未核验的成本数字仍保留其原始证据身份。

## 已完成的实质工作

| 要求 | R142 已落实的内容 | 证据边界 |
|---|---|---|
| Methods 有明确算法 | 四个编号算法，包含输入、输出、停止或失败分支及对应 Python | 校准、线性规划和最小割分别注明已有数学来源 |
| 理论与部署相连 | 固定测试记录的精确停止证明；有固定证书价格的最小割规划证明；完整依赖条件下的复用证明 | 不声称解决了未知证书获取、模型训练、连续采纳的联合最优问题 |
| 数值验证 | 24,562 个穷举测试案例；600 个优化问题与穷举最优值一致；360 个工作流复现全部完整决策及产物 | 全部明确标为人工设定的算法验证 |
| 更强学习方法对照 | 八批 × MUSCLE 损失与普通继续训练，共 16 个实际训练适配器；取回 200 项有校验值的原始产物 | 复现已发表损失，在本研究资源设置下运行；不是原论文整套基准复现 |
| 完整任务转移 | 重测每批全部 100 条旧任务，共 800 个行出现次数；报告失去正确、获得正确、保持正确、保持错误 | 同一基准池会跨批重复，不能当作 800 个独立部署案例 |
| 执行条件控制 | 相同权重、相同设备的批次组成检查，另取回 10 项原始产物 | 只确定一个 Qwen 案例的批次敏感性 |
| 图与支持材料 | 主图仍为 5 张；图 1、图 5 增加面板；Extended Data 从 4 组增到 10 组；所有原有面板保留 | 颜色之外增加线型、标记或纹理 |
| 双向引用 | 主文阅读索引、补充章节与结果索引、正文↔补充链接、内部链接合并版，以及可检查的引用表 | 分立 PDF 需放在同一文件夹；合并版可直接内部跳转 |

## 新实验改变了什么判断

原先六种选择规则的完整实验保留：在八批均达到新能力要求的规则中，直接验证的平均旧任务族损失为 2.64%，局部预测为 4.22%，校准预测为 8.33%，完整更新为 9.81%。这意味着校准声明本身不等于整个应用的损失被压低。不能把“选择性保证成立”写成“总体部署效果最佳”。

新增学习对照在同一批次内对四个模型联合评估，结果如下。这里的完整更新是本轮联合复测的基线，不能与上面原始记录的 9.81% 混用。

| 方法 | 平均旧任务族损失 | 平均旧任务总体正确率 | 平均新任务正确率 | 通过新能力要求 |
|---|---:|---:|---:|---:|
| 完整更新联合复测 | 8.92% | 69.00% | 55.83% | 8/8 |
| MUSCLE 损失继续训练 | 4.69% | 72.13% | 53.13% | 8/8 |
| 普通交叉熵继续训练 | 7.09% | 71.25% | 57.50% | 8/8 |

MUSCLE 损失相对完整更新在六批减少旧任务族损失、两批增加；相对普通继续训练则五批更低、一批相同、两批更高。它提供了有用的对照，但平均改善也伴随新任务正确率的变化，不能声称全面占优。MUSCLE 是已有研究，本论文需要明确自己增加了什么。[正式发表的 MUSCLE 论文](https://aclanthology.org/2024.findings-emnlp.430/)。

本轮全部旧任务评估还暴露出重要的可复现性细节：三批 Qwen 的旧损失计数与原先只生成受保护样本的审计略有差异。八批 incumbent 的每条旧任务 token 均复现，全部完整更新的新任务计数均复现。对 Qwen 第一批的控制实验表明，完整旧样本批次得到 1/72 损失，只放受保护样本的批次得到 3/72；恢复完整批次后再次得到 1/72，所有 token 相同。这证明了这一实例中批次组成会影响结果；尚未定位具体数值内核，也不能将其他所有差异一并归因于此。

## 距离 Nature/Science 的实质差距

研究问题的重要性是成立的。Nature 已发表关于局部微调产生广泛行为变化的研究；模型更新影响既有用途是值得跨学科读者关心的问题。该研究通过一组干预与控制实验界定了跨任务现象，而不是仅指出平均性能不足。[Nature 原文](https://www.nature.com/articles/s41586-025-09937-5)。

当前最突出的差距有三项。

1. **核心原创结论还需更集中。** 方向、边界距离、选择性声明、修复约束、证据成本与共享依赖目前形成了一个可执行分析框架。许多理论环节是共形预测、稳健优化、序贯检验和网络流的已有工具在明确子问题上的应用。应避免把这些已知工具的组合直接称为新的普遍定律。机器学习的发布条件与证据成本也已有连续集成研究。[ease.ml/ci 正式论文](https://proceedings.mlsys.org/paper_files/paper/2019/hash/4284d31e68c0a4a39dcdad167ac4bd72-Abstract.html)。
2. **跨情境证据仍窄。** 主要新输出实验集中于两类小型开放模型、单轮函数调用和已使用过的任务池。补做学习对照很有价值，但看过的审计集仍是看过的审计集。历史 stateful 结果还有重复基础任务和必要回答缺失的限制。
3. **部署影响尚未直接测量。** 最小割和工作流实验证明了限定问题的算法正确性，并展示共享依赖为什么重要。它们没有量到真实组织的复核工时、停机、业务损失或多次更新中的长期收益。经典共享固定成本理论已有对应基础。[Rhys](https://doi.org/10.1287/mnsc.17.3.200)、[Picard](https://doi.org/10.1287/mnsc.22.11.1268)。

## 哪种下一步最可能带来质变

最值得集中检验的问题是：**在必须保留新能力的情况下，能否事先判断一次更新需要多少重新验证和下游重做，并据此选择一个在完整成本与任务损失上都有优势的更新或修复方案？** 这把前瞻预测、干预和实际部署后果放在同一个可以被证伪的问题中。

相应证据应分三步取得：先在尚未分析过的新任务分布和多个实际更新机制上冻结预测；再按预测进行修复或测试分配，验证改变是否按机制发生；最后在有真实依赖结构的连续更新流程中核算完整成本，并与强兼容性训练、直接验证及简单缩放比较。成功标准应在看到结果前固定，同时满足新能力、旧任务损失和成本要求。真实环境还需包含任务所要求的最终回答、工具状态和数值执行条件。

R142 已把这一研究路线所需的若干部件落实为代码、证明、原始数据和可追溯材料。要把论文定位为具有奠基性影响的主刊工作，最关键的剩余增量是上面这条完整证据链的独立验证。持续增加相同设定下的模拟次数无法提供这部分证据。

## 材料使用

先读 R142_MAIN.pdf；需要核对时用 R142_COMBINED.pdf 内部跳转。R142_SUPPLEMENT.pdf 给出完整证明、原始实验表、假设来源及历史范围说明。R142_RESPONSE_TO_COMMENTS.pdf 包含最新五项要求与此前逐条意见；完整 ZIP 内有可编辑 LaTeX、矢量图、实验与分析程序、冻结协议、原始结果和校验清单。投稿信是未发送草稿，需要作者填写实际署名及期刊行政声明。
'''
    (ROOT/'R142_REVISION_REPORT_CN.md').write_text(report)
    previous=list(csv.DictReader((ROOT/'audit/R141_RESPONSE_MATRIX.csv').open()))
    # Preserve the original matrix; add a separate current-request ledger.
    latestrows=[{'item':f'R{i}','request':q,'status':s,'location':loc} for i,q,s,loc in [
      (1,'Nature/Science scientific positioning','Assessment completed; broad claim still unsupported','R142_REVISION_REPORT_CN.md'),
      (2,'Explicit executable algorithms','Completed: four algorithms, code and scoped proofs','manuscript/R142_ALGORITHMS.tex'),
      (3,'Extended Data and visual encodings','Completed: ten displays, original panels retained','manuscript/R142_EXTENDED_DATA.tex'),
      (4,'Bidirectional claim/support links','Completed and checked in final link audit','audit/R142_link_audit.json'),
      (5,'Substantive empirical and theoretical work','Completed within disclosed experiment scope; independent deployment evidence remains open','results/compatibility_analysis/SUMMARY.json')]]
    with (ROOT/'audit/R142_LATEST_REQUEST_MATRIX.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(latestrows[0]));w.writeheader();w.writerows(latestrows)
    print('Wrote current response, cover draft, Chinese scientific assessment and request ledger')
if __name__=='__main__':main()
