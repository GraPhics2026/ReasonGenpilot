# ReasonGenPilot 四人分工

> 配套：[ReasonGenPilot_开发计划.md](./ReasonGenPilot_开发计划.md)

建议按 **成员 1 → 2 → 3 → 4** 的顺序推进：先把 gen 跑通，再接 edit，然后 hybrid 和验证，最后做总入口和 Demo。

```
成员 1 ──► 成员 2 ──► 成员 3 ──► 成员 4
基座+gen    edit 全链路   hybrid+verify   集成+Demo+文档
```

| 成员 | 主要负责 | 做完之后 |
|------|----------|----------|
| 成员 1 | 基座 + gen 通路 | `run_gen_pipeline` 能出优化后的图 |
| 成员 2 | edit 全链路 | `run_edit_pipeline` 能完成假设性编辑 |
| 成员 3 | hybrid + edit 验证 | `run_hybrid_pipeline` 可用；edit 能迭代验证 |
| 成员 4 | Router + pipeline + Demo + 文档 | 一条命令走 gen / edit / hybrid |

---

## 成员 1：基座 + gen 通路

先把项目骨架和 **GenPilot 生成优化** 整条链路跑通，后面 edit / hybrid 都会用到这里的 `api_client` 和 `gen_pipeline`。

| 任务 | 产出 |
|------|------|
| 目录、`config/.env.example` | 项目骨架 |
| `reason/schemas.py` | 数据结构约定（后面的人在此基础上扩展） |
| `reason/api_client.py` | MLLM 文本 + 视觉调用 |
| GenPilot Stage 1 / 2 配置与跑通 | `genpilot/` |
| 无 GPU 时 `reason/t2i_client.py` | ComfyUI / 硅基流动出图 |
| `reason/gen_pipeline.py` | gen 一键调用 |
| gen 测试 prompt 与对比图 | `data/input/`、`data/output/gen/` |

跑通后可以类似这样测：

```bash
python -m reason.gen_pipeline --prompt "..." --output data/output/gen/test0
```

**报告**：项目背景、Gen 通路（GenPilot）、Gen 实验对比图。

---

## 成员 2：edit 通路

在成员 1 的基础上，做完整的 **假设性编辑**：Reason 推理 → 指令式图像编辑 → VQA 验证迭代。

| 任务 | 产出 |
|------|------|
| 扩展 `schemas.py`（`ReasonResult` / `EditPipelineResult` 等） | JSON 约定（含 `reasoning_type`、细粒度推理字段） |
| `prompts/reason_system.txt` + `reason/reason_agent.py`（edit 模式） | 四类推理 → `edit_prompt` + VQA 清单；`finalize_edit_prompt` / `build_reason_context` |
| `prompts/edit_refine.txt` + `prompts/edit_candidate.txt` | refine / 多候选 prompt 生成（注入 `reason_context`） |
| edit 测试图 + `data/input/edit/edit_cases.jsonl` | 至少 2 例（跷跷板、冰块融化、材质变化等） |
| `reason/edit_client.py` | DashScope 指令编辑 + `dry_run` 占位 |
| `reason/edit_pipeline.py` | edit 一键调用（多候选 VQA 选优 + verify loop；输出 `reason_analysis.json`） |
| `reason/edit_verify_loop.py` | 供成员 3 / 总入口调用的薄封装 |

当前实现采用 **image + text 指令编辑**（DashScope Qwen-Image），**不使用 mask / inpaint**（实验对比后选定）；参考 ReasonBrain 的 HI-IE 任务与四类推理，用零样本 Agent 替代论文训练模块。

```bash
python -m reason.edit_pipeline \
  --image data/input/edit/elephant_squirrel_grass.png \
  --instruction "大象和松鼠玩跷跷板会怎样呢?" \
  --output data/output/edit/elephant_seesaw \
  --iterations 2 --min-iterations 2 --candidates 2 \
  --real-api
```

**报告**：假设性编辑、Reason Agent（edit）、指令编辑与 VQA 闭环、Edit 对比图。

---

## 成员 3：hybrid + edit 验证调优

接上 gen 和 edit，补 **hybrid 通路**（Reason → `scene_prompt` → gen 整图重生成，**不用 Edit API**）；edit 的 VQA 闭环已在 `edit_pipeline.py` 中实现。

**何时走 hybrid**：构图/物体数量/全场景氛围大变；edit 指令改图保真不足时可退到 hybrid。

| 任务 | 产出 | 状态 |
|------|------|------|
| `reason/reason_agent.py` hybrid 模式（`scene_prompt`） | 共用 `reason_system.txt` | ✅ |
| `reason/hybrid_pipeline.py` | 封装 Reason → `run_gen_pipeline` | 待做 |
| `data/input/hybrid/hybrid_cases.jsonl` | 如「一束花变两束玫瑰」 | 待做 |
| edit 参数实验（可选） | iterations / candidates 记录 | 可选 |

hybrid 大致逻辑：

```python
def run_hybrid_pipeline(image_path, instruction, output_dir):
    r = run_reason_agent(image_path, instruction, mode="hybrid")
    return run_gen_pipeline(r.scene_prompt, output_dir)
```

edit 验证可直接复用成员 2 的接口：

```python
from reason.edit_verify_loop import run_edit_verify_loop

result = run_edit_verify_loop(image_path, instruction, output_dir)
```

**报告**：Hybrid 设计、edit 验证实验、hybrid 对比图。

---

## 成员 4：集成 + Demo + 文档

三条子通路都齐了，做 **Router、总入口、答辩 Demo 和文档统稿**。

| 任务 | 产出 |
|------|------|
| `router.py` + `router_system.txt` | gen / edit / hybrid 分流 |
| `pipeline.py` | 统一调度（主要调用已有函数） |
| `demo_gradio.py` | 三通路演示 |
| `README.md`、报告、PPT | 交付文档 |

Router 怎么分：

| 输入 | 走哪条 |
|------|--------|
| 无图，纯描述 | gen |
| 有图 + 假设性 + 局部改（融化、倾斜等） | edit |
| 有图 + 假设性 + 整图大变（分束花、昼夜等） | hybrid |

`pipeline.py` 示例：

```python
if route == "gen":
    return run_gen_pipeline(prompt, output_dir)
elif route == "edit":
    return run_edit_pipeline(image_path, prompt, output_dir).to_dict()
elif route == "hybrid":
    return run_hybrid_pipeline(image_path, prompt, output_dir)
```

**报告**：总体架构、Router、摘要、结论、附录。

---

## 数据结构约定

成员 1 先写 `schemas.py`，后面按需扩展。

**edit：**
```json
{
  "mode": "edit",
  "reasoning_type": "physical",
  "reasoning_chain": ["..."],
  "visual_cues": ["fine-grained facts from the image"],
  "physics_implications": ["expected visible outcome"],
  "target_objects": ["objects or regions to change"],
  "preserve_objects": ["background or objects to keep"],
  "edit_prompt": "...",
  "vqa_checklist": [{"q": "...", "expected": "yes"}]
}
```

`reasoning_type`：`physical` | `temporal` | `causal` | `story`（与 Reason50K 四类对齐）。

**hybrid：**
```json
{
  "mode": "hybrid",
  "reasoning_type": "story",
  "reasoning_chain": ["..."],
  "visual_cues": ["..."],
  "physics_implications": ["..."],
  "scene_prompt": "...",
  "vqa_checklist": [{"q": "...", "expected": "yes"}]
}
```

**各 pipeline 返回：**
```json
{
  "final_image": "...",
  "final_prompt": "...",
  "reasoning_chain": ["..."],
  "route": "gen|edit|hybrid"
}
```

---

## 交付一览

| 通路 | 谁做 | 内容 |
|------|------|------|
| gen | 成员 1 | 2 例 before/after |
| edit | 成员 2 | 2 例编辑对比（含 VQA 迭代） |
| hybrid + verify 调优 | 成员 3 | 1 例 hybrid；edit 验证参数实验 |
| 集成 | 成员 4 | pipeline、Gradio、README、报告 |
