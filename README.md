# ReasonGenPilot

无训练的图像生成与假设性编辑 Agent 系统。当前仓库已完成 **gen** 文生图优化通路与 **edit** 假设性编辑通路。

## 项目目标

ReasonGenPilot 计划支持三条通路：

| 通路 | 输入 | 目标 |
| --- | --- | --- |
| `gen` | 纯文本 prompt | 优化 prompt，并生成更符合描述的图像 |
| `edit` | 原图 + 假设性指令 | 推理反事实变化，指令式编辑并 VQA 验证 |
| `hybrid` | 原图 + 假设性指令，但需**整图重生成** | Reason 展开 `scene_prompt`，再走 gen 文生图（**非**指令编辑） |

当前已实现 `gen` 与 `edit`；`hybrid` 的 Reason 接口已有（`mode="hybrid"`），`hybrid_pipeline.py`、router 和 demo 待成员 3/4 接入。

## 已实现功能

**gen 通路（成员 1）**

- 项目基础目录与配置模板
- OpenAI-compatible MLLM 调用封装
- DashScope Qwen-Image / OpenAI-like 文生图接口封装
- `dry_run` 占位图模式，方便无 API 时验证流程
- `gen` 通路命令行入口

**edit 通路（成员 2）**

- Reason Agent（ReasonBrain 启发）：四类推理（physical / temporal / causal / story）、细粒度 `visual_cues` / `physics_implications` / `preserve_objects`
- 指令式图像编辑（DashScope Qwen-Image，**无 mask**，图条件全局编辑）
- `finalize_edit_prompt()` 将物理约束与保留对象注入编辑 prompt；推理上下文贯穿 VQA / refine / 多候选
- 多候选 `edit_prompt` 生成、VQA 打分选优与 refine 迭代（默认至少 2 轮）
- 输出 `reason_analysis.json`、`reason_context.txt` 便于调试与报告
- `edit` 通路命令行入口与统一返回结构

## 目录结构

```text
ReasonGenPilot/
├── config/
│   └── .env.example
├── data/
│   ├── input/
│   │   ├── original_prompts.txt
│   │   └── edit/edit_cases.jsonl
│   └── output/
│       ├── gen/
│       └── edit/
├── prompts/
│   ├── gen_system.txt
│   ├── reason_system.txt
│   ├── edit_refine.txt
│   └── edit_candidate.txt
├── reason/
│   ├── api_client.py
│   ├── gen_pipeline.py
│   ├── edit_pipeline.py
│   ├── edit_client.py
│   ├── edit_verify_loop.py
│   ├── reason_agent.py
│   ├── schemas.py
│   └── t2i_client.py
├── requirements.txt
├── README.md
└── 对接说明.md
```

## 环境准备

```powershell
cd E:\copgraphics\lab3\ReasonGenPilot
copy config\.env.example config\.env
pip install -r requirements.txt
```

然后在 `config/.env` 中填写自己的 API key。注意：`config/.env` 已加入 `.gitignore`，不要提交真实密钥。

## 配置示例

```env
MLLM_API_KEY=your_mllm_key_here
MLLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MLLM_MODEL=qwen-vl-plus
MLLM_TIMEOUT=60

T2I_BACKEND=dashscope
T2I_API_KEY=your_t2i_key_here
T2I_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
T2I_MODEL=qwen-image-2.0

EDIT_BACKEND=dashscope
EDIT_API_KEY=your_edit_key_here
EDIT_BASE_URL=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
EDIT_MODEL=qwen-image-2.0
```

`EDIT_*` 可与 `T2I_*` 共用同一组 DashScope 密钥。

## 运行方式

### gen：Dry-run 测试

不调用真实出图 API，只生成 SVG 占位图，用来检查 pipeline 是否跑通。

```powershell
python -m reason.gen_pipeline `
  --prompt "Exactly six steamed buns on a round plate on a wooden table, no extra buns." `
  --output data/output/gen/test0 `
  --iterations 2 `
  --seed 42
```

### gen：真实 API 出图

使用 `--real-api` 后，会调用配置好的 MLLM 优化 prompt，再调用文生图后端生成图片。

```powershell
python -m reason.gen_pipeline `
  --prompt "A grass field filled with red poppies and yellow daisies beside a wooden windmill." `
  --output data/output/gen/qwen_image_real `
  --iterations 1 `
  --real-api
```

### edit：Dry-run 测试

```powershell
python -m reason.edit_pipeline `
  --image data/input/edit/elephant_squirrel_grass.png `
  --instruction "大象和松鼠玩跷跷板会怎样呢?" `
  --output data/output/edit/elephant_seesaw_dry `
  --iterations 2 `
  --min-iterations 2 `
  --candidates 2
```

### edit：真实 API 编辑

```powershell
python -m reason.edit_pipeline `
  --image data/input/edit/elephant_squirrel_grass.png `
  --instruction "大象和松鼠玩跷跷板会怎样呢?" `
  --output data/output/edit/elephant_seesaw `
  --iterations 2 `
  --min-iterations 2 `
  --candidates 2 `
  --real-api
```

## 返回结构

`run_gen_pipeline()` 返回示例：

```json
{
  "final_image": "data/output/gen/case0/image_iter_1.png",
  "final_prompt": "...",
  "reasoning_chain": [],
  "route": "gen"
}
```

`run_edit_pipeline()` 返回 `EditPipelineResult`，含 `edit_prompt`、`vqa_checklist`、`vqa_result`、`iterations`、`metadata.reasoning_type` 等字段，`route` 为 `"edit"`。

单独调用 Reason Agent 时，`ReasonResult` 还包含 `reasoning_type`、`visual_cues`、`physics_implications`、`preserve_objects`（详见 [对接说明.md](./对接说明.md)）。

更详细的对接方式见 [对接说明.md](./对接说明.md)。

## 设计说明：与 ReasonBrain 的关系

本项目参考 [ReasonBrain 论文](https://arxiv.org/abs/2507.01908) 的**假设性指令编辑（HI-IE）**任务与四类推理场景，但在工程上采用 **无训练 Agent + 云端 API** 路线：

| ReasonBrain（论文） | ReasonGenPilot（本仓库） |
|---------------------|--------------------------|
| Reason50K 训练数据 | 零样本 MLLM prompt（`reason_system.txt`） |
| FRCE / CME 模块 | `visual_cues` + `physics_implications` + `preserve_objects` |
| FLUX 扩散端到端 | DashScope **Qwen-Image 指令编辑** |
| 一次 forward | 多候选 + VQA + refine 迭代 |

当前 **edit 默认使用指令编辑，不使用 mask inpaint**（实验表明整图条件编辑在主体保留与场景协调上更稳定）。

## 后续开发

**hybrid（成员 3）**：有原图 + 假设性指令，但变化过大、不适合在原图上指令编辑时（如物体数量重组、昼夜整场景切换）→ Reason Agent 输出 `scene_prompt`，复用 `run_gen_pipeline()` 从零出图。与 edit 共用 `reason_system.txt` 与四类推理字段，**不走 Edit API**。

- 待做：`reason/hybrid_pipeline.py`、`data/input/hybrid/hybrid_cases.jsonl`
- 已有：`run_reason_agent(..., mode="hybrid")` → `scene_prompt`

**集成（成员 4）**：router、统一入口 `pipeline.py`、Gradio demo。

## GenPilot 集成

网络可用时，可将官方 GenPilot 仓库放到 `genpilot/`：

```powershell
git clone https://github.com/27yw/GenPilot.git genpilot
pip install -r genpilot/requirements.txt
```

后续可以把当前轻量 prompt 优化逻辑替换为完整 GenPilot Stage 1/2，但保持 `run_gen_pipeline()` 的外部接口不变。
