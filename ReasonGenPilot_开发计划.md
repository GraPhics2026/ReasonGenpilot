# ReasonGenPilot 开发计划（精简版）

> **目标**：GenPilot（描述性生成优化）+ Agent 推理（假设性编辑），全程无训练。  
> **参考**：[GenPilot](https://github.com/27yw/GenPilot) · [ReasonBrain 论文](https://arxiv.org/abs/2507.01908)

---

## 1. 做什么

本项目构建一个 **无训练** 的统一 Agent 系统，支持 **三条通路**（gen / edit / hybrid）：

1. **描述性生成对不齐** → **gen**：GenPilot 优化 prompt；
2. **假设性指令 + 适合局部改图** → **edit**：Reason Agent 结构化推理 → **指令式图像编辑**（Qwen-Image，无 mask）；
3. **假设性指令 + 需整图重生成** → **hybrid**：Reason Agent 输出 `scene_prompt` → gen 通路 T2I（**非** Qwen-Image 指令编辑）。

Router 自动分流；三条通路共用 MLLM 验证逻辑，无需自建数据集或微调模型。

---

### 三条通路说明

#### gen — 描述性生成优化

**适用输入**：只有文字，没有原图。用户给出的是对画面的直接描述。

**典型例子**：「风车旁红色 poppy 和黄色 daisy 占满草地」「桌上正好六个包子」。

**问题**：T2I 模型常听不全复杂 prompt（数错对象、漏掉颜色、空间关系不对）。GenPilot 不能替用户「推理假设」，但擅长把**已经说清楚的描述**改到模型能听懂。

**流程**：
1. 把 prompt 拆成多个语义片段；
2. 首次文生图；
3. 用 Caption + VQA 双路分析与原 prompt 对比，找出错误；
4. 针对错误改写 prompt，生成多个候选，打分、聚类、选最优；
5. 迭代若干轮，输出对齐更好的图和优化后的 prompt。

**依赖**：[GenPilot](https://github.com/27yw/GenPilot) 现成代码 + MLLM API + T2I（本地 FLUX 或 ComfyUI/硅基流动）。

---

#### edit — 假设性图像编辑

**适用输入**：**必须有一张原图**，加上假设性/反事实指令。

**典型例子**：「如果冰块融化了会发生什么」「如果大象和老鼠站在跷跷板上会怎样」。

**问题**：这类指令没有说「删掉冰、加水渍」——需要先结合图像内容和物理/因果/时间常识，推断**应该变成什么样**。单靠 GenPilot 无法完成这一步，因为它默认 prompt 描述的是目标画面，而不是需要推理的「如果句」。

**流程**：
1. **Reason Agent**（MLLM 看图 + `prompts/reason_system.txt`）：输出 `reasoning_type`（physical / temporal / causal / story）、`visual_cues`、`physics_implications`、`preserve_objects`、推理链、英文 `edit_prompt`、VQA 检查清单；
2. **`finalize_edit_prompt()`**：将物理/因果结果与需保留对象注入编辑 prompt；
3. **指令式编辑**：原图 + 完整 `edit_prompt` 送入 DashScope Qwen-Image（**无 mask**，图条件全局编辑）；
4. **Verify**：每轮生成多个候选 `edit_prompt`，逐个编辑并用 MLLM VQA 打分选优（带 `reason_context`）；分数不足则 refine 后再编，默认至少 2 轮；
5. 输出 `reason_analysis.json`、`reason_context.txt` 便于调试与报告。

**与 ReasonBrain 论文的关系**（无训练替代方案）：

| 论文模块 | 本仓库实现 |
|----------|------------|
| Reason50K 训练 | 零样本 MLLM prompt |
| FRCE / CME 细粒度推理 | `visual_cues` + `physics_implications` + `preserve_objects` |
| FLUX 扩散编辑 | DashScope Qwen-Image 指令编辑 |
| 单次 forward | 多候选 + VQA + refine 迭代 |

**说明**：曾实验 mask inpaint（万相 API），对「反事实 + 多物体互动」类任务主体保真不如指令编辑，**当前主路径为指令编辑**。

---

#### hybrid — 推理后整图重生成（待实现 pipeline）

**定位**：与 edit 共用 HI-IE 推理，但 **不在原图上 edit**，而是生成完整 `scene_prompt` 后走 gen。

**适用**：有参考图 + 假设性指令，变化幅度大——构图重组、物体数量变化、全场景氛围切换等 **不适合** 原图指令编辑的 case。

**典型例子**：「如果把这束花分成两束玫瑰会是什么样子」「如果这张室内照变成深夜会是什么样子」。

**与 edit 的区别**：

| | edit | hybrid |
|--|------|--------|
| 是否保留原图像素 | 尽量保留 | 不保留，重新生成 |
| 输出方式 | 指令式编辑（image + text） | T2I 从零出图 |
| Reason 输出 | `edit_prompt` + `target_objects` | `scene_prompt`（完整场景描述） |
| 后续 | Edit API | GenPilot 优化 + 生成 |

**流程**：
1. Reason Agent 看图 + 假设指令 → 输出 `reasoning_chain` + **`scene_prompt`**（英文、可直接用于 T2I）；
2. 可选：把原图风格/色调写进 `scene_prompt` 作为参考约束；
3. 调用 **gen 通路**（GenPilot Stage 1/2）优化并生成新图；
4. 用 VQA 验证新图是否符合推理结果（如「是否两束玫瑰」）。

**Router 如何区分 edit / hybrid**：有图 + 假设性指令时，若变化涉及**整体构图、物体数量重组、全场景氛围**→ hybrid；若仅**局部物理/因果变化**→ edit。

---

### 通路对照表

| 输入 | 通路 | 核心能力 | 流程概要 |
|------|------|----------|----------|
| 纯文字描述 | **gen** | GenPilot：prompt 对齐优化 | 分解 → 生成 → 错误分析 → 迭代改 prompt |
| 图 + 「如果…会怎样」 | **edit** | Agent 推理 + 指令编辑 | Reason Agent → Edit API → 多候选 VQA 验证 |
| 图 + 假设性 + 需整图重画 | **hybrid** | 推理 + 生成 | Reason Agent → scene_prompt → GenPilot |

整体为 **Agent 编排 + 公开 API + GenPilot 现成流程**

---

## 2. 架构

```
用户输入 → Router → gen / edit / hybrid
              │
    gen ──────┴── GenPilot Stage1 + Stage2
    edit ───────── Reason Agent → Edit API（指令编辑）→ VQA verify loop
    hybrid ─────── Reason Agent → GenPilot
```

| 模块 | 来源 | 状态 |
|------|------|------|
| GenPilot Stage 1/2 | 现成 | gen 已封装；可后续替换为完整 Stage 1/2 |
| `gen_pipeline` / `t2i_client` | 成员 1 | ✅ |
| Reason Agent（edit） | 自研 | ✅ |
| Edit Client + edit_pipeline | 自研 | ✅ |
| edit verify loop（多候选 VQA） | 自研 | ✅（集成在 `edit_pipeline.py`） |
| Reason Agent（hybrid） | 自研 | ✅ 接口已有（`mode="hybrid"` → `scene_prompt`） |
| hybrid_pipeline | 自研 | ✅（Reason → `run_gen_pipeline`） |
| Router | 自研 | 待做 |
| pipeline.py | 自研 | 待做 |

---

## 3. 环境准备

```bash
conda create -n reasongenpilot python=3.12 && conda activate reasongenpilot
git clone https://github.com/27yw/GenPilot.git genpilot
pip install -r genpilot/requirements.txt
pip install gradio httpx pydantic python-dotenv pillow
```

**API（至少各 1 个）：**
- **MLLM**：DashScope Qwen-VL（推荐）或 Google AI Studio / 硅基流动
- **出图/编辑**：DashScope Qwen-Image（推荐，gen 与 edit 可共用密钥），或 ComfyUI / 硅基流动

`config/.env` 示例（DashScope）：

```env
MLLM_API_KEY=xxx
MLLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MLLM_MODEL=qwen-vl-plus
T2I_BACKEND=dashscope
T2I_MODEL=qwen-image-2.0
EDIT_BACKEND=dashscope
EDIT_MODEL=qwen-image-2.0
```

---

## 4. 开发步骤

### Step 0：初始化

1. 建目录：`ReasonGenPilot/{config,reason,prompts,data/input,data/output,genpilot}`
2. 写 `reason/api_client.py`（OpenAI 兼容，支持 text + vision）
3. 测通 MLLM：发一条文本、看一张图

---

### Step 1：跑通 GenPilot

1. 准备 `data/input/original_prompts.txt`（2–3 条测试 prompt）
2. 改 `genpilot/error_analysis_pipline.sh` 里的 API 参数，运行 Stage 1
3. 运行 `ttpo.py`（先把 `num_iterations=3`, `num_candidates=5` 调小省额度）
4. 无 GPU 时：写 `reason/t2i_client.py` 调 ComfyUI/硅基流动，替代本地 diffusers
5. 封装 `reason/gen_pipeline.py` 一键跑完 1 个 case

**验收**：有 before/after 图 + 优化后的 `prompt_final.json`

---

### Step 2：跑通 Edit 通路 ✅

1. 准备 `data/input/edit/edit_cases.jsonl` 与原图（如跷跷板、冰块融化、材质变化）
2. `reason/reason_agent.py`：edit 模式输出 `reasoning_type`、`visual_cues`、`physics_implications`、`preserve_objects`、`reasoning_chain`、`edit_prompt`、`vqa_checklist`、`target_objects`；`finalize_edit_prompt()` / `build_reason_context()` 供编辑与 VQA 使用
3. `reason/edit_client.py`：DashScope Qwen-Image 指令编辑 + `dry_run` 占位
4. `reason/edit_pipeline.py`：Reason → 编辑 → 多候选 VQA 选优 → refine 迭代（默认至少 2 轮）；写出 `reason_analysis.json`、`reason_context.txt`
5. `prompts/reason_system.txt`、`edit_refine.txt`、`edit_candidate.txt`

**验收**：1–2 个 edit case 端到端出图，输出 `edit_final.json`、`reason_analysis.json`、`image_before.png`、`image_after.png`

---

### Step 3：跑通 Hybrid 通路 ✅

> edit 用指令编辑改原图；hybrid 用 Reason 写 `scene_prompt` 再 T2I 重画，二者互补。

1. `reason/reason_agent.py` 已支持 `mode="hybrid"` → `scene_prompt`
2. ✅ `reason/hybrid_pipeline.py`（复用 `run_gen_pipeline`，含 7 层 scene_prompt 质量保障 + VQA 验证）
3. ✅ `data/input/hybrid/hybrid_cases.jsonl`（4 个用例：雪景公园/花分两束/白天变深夜/秋季落叶）
4. 验收：原图 → 推理链 → `scene_prompt` → 新图 ✅（case_1/2/4 成功，VQA 得分 1.0）

---

### Step 4：合并系统

1. 写 `reason/router.py`：
   - 无图 → **gen**
   - 有图 + 假设性 + 局部变化 → **edit**
   - 有图 + 假设性 + 整图重生成 → **hybrid**
2. `reason/edit_verify_loop.py` 已作为薄封装存在；Step 4 总入口可直接调 `run_edit_pipeline`
3. 写 `pipeline.py` 统一入口：

```bash
python pipeline.py --prompt "..." --image optional.png --mode auto|gen|edit|hybrid
```

**验收**：gen / edit / hybrid **各跑通 1 例**

---

### Step 5：Demo 与报告

1. `demo_gradio.py`：展示三条通路 + 推理链 / 优化 prompt
2. 对比图：Gen 优化前后；Edit 编辑前后；Hybrid 原图 vs 新生成图
3. 写 README + 实验报告

---

## 5. 目录结构

```
ReasonGenPilot/
├── pipeline.py              # 待做
├── demo_gradio.py           # 可选，待做
├── config/.env
├── genpilot/                # clone（可选）
├── reason/
│   ├── api_client.py
│   ├── router.py            # 待做
│   ├── reason_agent.py
│   ├── edit_client.py
│   ├── t2i_client.py
│   ├── gen_pipeline.py
│   ├── edit_pipeline.py
│   ├── edit_verify_loop.py
│   ├── hybrid_pipeline.py   # ✅
│   └── schemas.py
├── prompts/
│   ├── router_system.txt    # 待做
│   ├── reason_system.txt
│   ├── gen_system.txt
│   ├── edit_refine.txt
│   └── edit_candidate.txt
└── data/
    ├── input/
    │   ├── original_prompts.txt
    │   ├── edit/edit_cases.jsonl
    │   └── hybrid/hybrid_cases.jsonl
    └── output/
        ├── gen/
        ├── edit/
        └── hybrid/
```

---

## 6. API 对照表

| 步骤 | 用什么 |
|------|--------|
| 分解 / VQA / Caption / 改 prompt | MLLM API（gen 与 edit 共用） |
| T2I 生成 | DashScope Qwen-Image / 硅基流动 / ComfyUI |
| 假设性推理（edit / hybrid） | MLLM Vision（`reason_agent.py`） |
| 图像编辑 | DashScope Qwen-Image 指令编辑（image + text） |
| edit 多候选 / refine | MLLM Vision + `edit_candidate.txt` / `edit_refine.txt` |
| edit VQA 验证 | MLLM Vision（`edit_pipeline.verify_edit_result`） |
| 聚类选优 | gen 通路内置启发式 / 后续可接 GenPilot sklearn |
| hybrid scene_prompt 展开 | MLLM Vision（`mode="hybrid"`） |
| hybrid 最终出图 | 同 gen 通路（`run_gen_pipeline`） |

---

## 7. 最小交付

- [x] **gen**：1–2 例，优化前后对比
- [x] **edit**：1–2 例（如跷跷板、材质变化），编辑前后对比 + VQA 迭代
- [x] **hybrid**：1 例（如一束花变两束玫瑰），原图 vs 新图 + 推理链
- [ ] `pipeline.py` 三条通路均可 `--mode auto` 或手动指定
- [x] README（已更新 gen/edit/hybrid 三条通路）

---

