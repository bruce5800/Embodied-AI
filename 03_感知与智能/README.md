# 03 感知与智能 — 学习阶段的零散探索

本阶段是计算机视觉、目标检测、LLM agent、语音识别等具身 AI 感知/智能模块的学习与试验，每个子目录是一个独立方向的小实验。
**这些功能（除语音外）后续被整合进 [04 自主决策与学习](../04_自主决策与学习)** 形成完整 demo（YOLO 视觉感知 + DeepSeek LLM 任务规划），本目录保留作为单点学习记录。

## 子目录

| 子目录 | 内容 | 整合去向 |
|--------|------|---------|
| `computer_vision/` | OpenCV 基础（图像/相机/HSV 色彩空间/ROI） | 04 `vision_pipeline.py`（基础工具） |
| `object_detection/` | YOLO 训练 + 推理 + 数据增广 | 04 `vision_pipeline.py`（detect 模式 + dataset 生成） |
| `llm_agent/` | LLM API 接入（chat + MCP 协议试验） | 04 `llm_planner.py`（DeepSeek 自然语言 → JSON 指令） |
| `speech/` | 语音识别 / TTS / 离线语音 | **暂未整合**（探索性试验，未进入主流程） |

## 命名说明

子目录里的脚本多以 `01_xxx.py / 02_xxx.py` 编号 —— 这是**学习顺序**，不是依赖顺序。每个文件可独立运行，作为某个知识点的最小验证。

## 关键试验

- `computer_vision/05~07_opencv_camera_hsv_*.py` — HSV 色彩分割（蓝色/黑色/分类标记），后续被 YOLO 替代
- `object_detection/yolo_training/` — YOLO 训练流水线，模型最终用在 04
- `object_detection/data_augmentation/` — 训练数据增广脚本
- `llm_agent/ai_chat/` — DeepSeek API 多轮对话，后续被改造成 04 的 llm_planner
- `llm_agent/mcp/` — Model Context Protocol 试验，未进入主流程
- `speech/02_录音转文本.py` + `03_文本转语音.py` — 语音控制接口探索（未整合）

## 不需要看本目录的情况

如果只关心整合后的 demo 或最终成果，**直接看 [04 自主决策与学习](../04_自主决策与学习) 即可**。本目录是过程记录，对外部读者参考价值有限。
