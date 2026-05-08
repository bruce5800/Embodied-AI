"""
LLM 任务规划器 — 自然语言 → 放置指令
====================================
用 DeepSeek API 将用户的自然语言指令解析为结构化的放置指令 JSON。

用法：
  # 作为模块被 grasp_pipeline.py 调用
  from llm_planner import LLMPlanner
  planner = LLMPlanner()
  instructions = planner.parse("把红色物体放到蓝色方块上方")

  # 独立测试
  python llm_planner.py "把四个物体排成一排"
"""

import json
import os

# ─── DeepSeek API 配置 ───
# 优先从环境变量读取，其次用默认值
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-e12881e9ac7a4ab2a8a5a67ef21b083b")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ─── 场景描述（注入给 LLM 的上下文）───
SCENE_CONTEXT = """
你是一个机械臂分拣任务规划器。用户会用自然语言描述想要的物体摆放方式，你需要将其转换为结构化的 JSON 指令列表。

## 当前场景

机械臂工作台上有以下物体：
- red_cylinder（红色圆柱）：大约在 (-0.20, -0.10) 位置
- blue_cube（蓝色方块）：大约在 (-0.35, 0.12) 位置
- green_sphere（绿色球体）：大约在 (-0.50, -0.05) 位置
- yellow_cylinder（黄色圆柱）：大约在 (-0.55, 0.15) 位置

预定义放置区域：
- zone_red：(-0.15, -0.35)  红色放置区
- zone_blue：(-0.15, 0.35)  蓝色放置区
- zone_green：(-0.50, 0.00) 绿色放置区
- zone_yellow：(-0.15, 0.00) 黄色放置区

坐标系：X 轴指向机械臂前方（负方向远离基座），Y 轴是左右方向，Z 轴是高度。
机械臂基座在原点 (0, 0, 0)，工作范围大约 X: -0.15 ~ -0.55, Y: -0.35 ~ 0.35。
物体放在地面时 Z = 0。

## 输出格式

你必须输出一个 JSON 数组，每个元素是一条放置指令。支持以下 4 种模式：

### 1. zone 模式 — 放到预定义区域
```json
{"object": "red_cylinder", "mode": "zone", "zone": "zone_red"}
```

### 2. absolute 模式 — 放到绝对坐标
```json
{"object": "red_cylinder", "mode": "absolute", "position": [-0.3, 0.1, 0.0]}
```

### 3. relative 模式 — 放到另一物体的相对位置
```json
{"object": "red_cylinder", "mode": "relative", "ref_object": "blue_cube", "relation": "above", "offset": 0.08}
```
relation 可选值：above（上方）, below（下方）, left（左）, right（右）, front（前）, behind（后）
offset 是偏移距离（米），默认 0.08

### 4. line 模式 — 多个物体排成一排
```json
{"objects": ["red_cylinder", "blue_cube", "green_sphere"], "mode": "line", "start": [-0.35, -0.20, 0.0], "end": [-0.35, 0.20, 0.0]}
```

## 重要规则

1. 只输出 JSON 数组，不要输出任何其他文字
2. 物体名称必须使用英文 body 名称（red_cylinder, blue_cube, green_sphere, yellow_cylinder）
3. 坐标必须在机械臂可达范围内
4. 如果用户说"所有物体"，包含全部 4 个
5. 堆叠时 offset 建议 0.06~0.10
6. 排列时保持在工作台范围内（X: -0.15 ~ -0.50, Y: -0.30 ~ 0.30）
7. 如果指令不明确，使用合理的默认值
"""


class LLMPlanner:
    """调用 DeepSeek API 将自然语言转换为放置指令。"""

    def __init__(self, api_key=None, base_url=None, model=None):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=base_url or DEEPSEEK_BASE_URL,
        )
        self.model = model or DEEPSEEK_MODEL
        self.history = [{"role": "system", "content": SCENE_CONTEXT}]

    def parse(self, user_input, scene_objects=None):
        """
        将自然语言指令解析为放置指令列表。

        Args:
            user_input: 用户自然语言，如 "把红色物体放到蓝色方块上方"
            scene_objects: 可选，当前检测到的物体信息（用于动态更新上下文）

        Returns:
            list[dict] — 放置指令列表，可直接传给 resolve_place_target()
        """
        # 如果有实时检测到的物体位置，附加到消息中
        prompt = user_input
        if scene_objects:
            obj_info = "\n".join(
                f"  {name}: 当前位置 ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                for name, _conf, _body, pos in scene_objects
            )
            prompt = f"当前物体实际位置:\n{obj_info}\n\n用户指令: {user_input}"

        self.history.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=0.1,  # 低温度，输出更稳定
            )

            reply = response.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": reply})

            # 提取 JSON（处理 LLM 可能包裹在 ```json ``` 中的情况）
            instructions = self._extract_json(reply)
            return instructions

        except Exception as e:
            print(f"  ✗ DeepSeek API 调用失败: {e}")
            return None

    def _extract_json(self, text):
        """从 LLM 回复中提取 JSON 数组。"""
        # 去掉 markdown 代码块
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        text = text.strip()

        try:
            result = json.loads(text)
            # 确保是列表
            if isinstance(result, dict):
                result = [result]
            return result
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON 解析失败: {e}")
            print(f"  原始回复: {text[:200]}")
            return None


# ═══════════════════════════════════════════════════════════════
#  独立测试
# ═══════════════════════════════════════════════════════════════

def main():
    import sys

    planner = LLMPlanner()

    if len(sys.argv) > 1:
        # 命令行模式
        user_input = " ".join(sys.argv[1:])
        print(f"用户指令: {user_input}\n")
        instructions = planner.parse(user_input)
        if instructions:
            print("生成的放置指令:")
            print(json.dumps(instructions, indent=2, ensure_ascii=False))
        else:
            print("解析失败")
    else:
        # 交互模式
        print("LLM 任务规划器 — 输入自然语言指令，输出 JSON 放置指令")
        print("输入 quit 退出\n")

        test_cases = [
            "把红色圆柱放到蓝色方块的上方",
            "把所有物体排成一排",
            "把绿色球放到红色区域",
            "把红色和蓝色物体交换位置",
        ]
        print("测试用例:")
        for i, tc in enumerate(test_cases):
            print(f"  {i+1}. {tc}")
        print()

        while True:
            user_input = input(">>> ").strip()
            if not user_input or user_input.lower() == "quit":
                break

            # 支持输入数字选择测试用例
            if user_input.isdigit() and 1 <= int(user_input) <= len(test_cases):
                user_input = test_cases[int(user_input) - 1]
                print(f"  → {user_input}")

            instructions = planner.parse(user_input)
            if instructions:
                print("\n生成的放置指令:")
                print(json.dumps(instructions, indent=2, ensure_ascii=False))
            else:
                print("  解析失败")
            print()


if __name__ == "__main__":
    main()
