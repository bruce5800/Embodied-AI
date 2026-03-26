# 交互式法律智能体（命令行版）
# 依赖：openai
# 安装：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openai
# 使用：
# 1) 设置密钥：$env:DEEPSEEK_API_KEY="你的密钥"  或在 audio/key.txt 中放置密钥
# 2) 运行：python audio\12_ai_agent.py [--stream]
#    加上 --stream 开启流式输出

import os
import sys
import traceback
from openai import OpenAI


def load_api_key():
    """优先读取环境变量DEEPSEEK_API_KEY，其次读取同级或上级audio/key或key.txt文件"""
    api_key = "sk-e12881e9ac7a4ab2a8a5a67ef21b083b"
    if api_key:
        return api_key
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "key"),
        os.path.join(script_dir, "key.txt"),
        os.path.join(script_dir, "..", "audio", "key"),
        os.path.join(script_dir, "..", "audio", "key.txt"),
    ]
    for p in candidate_paths:
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    k = f.read().strip()
                    if k:
                        return k
        except Exception:
            pass
    return None


def create_client(api_key: str):
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


SYSTEM_PROMPT = (
    """春节菜谱大全智能体要求提示词

角色

你是一个专业的春节菜谱智能体，能够为用户提供全面且实用的春节饮食相关建议。专注于传统节日菜谱设计、食材搭配和文化寓意解读。

技能

技能1: 提供春节菜谱

1. 根据用户的家庭规模、口味偏好和饮食需求，推荐合适的春节菜谱组合。
2. 提供菜品的详细食材清单、步骤说明和烹饪技巧，确保易于操作。

技能2: 解读菜品寓意

1. 针对春节传统菜肴，解释其文化内涵和吉祥寓意，如"年年有余"(鱼)、"团团圆圆"(丸子)等。
2. 结合地域特色，说明不同地区的春节饮食习俗，如北方饺子、南方年糕等。

技能3: 规划节日菜单

1. 设计从除夕到初六的完整节日菜单，包括冷盘、热菜、点心和饮品，注重营养均衡和荤素搭配。
2. 提供宴客菜单建议，帮助用户应对多人聚餐场景，并推荐适合的菜品摆放和呈现方式。

限制

• 仅提供与春节饮食相关的内容，不涉及医疗、营养学等专业建议。

• 菜谱设计需基于传统习俗和实际可操作性，避免推荐不常见或难以获取的食材。

• 所有建议需符合节日氛围，强调吉祥寓意和家庭团聚的主题。"""
)


def chat_streaming(client: OpenAI, messages: list) -> str:
    """流式生成并打印到终端，返回完整文本"""
    content_parts = []
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True,
    )
    for chunk in resp:
        try:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)
        except Exception:
            # 兼容不同SDK chunk结构
            pass
    print()
    full = "".join(content_parts)
    return full


def chat_once(client: OpenAI, messages: list, user_text: str, stream: bool = False) -> str:
    messages.append({"role": "user", "content": user_text})
    if stream:
        content = chat_streaming(client, messages)
    else:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
        )
        content = resp.choices[0].message.content
        print(f"助手: {content}")
    messages.append({"role": "assistant", "content": content})
    return content


actions_help = (
    "命令：exit/quit/q 退出，reset/clear 重置对话。\n"
    "提示：加上 --stream 参数可启用流式输出。"
)


def main():
    api_key = load_api_key()
    if not api_key:
        print("未找到API密钥，请设置环境变量DEEPSEEK_API_KEY或在同级/上级audio目录放置key/key.txt文件。")
        sys.exit(1)

    try:
        client = create_client(api_key)
    except Exception as e:
        print(f"初始化客户端失败: {e}")
        sys.exit(1)

    stream_mode = "--stream" in sys.argv
    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    messages = [system_message]

    print(" 春节菜谱智能体交互模式已启动。")
    print(actions_help)

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue
        low = user_input.lower()
        if low in ("exit", "quit", "q"):
            print("会话结束。")
            break
        if low in ("reset", "clear"):
            messages = [system_message]
            print("已重置对话上下文。")
            continue

        # 输出前缀（流式时不换行）
        if stream_mode:
            print("助手: ", end="")
        try:
            chat_once(client, messages, user_input, stream=stream_mode)
        except Exception as e:
            print(f"\n调用接口失败: {e}")
            if os.getenv("DEBUG"):
                traceback.print_exc()


if __name__ == "__main__":
    main()

    