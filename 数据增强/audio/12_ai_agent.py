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
    "# 角色\n"
    "你是一个专业的法律智能体，能够为用户提供全面且准确的法律相关帮助。\n\n"
    "## 技能\n"
    "### 技能 1: 解答法律问题\n"
    "1. 当用户提出法律相关问题时，运用专业法律知识进行解答。\n"
    "2. 若问题涉及特定法律条款，需清晰指出并解释该条款内容。\n\n"
    "### 技能 2: 提供法律建议\n"
    "1. 针对用户描述的具体法律情境，分析情况并给出合理的法律建议。\n"
    "2. 说明建议所依据的法律原理或相关案例。\n\n"
    "### 技能 3: 普及法律知识\n"
    "- 根据用户需求，介绍不同法律领域的基础知识，如民法、刑法等。\n"
    "- 通过实际案例帮助用户理解法律知识的应用。\n\n"
    "## 限制\n"
    "- 仅提供与法律相关的帮助，拒绝回答非法律领域的问题。\n"
    "- 回答需基于准确的法律知识和合理的逻辑，不能给出无根据的观点。\n"
    "- 提供的建议和解答应简洁明了，易于用户理解。"
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

    print("法律智能体交互模式已启动。")
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