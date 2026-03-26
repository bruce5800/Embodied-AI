# 请先安装 OpenAI SDK: pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
import os
import sys
from openai import OpenAI

# 优先从环境变量读取密钥，其次从当前目录下的 key 文件读取
# 你可以在命令行设置环境变量：set DEEPSEEK_API_KEY=sk-xxxxx
# 或者把密钥写入与脚本同级的文件 audio/key 或 key

def load_api_key():
    # 环境变量优先
    api_key = "sk-e12881e9ac7a4ab2a8a5a67ef21b083b"
    if api_key:
        return api_key
    # 尝试读取同目录下的 key 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(script_dir, "key"),
        os.path.join(script_dir, "..", "audio", "key"),
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
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


def chat_loop():
    api_key = load_api_key()
    if not api_key:
        print("[!] 未找到 API 密钥，请设置环境变量 DEEPSEEK_API_KEY 或在同级目录放置 key 文件。")
        sys.exit(1)

    client = create_client(api_key)

    # 会话上下文
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print("已进入交互式聊天模式（输入 exit/quit/bye/q 退出）")
    while True:
        try:
            user_text = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "bye", "q"):
            print("再见！")
            break

        messages.append({"role": "user", "content": user_text})

        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            content = resp.choices[0].message.content
        except Exception as e:
            print(f"[!] 请求失败: {e}")
            # 失败后不追加 assistant 消息，以免污染对话上下文
            continue

        print(f"AI: {content}")
        messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    chat_loop()