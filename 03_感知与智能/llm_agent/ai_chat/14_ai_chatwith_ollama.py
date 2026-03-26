import sys
import ollama

MODEL_NAME = 'deepseek-r1:1.5b'


def model_exists(name: str) -> bool:
    """Check if the model exists locally. If the check fails, assume true."""
    try:
        models = ollama.list()
        return any(
            m.get('model') == name or m.get('name') == name
            for m in models.get('models', [])
        )
    except Exception:
        # If listing models fails (e.g., server not running), don't block the chat
        return True


def chat_loop():
    print(f"已准备使用模型: {MODEL_NAME}")
    print("输入内容开始聊天，输入 /exit 退出。\n")

    messages = [
        {
            'role': 'system',
            'content': '你是一个乐于助人的中文 AI 助手，简洁、直接、友好地回答问题。',
        }
    ]

    while True:
        try:
            user_input = input("你> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"/exit", "/quit", "/q"}:
                print("已退出聊天。")
                break

            messages.append({'role': 'user', 'content': user_input})
            response = ollama.chat(model=MODEL_NAME, messages=messages)
            assistant_content = response['message']['content']

            print(f"AI> {assistant_content}\n")
            messages.append({'role': 'assistant', 'content': assistant_content})

        except KeyboardInterrupt:
            print("\n已退出聊天。")
            break
        except Exception as e:
            print(f"[错误] {e}")
            print("请确认 Ollama 正在运行且模型已拉取: ollama pull deepseek-r1:1.5b")
            break


def demo_once():
    """Run a single chat turn like the example provided."""
    prompt = '从前有座山，山里有个庙，续写一下'
    response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': prompt}])

    # Some versions of the client return a dict; user's example shows both attribute and key access
    try:
        print(response.message.content)
    except Exception:
        pass
    print(response['message']['content'])


if __name__ == '__main__':
    # Optional: Check model availability and give a helpful hint
    if not model_exists(MODEL_NAME):
        print(f"未找到本地模型 {MODEL_NAME}。请先运行: ollama pull {MODEL_NAME}\n")

    # If user passes --demo, run a single-turn demo; otherwise start interactive chat
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_once()
    else:
        chat_loop()