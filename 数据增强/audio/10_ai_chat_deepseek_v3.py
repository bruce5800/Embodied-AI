# 依赖：openai, pyaudio, openai-whisper, edge_tts
# pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install openai-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install edge_tts -i https://pypi.tuna.tsinghua.edu.cn/simple
# pyaudio安装若失败，可: pip install pipwin && pipwin install pyaudio

import os
import sys
import wave
import pyaudio
import whisper
import asyncio
import edge_tts
from openai import OpenAI

# 录音配置
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAVE_FILENAME = "interactive_input.wav"

# 语音合成配置
VOICE = "zh-CN-XiaoxiaoNeural"  # 可改为 zh-CN-YunjianNeural 等
OUTPUT_MP3 = "interactive_output.mp3"


def load_api_key():
    """优先读取环境变量DEEPSEEK_API_KEY，其次读取同级或上级audio/key文件"""
    api_key = "sk-e12881e9ac7a4ab2a8a5a67ef21b083b"
    if api_key:
        return api_key
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
        base_url="https://api.deepseek.com",
    )


def record_audio():
    """按回车开始录音，Ctrl+C结束录音，保存为WAV"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("\n" + "="*50)
    print("  提示：按下回车键开始录音，按下 Ctrl+C 结束录音。")
    print("="*50)
    input("  请按回车键继续...")

    print("\n[*] 正在录音... 请说话。")
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("[*] 录音结束。")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(TEMP_WAVE_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return TEMP_WAVE_FILENAME


def transcribe_audio_with_whisper(audio_path, model):
    print("[*] 正在进行语音识别，请稍候...")
    result = model.transcribe(audio_path, language="Chinese")
    return result.get("text", "")


async def speak_text(text, voice=VOICE, output_file=OUTPUT_MP3):
    """使用edge_tts合成语音并保存为MP3，然后在Windows自动播放"""
    print("[*] 正在合成语音...")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    print(f"[*] 语音已保存到：{output_file}")
    try:
        os.startfile(output_file)  # Windows下自动打开播放
    except Exception:
        pass


def chat_voice_loop():
    # 加载API Key
    api_key = load_api_key()
    if not api_key:
        print("[!] 未找到API密钥，请设置环境变量DEEPSEEK_API_KEY或在同级/上级audio目录放置key文件。")
        sys.exit(1)

    # 初始化DeepSeek Client
    client = create_client(api_key)

    # 加载Whisper模型（需要同级目录存在base.pt）
    print("[*] 正在加载Whisper模型 (确保同级文件夹有base.pt文件)...")
    try:
        model = whisper.load_model("./base.pt")
        print("[*] Whisper模型加载成功！")
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        sys.exit(1)

    # 会话上下文
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    print("已进入语音聊天模式（输入 q 后回车退出；每轮按回车开始录音，Ctrl+C结束）")
    while True:
        cmd = input("\n按回车开始录音，或输入 q 退出: ").strip().lower()
        if cmd == 'q':
            print("再见！")
            break

        try:
            # 录音
            audio_file = record_audio()
            # 识别
            user_text = transcribe_audio_with_whisper(audio_file, model)

            print("\n" + "="*50)
            print("  识别结果:")
            print("="*50)
            print(user_text)
            print("\n")

            # 删除临时文件
            try:
                os.remove(audio_file)
            except Exception:
                pass

            if not user_text.strip():
                print("[!] 未识别到有效文本，请重试。")
                continue

            # 调用DeepSeek
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
                continue

            # 输出文本 & 语音播报
            print(f"AI: {content}")
            asyncio.run(speak_text(content))

            messages.append({"role": "assistant", "content": content})

        except Exception as e:
            print(f"[!] 发生错误: {e}")


if __name__ == "__main__":
    chat_voice_loop()