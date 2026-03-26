import re
import random
import asyncio
import edge_tts
import pyaudio
import wave
import whisper
import os


class SimpleKernel:
    def __init__(self):
        self.categories = {}
        self.last_response = None
        self.variables = {}

    def learn(self, aiml_content):
        pattern = re.compile(r'<category>.*?<pattern>(.*?)</pattern>(?:\s*<that>(.*?)</that>)?\s*<template>(.*?)</template>.*?</category>', re.DOTALL)
        matches = pattern.findall(aiml_content)
        for match in matches:
            pattern_text = match[0].strip().upper()
            that_text = match[1].strip().upper() if match[1] else None
            template_text = match[2].strip()
            self.categories[(pattern_text, that_text)] = template_text

    def respond(self, input_text):
        input_text = input_text.strip().upper()
        for (pattern_text, that_text), template_text in self.categories.items():
            if re.match(pattern_text.replace('*', '.*'), input_text):
                if that_text is None or (self.last_response and re.match(that_text.replace('*', '.*'), self.last_response)):
                    self.last_response = input_text
                    response = self.process_template(template_text)
                    return response
        return "抱歉，我不太理解你的意思。"

    def process_template(self, template):
        if '<random>' in template:
            choices = re.findall(r'<li>(.*?)</li>', template, re.DOTALL)
            template = random.choice(choices)
        template = re.sub(r'<get name="(.*?)"/>', lambda match: self.variables.get(match.group(1), ''), template)
        return template

    def set_variable(self, name, value):
        self.variables[name] = value


aiml_content = """
<?xml version="1.0" encoding="UTF-8"?>
<aiml version="1.0.1">
    <category>
        <pattern>HELLO</pattern>
        <template>Hi there!</template>
    </category>
    <category>
        <pattern>HOW ARE YOU</pattern>
        <template>I'm doing well, thank you!</template>
    </category>
    <category>
        <pattern>WHAT IS YOUR NAME</pattern>
        <template>My name is Alice.</template>
    </category>
    <category>
        <pattern>你多大了</pattern>
        <template>
            <random>
                <li>哦，我今年 <get name="age"/> 岁，如花似玉的年龄。</li>
                <li>我都 <get name="age"/> 岁了，好棒。</li>
                <li><get name="age"/> 岁？我比你年轻好多呢。</li>
                <li>哦，<get name="age"/> 岁，您学到的知识比我多得多呢。</li>
            </random>
        </template>
    </category>
    <category>
        <pattern>*睡*</pattern>
        <template>我是人工智能，不需要睡觉。不过，真希望自己也能做个美梦呢。</template>
    </category>
    <category>
        <pattern>今天天气如何</pattern>
        <template>你想咨询哪个地区的天气?</template>
    </category>
    <category>
        <pattern>今天股票怎么样</pattern>
        <template>今天买啥啥都涨</template>
    </category>
</aiml>
"""



CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAVE_FILENAME = "interactive_input.wav"
VOICE = "zh-CN-XiaoxiaoNeural"
OUTPUT_MP3 = "interactive_output.mp3"


def record_audio():
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
    print("[*] 正在合成语音...")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    print(f"[*] 语音已保存到：{output_file}")
    try:
        os.startfile(output_file)
    except Exception:
        pass


def main():
    print("[*] 正在加载 Whisper 模型 (确保同级文件夹有base.pt文件)...")
    try:
        model = whisper.load_model("./base.pt")
        print("[*] Whisper 模型加载成功！")
    except Exception as e:
        print(f"[!] 模型加载失败: {e}")
        return

    alice = SimpleKernel()
    alice.learn(aiml_content)
    alice.set_variable("age", "25")

    print("已进入语音聊天模式（按 Ctrl+C 结束录音；输入 q 后回车退出程序）")
    while True:
        cmd = input("\n按回车开始录音，或输入 q 退出: ").strip().lower()
        if cmd == 'q':
            print("再见！")
            break
        try:
            audio_file = record_audio()
            user_text = transcribe_audio_with_whisper(audio_file, model)
            print("\n" + "="*50)
            print("  识别结果:")
            print("="*50)
            print(user_text)
            print("\n")
            try:
                os.remove(audio_file)
            except Exception:
                pass
            response = alice.respond(user_text)
            print(f"Alice: {response}")
            asyncio.run(speak_text(response))
        except Exception as e:
            print(f"[!] 发生错误: {e}")


if __name__ == "__main__":
    main()