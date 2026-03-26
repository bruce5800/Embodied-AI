import pyttsx3
# pip install pyttsx3 -i https://pypi.tuna.tsinghua.edu.cn/simple 

# 1. 初始化语音引擎
engine = pyttsx3.init()

# 2. 获取并设置中文语音
voices = engine.getProperty('voices')
for voice in voices:
    # 查找语言代码包含 'zh' 的中文语音
    if 'zh' in voice.languages:
        print(f"找到中文语音: {voice.id}")
        engine.setProperty('voice', voice.id)
        break

# 要转换的文字
text = "你好，这是一个完全离线的语音合成测试。"

# 3. 让引擎朗读文字
print("正在朗读...")
engine.say(text)

# 4. 等待朗读完成
engine.runAndWait()

print("朗读完毕。")