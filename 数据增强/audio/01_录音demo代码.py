import pyaudio  # pip install pyaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
import wave
import sys

# 定义录音参数 (与上面相同)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 # 44100 mp3 采样率
WAVE_OUTPUT_FILENAME = "interactive_output.wav"

p = pyaudio.PyAudio()

# 打开音频流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

frames = []

print("按下回车键开始录音，按下 Ctrl+C 结束录音。")
# 等待用户按下回车
input()
print("* 正在录音...")

try:
    # 持续录音直到用户中断
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
except KeyboardInterrupt:
    # 当用户按下 Ctrl+C 时，会触发 KeyboardInterrupt 异常
    print("* 录音结束。")

# 停止并关闭流
stream.stop_stream()
stream.close()
p.terminate()

# 保存文件
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"音频文件已保存为: {WAVE_OUTPUT_FILENAME}")