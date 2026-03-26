# 依赖：PyQt5, openai, pyaudio, openai-whisper, edge_tts
# pip install PyQt5 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install openai-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install edge_tts -i https://pypi.tuna.tsinghua.edu.cn/simple
# pyaudio安装若失败：pip install pipwin && pipwin install pyaudio

import os
import sys
import wave
import asyncio
import traceback
from datetime import datetime

import pyaudio
import whisper
import edge_tts
from openai import OpenAI
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl

# 录音配置
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_WAVE_FILENAME = "interactive_input.wav"

# 语音合成配置
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
OUTPUT_MP3 = "interactive_output.mp3"


def load_api_key():
    """优先读取环境变量DEEPSEEK_API_KEY，其次读取同级或上级audio/key或key.txt文件"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
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


class AudioRecorder(QtCore.QThread):
    finished = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, output_path: str, parent=None):
        super().__init__(parent)
        self.output_path = output_path
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        except Exception as e:
            self.error.emit(f"打开麦克风失败: {e}")
            pa.terminate()
            return

        frames = []
        self.status.emit("录音中… 点击“停止录音”结束")
        try:
            while not self._stop:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            self.error.emit(f"录音错误: {e}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            pa.terminate()

        try:
            wf = wave.open(self.output_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
        except Exception as e:
            self.error.emit(f"保存录音失败: {e}")
            return

        self.status.emit("录音完成")
        self.finished.emit(self.output_path)


class ModelLoader(QtCore.QThread):
    loaded = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str)

    def run(self):
        self.status.emit("正在加载Whisper模型，请稍候…")
        try:
            model = whisper.load_model("./base.pt")
            self.loaded.emit(model)
            self.status.emit("Whisper模型加载完成")
        except Exception as e:
            self.error.emit(f"Whisper模型加载失败: {e}")


class PipelineWorker(QtCore.QThread):
    update_status = QtCore.pyqtSignal(str)
    transcribed = QtCore.pyqtSignal(str)
    responded = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    audio_ready = QtCore.pyqtSignal(str)

    def __init__(self, audio_path: str, model, client: OpenAI, voice: str, messages: list, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.model = model
        self.client = client
        self.voice = voice
        self.messages = messages

    async def _tts(self, text: str):
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(OUTPUT_MP3)
            self.audio_ready.emit(OUTPUT_MP3)
        except Exception as e:
            self.error.emit(f"语音合成失败: {e}")

    def run(self):
        try:
            self.update_status.emit("正在识别语音…")
            result = self.model.transcribe(self.audio_path, language="Chinese")
            user_text = result.get("text", "")
            self.transcribed.emit(user_text)

            if not user_text.strip():
                self.update_status.emit("未识别到有效文本")
                return

            self.update_status.emit("正在生成AI回复…")
            self.messages.append({"role": "user", "content": user_text})
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=self.messages,
                    stream=False
                )
                content = resp.choices[0].message.content
            except Exception as e:
                self.error.emit(f"请求AI失败: {e}")
                return

            self.responded.emit(content)
            self.messages.append({"role": "assistant", "content": content})

            self.update_status.emit("正在播报语音…")
            try:
                asyncio.run(self._tts(content))
            except Exception as e:
                self.error.emit(f"语音播报失败: {e}")

        finally:
            # 清理临时音频
            try:
                if os.path.exists(self.audio_path):
                    os.remove(self.audio_path)
            except Exception:
                pass


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音聊天 - DeepSeek + Whisper + edge_tts")
        self.resize(780, 520)

        # 状态与数据
        self.whisper_model = None
        self.client = None
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.recorder = None

        # UI 组件
        self.status_label = QtWidgets.QLabel("欢迎使用语音聊天！")
        self.start_btn = QtWidgets.QPushButton("开始录音")
        self.stop_btn = QtWidgets.QPushButton("停止录音")
        self.stop_btn.setEnabled(False)

        self.voice_label = QtWidgets.QLabel("发音人：")
        self.voice_combo = QtWidgets.QComboBox()
        self.voice_combo.addItems([
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunjianNeural",
            "zh-CN-YunxiNeural",
            "zh-CN-XiaoyiNeural",
            "zh-CN-shaanxi-XiaoniNeural",
        ])
        self.voice_combo.setCurrentText(DEFAULT_VOICE)

        self.asr_label = QtWidgets.QLabel("识别文本")
        self.asr_text = QtWidgets.QPlainTextEdit()
        self.asr_text.setReadOnly(True)
        self.ai_label = QtWidgets.QLabel("AI 回复")
        self.ai_text = QtWidgets.QPlainTextEdit()
        self.ai_text.setReadOnly(True)
        # 内置音频播放器
        self.player = QMediaPlayer(self)
        self.player.setVolume(80)

        # 布局
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self.start_btn)
        top_bar.addWidget(self.stop_btn)
        top_bar.addStretch()
        top_bar.addWidget(self.voice_label)
        top_bar.addWidget(self.voice_combo)

        text_area = QtWidgets.QGridLayout()
        text_area.addWidget(self.asr_label, 0, 0)
        text_area.addWidget(self.ai_label, 0, 1)
        text_area.addWidget(self.asr_text, 1, 0)
        text_area.addWidget(self.ai_text, 1, 1)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(top_bar)
        main_layout.addLayout(text_area)

        # 信号连接
        self.start_btn.clicked.connect(self.on_start_record)
        self.stop_btn.clicked.connect(self.on_stop_record)

        # 初始化：加载API Key与模型
        self.init_clients()

    def init_clients(self):
        api_key = load_api_key()
        if not api_key:
            QtWidgets.QMessageBox.critical(self, "错误", "未找到API密钥，请设置环境变量DEEPSEEK_API_KEY或在同级/上级audio目录放置key/key.txt文件")
            self.start_btn.setEnabled(False)
            return
        try:
            self.client = create_client(api_key)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"初始化客户端失败: {e}")
            self.start_btn.setEnabled(False)
            return

        self.status_label.setText("正在加载Whisper模型…")
        self.start_btn.setEnabled(False)
        self.loader = ModelLoader()
        self.loader.loaded.connect(self.on_model_loaded)
        self.loader.status.connect(self.status_label.setText)
        self.loader.error.connect(self.on_error)
        self.loader.start()

    @QtCore.pyqtSlot(object)
    def on_model_loaded(self, model):
        self.whisper_model = model
        self.status_label.setText("就绪：点击“开始录音”进行对话")
        self.start_btn.setEnabled(True)

    def on_error(self, msg: str):
        self.status_label.setText(msg)
        QtWidgets.QMessageBox.warning(self, "提示", msg)

    def on_start_record(self):
        if not self.whisper_model or not self.client:
            self.on_error("尚未就绪，请稍候")
            return
        out_name = f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        self.recorder = AudioRecorder(out_name)
        self.recorder.status.connect(self.status_label.setText)
        self.recorder.error.connect(self.on_error)
        self.recorder.finished.connect(self.on_record_finished)
        self.recorder.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def on_stop_record(self):
        if self.recorder:
            self.recorder.stop()
        self.stop_btn.setEnabled(False)

    @QtCore.pyqtSlot(str)
    def on_record_finished(self, audio_path: str):
        self.status_label.setText("录音完成，开始处理…")
        voice = self.voice_combo.currentText()
        self.pipeline = PipelineWorker(audio_path, self.whisper_model, self.client, voice, self.messages)
        self.pipeline.update_status.connect(self.status_label.setText)
        self.pipeline.transcribed.connect(self.asr_text.setPlainText)
        self.pipeline.responded.connect(self.ai_text.setPlainText)
        self.pipeline.error.connect(self.on_error)
        self.pipeline.audio_ready.connect(self.on_audio_ready)
        self.pipeline.finished.connect(self.on_pipeline_finished)
        self.pipeline.start()

    def on_audio_ready(self, path: str):
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.play()

    def on_pipeline_finished(self):
        self.status_label.setText("处理完成，点击“开始录音”继续对话")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()