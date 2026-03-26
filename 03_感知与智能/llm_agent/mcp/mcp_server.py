from mcp.server.fastmcp import FastMCP
# 安装fastmcp python支持库 
#pip install fastmcp -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建MCP服务器实例
mcp = FastMCP()
from typing import List
import os
import fnmatch
import time
try:
    import serial
except Exception:
    serial = None  # 运行时再提示缺失依赖
try:
    import pyttsx3
except Exception:
    pyttsx3 = None  # 运行时再提示缺失依赖

#### 工具函数 ####
# 添加加法工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加"""
    print(f"计算 {a} 加 {b}")
    return a + b


# 添加一个减法工具
@mcp.tool()
def sub(a: int, b: int) -> int:
    """两数相减"""
    print(f"计算 {a} 减 {b}")
    return a - b


# 添加一个乘法工具
@mcp.tool()
def mul(a: int, b: int) -> int:
    """两数相乘"""
    print(f"计算 {a} 乘 {b}")
    return a * b


# 添加一个除法工具
@mcp.tool()
def div(a: int, b: int) -> int:
    """两数相除"""
    print(f"计算 {a} 除 {b}")
    return a / b


# 文件搜索工具
@mcp.tool()
def file_search(
    pattern: str,
    start: str = "D:\\",
    exact: bool = False,
    case_sensitive: bool = False,
    first: bool = False,
    exclude: List[str] | None = None,
) -> str:
    """
    在指定起始目录按文件名搜索，支持通配符/精确匹配/大小写控制。

    参数：
      - pattern: 文件名或通配符（如 *.txt）
      - start: 起始目录（默认 D:\\）
      - exact: 是否精确匹配文件名
      - case_sensitive: 是否区分大小写（默认不区分）
      - first: 是否找到第一个匹配后立即停止
      - exclude: 要排除的目录名列表（仅目录名）
    返回：匹配结果的文本列表（每行一个路径），若无结果返回提示文本。
    """

    # 默认排除部分系统目录，避免权限问题或冗余遍历
    default_exclude = ["System Volume Information", "$Recycle.Bin"]
    exclude_set = {d.lower() for d in (exclude if exclude is not None else default_exclude)}

    def match_filename(name: str, patt: str) -> bool:
        if not case_sensitive:
            name = name.lower()
            patt = patt.lower()
        if exact:
            return name == patt
        # 含通配符则使用模式匹配，否则按包含匹配
        if any(ch in patt for ch in "*?"):
            return fnmatch.fnmatch(name, patt)
        return patt in name

    results: List[str] = []
    if not os.path.exists(start):
        return f"起始目录不存在：{start}"

    try:
        for root, dirs, files in os.walk(start, topdown=True):
            # 过滤需要排除的目录（忽略大小写）
            dirs[:] = [d for d in dirs if d.lower() not in exclude_set]
            for f in files:
                if match_filename(f, pattern):
                    full_path = os.path.join(root, f)
                    results.append(full_path)
                    if first:
                        break
            if first and results:
                break
    except PermissionError as e:
        return f"遍历目录时出现权限错误：{e}"
    except FileNotFoundError as e:
        return f"遍历目录时出现文件不存在错误：{e}"

    if results:
        header = f"共找到 {len(results)} 个匹配："
        return header + "\n" + "\n".join(results)
    return "未找到匹配的文件。"


# 机械臂：设置 6 个电机角度工具
@mcp.tool()
def set_student_servo_angles(
    angles: List[float],
    port: str = "COM7",
    baudrate: int = 1000000,
    timeout: float = 0.1,
) -> str:
    """
    设置学生机械臂 6 个电机（ID 1-6）的角度（-90 到 90 度）。

    参数：
      - angles: 长度为 6 的角度列表（单位：度，范围 -90~90）
      - port: 串口名称（默认 COM7）
      - baudrate: 波特率（默认 1000000）
      - timeout: 串口超时（秒，默认 0.1）

    返回：设置结果文本。
    """

    if serial is None:
        return "缺少依赖：未安装 pyserial，请运行 'pip install pyserial' 后重试。"

    if not isinstance(angles, list) or len(angles) != 6:
        return "参数错误：angles 需要为长度为 6 的列表。"

    # Servo 通讯常量（与 05_write_all_student_angle.py 保持一致）
    ADDR_GOAL_POSITION = 42
    ADDR_TORQUE_ENABLE = 40
    INST_WRITE = 3

    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def angle_to_position(angle: float) -> int:
        # 角度 (-90~90) -> 位置 (1024~3072)
        angle = clamp(angle, -90.0, 90.0)
        position = int(((angle + 90.0) / 180.0) * (3072.0 - 1024.0) + 1024.0)
        return clamp(position, 1024, 3072)

    def calculate_checksum(data: List[int]) -> int:
        return (~sum(data)) & 0xFF

    def send_packet(ser: serial.Serial, servo_id: int, instruction: int, parameters: List[int]) -> bool: # type: ignore
        length = len(parameters) + 2
        core = [servo_id, length, instruction] + parameters
        checksum = calculate_checksum(core)
        packet = bytes([0xFF, 0xFF] + core + [checksum])
        try:
            ser.reset_input_buffer()
            ser.write(packet)
            ser.flush()
            return True
        except Exception:
            return False

    def write_register(ser: serial.Serial, servo_id: int, address: int, value: int, size: int = 2) -> bool: # type: ignore
        params = [address]
        if size == 1:
            params.append(value & 0xFF)
        elif size == 2:
            params.extend([value & 0xFF, (value >> 8) & 0xFF])
        else:
            return False
        return send_packet(ser, servo_id, INST_WRITE, params)

    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.1)
        ser.reset_input_buffer()
    except Exception as e:
        return f"无法打开串口 {port}：{e}"

    try:
        # 先为 1-6 号电机上扭矩
        for sid in range(1, 7):
            write_register(ser, sid, ADDR_TORQUE_ENABLE, 1, size=1)
            time.sleep(0.02)

        # 设置角度
        applied: List[str] = []
        for idx, angle in enumerate(angles, start=1):
            pos = int(angle_to_position(float(angle)))
            ok = write_register(ser, idx, ADDR_GOAL_POSITION, pos, size=2)
            if not ok:
                applied.append(f"Servo {idx}: 失败")
            else:
                applied.append(f"Servo {idx}: {float(angle):.1f}° (pos={pos})")
            time.sleep(0.02)

        return "设置结果:\n" + "\n".join(applied)
    except Exception as e:
        return f"设置过程中发生错误：{e}"
    finally:
        try:
            if ser and ser.is_open:
                ser.close()
        except Exception:
            pass


# 文本转语音播放工具（离线）
@mcp.tool()
def tts_speak(
    text: str,
    language_hint: str = "zh",
    voice_id: str | None = None,
    rate: int | None = None,
    volume: float | None = None,
) -> str:
    """
    使用 pyttsx3 离线将文本朗读出来。

    参数：
      - text: 要朗读的文本内容。
      - language_hint: 语言提示，用于挑选合适语音（默认 'zh'）。
      - voice_id: 指定语音 ID（可选）；如未指定会尝试按语言提示选择。
      - rate: 语速（整数，可选），通常默认约 200。
      - volume: 音量（0.0~1.0，可选），默认 1.0。

    返回：执行结果说明文本。
    """

    if pyttsx3 is None:
        return "缺少依赖：未安装 pyttsx3，请运行 'pip install pyttsx3' 后重试。"

    try:
        engine = pyttsx3.init()

        # 可选属性设置
        if rate is not None:
            try:
                engine.setProperty('rate', int(rate))
            except Exception:
                pass
        if volume is not None:
            try:
                v = max(0.0, min(1.0, float(volume)))
                engine.setProperty('volume', v)
            except Exception:
                pass

        selected_voice_id = None

        voices = []
        try:
            voices = engine.getProperty('voices') or []
        except Exception:
            voices = []

        # 如果指定了 voice_id，优先使用
        if voice_id:
            for v in voices:
                if getattr(v, 'id', None) == voice_id:
                    selected_voice_id = v.id
                    break

        # 未指定或未找到时按语言提示选择
        if not selected_voice_id and language_hint:
            hint = str(language_hint).lower()
            for v in voices:
                langs = []
                try:
                    langs = getattr(v, 'languages', [])
                except Exception:
                    langs = []
                # 兼容 bytes 和 str
                normalized = [
                    (l.decode('utf-8', errors='ignore') if isinstance(l, (bytes, bytearray)) else str(l)).lower()
                    for l in langs
                ]
                if any(hint in l for l in normalized):
                    selected_voice_id = getattr(v, 'id', None)
                    break

        # 如有选中语音则设置
        if selected_voice_id:
            try:
                engine.setProperty('voice', selected_voice_id)
            except Exception:
                selected_voice_id = None

        # 朗读
        engine.say(text)
        engine.runAndWait()

        info = f"已朗读文本（长度 {len(text)}）。"
        if selected_voice_id:
            info += f" 使用语音: {selected_voice_id}"
        elif voice_id:
            info += f" 指定语音未找到，使用默认语音。"
        return info
    except Exception as e:
        return f"朗读失败：{e}"


# 机械臂：单独设置某个电机角度工具
@mcp.tool()
def set_student_servo_angle(
    servo_id: int,
    angle: float,
    port: str = "COM12",
    baudrate: int = 1000000,
    timeout: float = 0.1,
) -> str:
    """
    单独设置学生机械臂某个电机的角度（-90 到 90 度）。

    参数：
      - servo_id: 电机 ID（1~6）
      - angle: 角度（度，范围 -90~90）
      - port: 串口名称（默认 COM12）
      - baudrate: 波特率（默认 1000000）
      - timeout: 串口超时（秒，默认 0.1）

    返回：设置结果文本。
    """

    if serial is None:
        return "缺少依赖：未安装 pyserial，请运行 'pip install pyserial' 后重试。"

    if not (1 <= int(servo_id) <= 6):
        return "参数错误：servo_id 需要在 1 到 6 之间。"

    # 常量复用
    ADDR_GOAL_POSITION = 42
    ADDR_TORQUE_ENABLE = 40
    INST_WRITE = 3

    def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def angle_to_position(a: float) -> int:
        a = clamp(a, -90.0, 90.0)
        pos = int(((a + 90.0) / 180.0) * (3072.0 - 1024.0) + 1024.0)
        return int(clamp(pos, 1024, 3072))

    def checksum(data: List[int]) -> int:
        return (~sum(data)) & 0xFF

    def send(ser, sid: int, inst: int, params: List[int]) -> bool:
        length = len(params) + 2
        core = [sid, length, inst] + params
        pkt = bytes([0xFF, 0xFF] + core + [checksum(core)])
        try:
            ser.reset_input_buffer()
            ser.write(pkt)
            ser.flush()
            return True
        except Exception:
            return False

    def write_reg(ser, sid: int, addr: int, val: int, size: int = 2) -> bool:
        params = [addr]
        if size == 1:
            params.append(val & 0xFF)
        elif size == 2:
            params.extend([val & 0xFF, (val >> 8) & 0xFF])
        else:
            return False
        return send(ser, sid, INST_WRITE, params)

    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.1)
        ser.reset_input_buffer()
    except Exception as e:
        return f"无法打开串口 {port}：{e}"

    try:
        # 上扭矩后设置目标位置
        ok_torque = write_reg(ser, int(servo_id), ADDR_TORQUE_ENABLE, 1, size=1)
        if not ok_torque:
            return f"Servo {servo_id}: 启用扭矩失败"

        pos = angle_to_position(float(angle))
        ok = write_reg(ser, int(servo_id), ADDR_GOAL_POSITION, pos, size=2)
        if not ok:
            return f"Servo {servo_id}: 设置角度失败"
        return f"Servo {servo_id}: {float(angle):.1f}° (pos={pos})"
    except Exception as e:
        return f"设置过程中发生错误：{e}"
    finally:
        try:
            if ser and ser.is_open:
                ser.close()
        except Exception:
            pass




# 可以在此处添加更多工具
if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run(transport='sse')