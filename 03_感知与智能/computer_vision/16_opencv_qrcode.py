import cv2
import numpy as np
from pyzbar import pyzbar
import time
from PIL import Image, ImageDraw, ImageFont
import os

def decode_qr_data(raw_data):
    """
    尝试多种编码方式解码二维码数据，优先处理中文
    """
    # 打印调试信息
    print(f"\n=== QR码解码调试信息 ===")
    print(f"原始数据 (bytes): {raw_data}")
    print(f"原始数据 (hex): {raw_data.hex()}")
    print(f"原始数据长度: {len(raw_data)}")
    
    # 检查是否包含UTF-8替换字符的编码模式
    if b'\xef\xbd' in raw_data:
        print("⚠️ 检测到二维码数据包含UTF-8替换字符，可能是生成时编码损坏")
        # 尝试修复常见的编码损坏模式
        repaired_data = attempt_repair_encoding(raw_data)
        if repaired_data != raw_data:
            print(f"🔧 尝试修复编码损坏: {repaired_data.hex()}")
            # 递归调用，使用修复后的数据
            return decode_qr_data(repaired_data)
    
    # 尝试的编码方式列表，优先使用常见的中文编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'shift_jis', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            decoded_text = raw_data.decode(encoding)
            print(f"尝试 {encoding} 编码: '{decoded_text}' (长度: {len(decoded_text)})")
            
            # 验证解码结果的有效性
            if is_valid_decoded_text(decoded_text):
                print(f"✓ 选择编码: {encoding}, 结果: '{decoded_text}'")
                return decoded_text
            else:
                print(f"✗ {encoding} 编码验证失败")
                
        except (UnicodeDecodeError, UnicodeError, LookupError) as e:
            print(f"✗ {encoding} 编码失败: {e}")
            continue
    
    # 如果所有编码都失败，使用错误替换模式
    try:
        fallback_result = raw_data.decode('utf-8', errors='replace')
        print(f"使用UTF-8错误替换模式: '{fallback_result}'")
        print(f"⚠️ 无法正确解码，可能是二维码生成时编码损坏")
        print(f"💡 建议：重新生成二维码，确保使用正确的UTF-8编码")
        return f"[编码损坏] {fallback_result}"
    except:
        hex_result = f"Raw: {raw_data.hex()}"
        print(f"最终回退到十六进制: {hex_result}")
        return hex_result

def attempt_repair_encoding(raw_data):
    """
    尝试修复常见的编码损坏模式
    """
    # 将数据转换为字符串进行分析
    try:
        text = raw_data.decode('utf-8', errors='replace')
        
        # 检查是否包含替换字符
        if '�' in text:
            # 尝试一些常见的修复策略
            
            # 策略1: 如果是GBK->UTF-8转换错误，尝试反向修复
            try:
                # 将替换字符的位置标记出来，尝试用原始字节重新解码
                original_bytes = bytearray(raw_data)
                
                # 移除UTF-8替换字符的字节序列 (0xEF 0xBD 0x*)
                repaired_bytes = bytearray()
                i = 0
                while i < len(original_bytes):
                    if (i + 2 < len(original_bytes) and 
                        original_bytes[i] == 0xEF and 
                        original_bytes[i + 1] == 0xBD):
                        # 跳过UTF-8替换字符序列
                        i += 3
                    else:
                        repaired_bytes.append(original_bytes[i])
                        i += 1
                
                if len(repaired_bytes) > 0 and repaired_bytes != original_bytes:
                    return bytes(repaired_bytes)
                    
            except:
                pass
    
    except:
        pass
    
    return raw_data

def is_valid_decoded_text(text):
    """
    验证解码后的文本是否有效（不是乱码）
    """
    if not text:
        return False
    
    # 检查是否包含过多的控制字符或无效字符
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    if control_chars > len(text) * 0.1:  # 如果控制字符超过10%，可能是乱码
        return False
    
    # 检查是否包含常见的乱码字符
    garbled_chars = ['�', '?', '\ufffd']
    if any(char in text for char in garbled_chars):
        return False
    
    # 检查字符的连续性（中文字符通常在特定Unicode范围内）
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    other_chars = len(text) - chinese_chars - ascii_chars
    
    # 如果其他字符（可能的乱码）占比过高，认为是无效的
    if other_chars > len(text) * 0.3:
        return False
    
    return True

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在OpenCV图像上显示中文文本，支持中文字体
    """
    print(f"\n=== 中文显示调试信息 ===")
    print(f"要显示的文本: '{text}'")
    print(f"文本长度: {len(text)}")
    print(f"文本编码: {[ord(c) for c in text[:10]]}")  # 显示前10个字符的Unicode编码
    
    try:
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载中文字体
        font = None
        font_paths = [
            r'C:\Windows\Fonts\msyh.ttc',      # 微软雅黑
            r'C:\Windows\Fonts\simsun.ttc',    # 宋体
            r'C:\Windows\Fonts\simhei.ttf',    # 黑体
            r'C:\Windows\Fonts\simkai.ttf',    # 楷体
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"✓ 成功加载字体: {font_path}")
                    break
                except Exception as e:
                    print(f"✗ 字体加载失败 {font_path}: {e}")
                    continue
        
        if font is None:
            try:
                font = ImageFont.load_default()
                print("✓ 使用默认字体")
            except:
                print("✗ 默认字体加载失败")
                return img
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color)
        print(f"✓ 文本绘制完成")
        
        # 转换回OpenCV格式
        result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        print(f"✓ 图像转换完成")
        return result_img
        
    except Exception as e:
        print(f"✗ 中文显示失败: {e}")
        # 如果PIL显示失败，尝试用OpenCV显示英文
        try:
            cv2.putText(img, f"Text: {text}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f"✓ 使用OpenCV英文显示作为备选")
        except Exception as e2:
            print(f"✗ OpenCV备选显示也失败: {e2}")
        return img

def main():
    """
    使用OpenCV和pyzbar实现实时二维码识别和信息显示
    """
    # 创建VideoCapture对象，参数2表示摄像头ID
    cap = cv2.VideoCapture(2)  # 如果电脑上有多个摄像头，需要调整摄像头的id，默认值0
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("摄像头已打开，二维码识别程序启动")
    print("按 'q' 键退出程序")
    print("按 'c' 键清除历史识别记录")
    
    # 存储识别到的二维码信息
    qr_history = []
    last_detection_time = 0
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        # 检查是否成功读取到画面
        if not ret:
            print("错误：无法读取摄像头画面")
            break
        
        # 获取原始画面的尺寸
        height, width = frame.shape[:2]
        
        # 将画面缩小为原来的1/2
        new_width = width // 1
        new_height = height // 1
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 使用pyzbar检测二维码
        qr_codes = pyzbar.decode(resized_frame)
        
        # 处理检测到的二维码
        for qr_code in qr_codes:
            # 获取二维码的边界框坐标
            (x, y, w, h) = qr_code.rect
            
            # 绘制二维码边框
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 获取二维码数据，处理中文编码
            qr_data = decode_qr_data(qr_code.data)
            qr_type = qr_code.type
            
            # 在二维码上方显示解码信息（支持中文）
            text = f"{qr_type}: {qr_data}"
            
            # 计算文本显示位置
            text_y = y - 30 if y > 50 else y + h + 30
            
            # 使用中文字体渲染文本
            try:
                resized_frame = put_chinese_text(resized_frame, text, (x, text_y), font_size=16, color=(0, 255, 0))
            except Exception as e:
                # 如果中文渲染失败，使用英文显示
                cv2.putText(resized_frame, f"QR: {len(qr_data)} chars", (x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 记录识别历史（避免重复记录）
            current_time = time.time()
            if current_time - last_detection_time > 1.0:  # 1秒内不重复记录
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                qr_info = {
                    'time': timestamp,
                    'type': qr_type,
                    'data': qr_data
                }
                
                # 检查是否已存在相同数据
                if not any(item['data'] == qr_data for item in qr_history):
                    qr_history.append(qr_info)
                    print(f"[{timestamp}] 检测到{qr_type}: {qr_data}")
                
                last_detection_time = current_time
        
        # 在画面上显示识别历史
        resized_frame = display_history(resized_frame, qr_history)
        
        # 显示帮助信息
        resized_frame = display_help(resized_frame)
        
        # 在窗口中显示画面
        cv2.imshow('QR Code Scanner', resized_frame)
        
        # 检查按键输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户按下 'q' 键，退出程序")
            break
        elif key == ord('c'):
            qr_history.clear()
            print("已清除识别历史记录")
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    print("摄像头已关闭，程序结束")

def display_history(frame, qr_history):
    """
    在画面上显示二维码识别历史（支持中文）
    """
    if not qr_history:
        return frame
    
    # 显示历史记录标题
    try:
        frame = put_chinese_text(frame, "二维码历史:", (10, 30), font_size=18, color=(255, 255, 255))
    except:
        cv2.putText(frame, "QR History:", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示最近的5条记录
    max_display = min(5, len(qr_history))
    for i in range(max_display):
        idx = len(qr_history) - 1 - i  # 从最新的开始显示
        qr_info = qr_history[idx]
        
        # 截断过长的数据
        display_data = qr_info['data']
        if len(display_data) > 25:
            display_data = display_data[:22] + "..."
        
        text = f"{qr_info['time']} - {display_data}"
        y_pos = 60 + i * 25
        
        # 使用中文字体显示
        try:
            frame = put_chinese_text(frame, text, (10, y_pos), font_size=14, color=(0, 255, 255))
        except:
            # 如果中文渲染失败，使用英文显示
            cv2.putText(frame, f"{qr_info['time']} - {len(display_data)} chars", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame

def display_help(frame):
    """
    显示帮助信息（支持中文）
    """
    height, width = frame.shape[:2]
    
    help_texts = [
        "按 'q' 键退出",
        "按 'c' 键清除历史"
    ]
    
    for i, text in enumerate(help_texts):
        y_pos = height - 40 + i * 25
        
        # 使用中文字体显示帮助信息
        try:
            frame = put_chinese_text(frame, text, (width - 150, y_pos), font_size=14, color=(255, 255, 255))
        except:
            # 如果中文渲染失败，使用英文显示
            english_texts = ["Press 'q' to quit", "Press 'c' to clear history"]
            cv2.putText(frame, english_texts[i], (width - 200, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

if __name__ == "__main__":
    main()