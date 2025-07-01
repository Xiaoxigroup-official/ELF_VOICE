# --- START OF FILE audio_capture.py (VAD Enhanced) ---

# audio_capture.py
import sounddevice as sd
import soundfile as sf
import numpy as np
import argparse
import os
import platform
import time
import queue # 用于在回调和主线程间传递数据
import threading # 用于 VAD 录音中的事件信号
from typing import Optional, Tuple

# VAD 相关导入
try:
    import webrtcvad
    _vad_available = True
except ImportError:
    print("警告: 'webrtcvad-wheels' 未找到。VAD 功能将不可用。")
    print("请安装: pip install webrtcvad-wheels")
    _vad_available = False
    webrtcvad = None # 避免后续引用错误

# 默认参数 (保持不变)
DEFAULT_SAMPLERATE = 16000  # 16kHz
DEFAULT_CHANNELS = 1        # 单声道
DEFAULT_DTYPE = 'int16'     # VAD 要求 int16
DEFAULT_DURATION = 5        # 默认固定录制时长
DEFAULT_FILENAME = "recorded_audio.wav"

# VAD 默认参数
DEFAULT_VAD_AGGRESSIVENESS = 1 # VAD 敏感度 (0-3)，越高越容易检测为语音
DEFAULT_VAD_FRAME_MS = 30      # VAD 处理帧时长 (ms), 必须是 10, 20, or 30
DEFAULT_VAD_SILENCE_TIMEOUT_MS = 1000 # 检测到语音后，多少毫秒静音算结束
DEFAULT_VAD_MIN_SPEECH_DURATION_MS = 250 # 最短有效语音时长 (ms)，过滤短促噪音
DEFAULT_VAD_MAX_RECORDING_S = 15 # VAD 模式最长录音时间 (秒)，防止无限录制

def check_audio_dependencies():
    """检查并提示安装音频库依赖 (包括 VAD)"""
    print("Checking audio dependencies...")
    base_ok = True
    try:
        import sounddevice
        import soundfile
        import numpy
        print("sounddevice, soundfile, numpy seem installed.")
    except ImportError:
        print("\nWarning: Required base audio libraries not found.")
        print("Please install them: pip install sounddevice soundfile numpy")
        base_ok = False

    vad_ok = True
    if not _vad_available:
        print("\nWarning: VAD library 'webrtcvad-wheels' not found.")
        print("Please install it: pip install webrtcvad-wheels")
        vad_ok = False
    else:
        print("webrtcvad seems installed.")

    if platform.system() == "Linux":
         # 这个提示可以保留，虽然 PortAudio 主要影响 sounddevice
        print("  - On Debian/Ubuntu Linux, you might still need: sudo apt-get update && sudo apt-get install libportaudio2 libasound2-dev")

    if not base_ok or not vad_ok:
        print("Installation must be done in your Python environment (e.g., activated conda env).\n")
        return False

    return True

def list_audio_devices():
    # ... (此函数保持不变) ...
    """列出可用的音频输入设备"""
    print("\nAvailable audio input devices:")
    try:
        devices = sd.query_devices()
        input_devices = [d for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        if not input_devices:
            print("No input devices found!")
            return
        for i, device in enumerate(devices):
             # 尝试解码设备名称，处理可能的编码问题
            try:
                # 尝试常见的编码
                name = device['name']
                try:
                    name = name.encode('shift_jis').decode('utf-8', errors='ignore') # Common issue on Windows Japanese env
                except:
                     try:
                        name = name.encode('latin-1').decode('utf-8', errors='ignore')
                     except:
                         # 使用原始名称，如果解码失败
                         pass # name is already the original device['name']

            except Exception: # Fallback in case of any decoding error
                 name = device['name']

            if device['max_input_channels'] > 0:
                print(f"  Index {device['index']}: {name} (Max Input Channels: {device['max_input_channels']})")
            else:
                 print(f"  Index {device['index']}: {name} (Output Only)")
        # 获取默认输入设备索引
        default_input_idx = -1
        try:
            default_indices = sd.default.device
            if isinstance(default_indices, int): # Sometimes it's just one index
                # Check if it's an input device index (not reliable way, but a heuristic)
                 if devices[default_indices]['max_input_channels'] > 0:
                     default_input_idx = default_indices
            elif isinstance(default_indices, (list, tuple)) and len(default_indices) > 0:
                # Typically (input_idx, output_idx)
                default_input_idx = default_indices[0]
            else: # Fallback query
                 query_host_api = sd.query_hostapis()
                 if query_host_api:
                    default_api_info = query_host_api[sd.default.hostapi]
                    if default_api_info and 'default_input_device' in default_api_info:
                        default_input_idx = default_api_info['default_input_device']

        except Exception as e_def:
            print(f"  Could not reliably determine default input device: {e_def}")

        if default_input_idx != -1:
            print(f"Default input device index: {default_input_idx}")
        else:
             print("Default input device index could not be determined.")
        print("-" * 20)
    except Exception as e:
        print(f"Error querying audio devices: {e}")
        print("Make sure audio drivers are installed and PortAudio library is accessible.")


def record_audio(filename=DEFAULT_FILENAME, duration=DEFAULT_DURATION, samplerate=DEFAULT_SAMPLERATE, channels=DEFAULT_CHANNELS, dtype=DEFAULT_DTYPE, device_index=None):
    """
    录制指定【固定时长】的音频并保存到文件。
    (此函数功能保持不变，用于对比或不需要 VAD 的场景)
    """
    print(f"\n--- 开始固定时长录音 ({duration}s) ---")
    # ... (函数其余部分与之前版本相同，为了简洁省略) ...
    # 只是在打印信息前加个标记
    if not check_audio_dependencies(): # 基础检查还是要做的
        # VAD 不需要在这里检查
        print("基础音频库依赖检查未通过。")
        return False, ""

    actual_device = device_index if device_index is not None else sd.default.device[0]
    print(f"\nPreparing to record audio...")
    print(f" - Mode: Fixed duration")
    print(f" - Device: Use {'default device' if device_index is None else 'device index ' + str(device_index)}")
    print(f" - Duration: {duration} seconds")
    print(f" - Sample rate: {samplerate} Hz")
    print(f" - Channels: {channels}")
    print(f" - Data type: {dtype}")
    print(f" - Output file: {filename}")

    try:
        # 检查设备是否有效
        sd.check_input_settings(device=actual_device, channels=channels, dtype=dtype, samplerate=samplerate)

        num_frames = int(duration * samplerate)

        print("\nRecording started... Speak now!")
        recording = sd.rec(num_frames, samplerate=samplerate, channels=channels, dtype=dtype, device=actual_device, blocking=True)
        print("Recording finished.")

        # 确保目录存在
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

        # 保存为 WAV 文件
        sf.write(filename, recording, samplerate)
        print(f"Audio successfully saved to '{filename}'")
        print(f"--- 固定时长录音结束 ---")
        return True, os.path.abspath(filename)

    except sd.PortAudioError as e:
        print(f"\nError during recording: {e}")
        print("Possible issues:")
        print(" - Microphone not connected or configured correctly.")
        print(" - Incorrect device index specified.")
        print(" - Permissions issue (especially on Linux).")
        print(" - Conflicting application using the audio device.")
        list_audio_devices() # 列出设备帮助调试
        return False, ""
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return False, ""


# --- 新增 VAD 录音功能 ---
def record_with_vad(
    filename: str = DEFAULT_FILENAME,
    samplerate: int = DEFAULT_SAMPLERATE,
    channels: int = DEFAULT_CHANNELS,
    dtype: str = DEFAULT_DTYPE,
    device_index: Optional[int] = None,
    vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS,
    frame_ms: int = DEFAULT_VAD_FRAME_MS,
    silence_timeout_ms: int = DEFAULT_VAD_SILENCE_TIMEOUT_MS,
    min_speech_duration_ms: int = DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
    max_recording_s: int = DEFAULT_VAD_MAX_RECORDING_S
) -> Tuple[bool, str, Optional[float]]:
    """
    使用 VAD 录制音频，直到检测到静音结束。

    Args:
        filename (str): 保存音频的文件名。
        samplerate (int): 采样率 (必须是 VAD 支持的，如 16000)。
        channels (int): 声道数 (必须是 1)。
        dtype (str): 数据类型 (必须是 'int16')。
        device_index (int, optional): 输入设备索引。
        vad_aggressiveness (int): VAD 敏感度 (0-3)。
        frame_ms (int): VAD 处理帧时长 (10, 20, or 30)。
        silence_timeout_ms (int): 语音后静音多久算结束。
        min_speech_duration_ms (int): 最短有效语音时长。
        max_recording_s (int): 最大录音时长（秒）。

    Returns:
        Tuple[bool, str, Optional[float]]: (是否成功, 文件路径, 录音时长秒数 或 None)
    """
    print(f"\n--- 开始 VAD 录音 ---")
    if not _vad_available:
        print("错误: VAD 库 (webrtcvad) 不可用。无法使用 VAD 录音。")
        return False, "", None
    if not check_audio_dependencies(): # 检查所有依赖
        return False, "", None

    # VAD 参数校验
    if samplerate not in [8000, 16000, 32000, 48000]:
        print(f"错误: VAD 不支持采样率 {samplerate}Hz。请使用 8000, 16000, 32000 或 48000。")
        return False, "", None
    if channels != 1:
        print(f"错误: VAD 当前实现仅支持单声道 (channels=1)，收到 {channels}。")
        return False, "", None
    if dtype != 'int16':
        print(f"错误: VAD 需要 'int16' 数据类型，收到 '{dtype}'。")
        return False, "", None
    if frame_ms not in [10, 20, 30]:
        print(f"错误: VAD 帧时长必须是 10, 20 或 30 ms，收到 {frame_ms}。")
        return False, "", None
    if not (0 <= vad_aggressiveness <= 3):
        print(f"错误: VAD 敏感度必须在 0 到 3 之间，收到 {vad_aggressiveness}。")
        return False, "", None

    actual_device = device_index if device_index is not None else sd.default.device[0]
    vad = webrtcvad.Vad(vad_aggressiveness)
    frames_per_buffer = int(samplerate * frame_ms / 1000) # 每回调帧数
    bytes_per_frame = np.dtype(dtype).itemsize * channels # 每帧字节数
    vad_frame_bytes = frames_per_buffer * bytes_per_frame # VAD 需要的字节数

    print(f"VAD 配置: aggressiveness={vad_aggressiveness}, frame={frame_ms}ms, silence_timeout={silence_timeout_ms}ms, min_speech={min_speech_duration_ms}ms, max_record={max_recording_s}s")
    print(f"音频流: samplerate={samplerate}, channels={channels}, dtype={dtype}, device={actual_device}, frame_size={frames_per_buffer}")

    # 共享状态变量
    audio_buffer = queue.Queue() # 存储录制的语音帧数据 (bytes)
    recording_complete = threading.Event() # 信号：录音是否结束
    is_speaking = False
    consecutive_silence_frames = 0
    total_frames_recorded = 0
    speech_frames_recorded = 0
    start_time = time.time()

    # 计算静音帧数阈值
    silence_frames_needed = silence_timeout_ms // frame_ms
    min_speech_frames = min_speech_duration_ms // frame_ms
    max_total_frames = max_recording_s * samplerate

    print(f"计算值: silence_frames_needed={silence_frames_needed}, min_speech_frames={min_speech_frames}, max_total_frames={max_total_frames}")

    def audio_callback(indata, frames, time_info, status):
        nonlocal is_speaking, consecutive_silence_frames, total_frames_recorded, speech_frames_recorded, start_time

        if status:
            print(f"音频流状态错误: {status}", flush=True)
            # 可以考虑在这里设置 recording_complete 来停止录音
            # recording_complete.set()
            return

        if recording_complete.is_set(): # 如果已经被标记为完成，则不再处理
             return

        # 确保数据是 int16 bytes
        frame_bytes = indata.tobytes()
        if len(frame_bytes) != vad_frame_bytes:
             print(f"警告: 收到预期外的数据长度 {len(frame_bytes)}, 需要 {vad_frame_bytes}", flush=True)
             # 填充或截断可能导致问题，暂时跳过此帧
             return

        try:
            is_speech = vad.is_speech(frame_bytes, samplerate)
        except Exception as e:
            print(f"VAD 处理错误: {e}", flush=True)
            # recording_complete.set() # 遇到错误可能也需要停止
            return

        total_frames_recorded += frames

        if is_speech:
            # print("语音", end="", flush=True) # 调试用
            if not is_speaking:
                print("\n检测到语音开始...", flush=True)
                is_speaking = True
                consecutive_silence_frames = 0 # 重置静音计数
                 # 如果之前有静音帧，可能需要决定是否保留（缓冲）
                 # 为了简单起见，我们只在 is_speaking 时开始存入 buffer
            audio_buffer.put(frame_bytes)
            speech_frames_recorded += frames
            consecutive_silence_frames = 0
        else:
            # print(".", end="", flush=True) # 调试用
            if is_speaking:
                # 只有在已经开始说话后，才计数静音帧并保存
                consecutive_silence_frames += 1
                audio_buffer.put(frame_bytes) # 静音也存入，避免截断
                if consecutive_silence_frames >= silence_frames_needed:
                    print(f"\n检测到足够静音 ({consecutive_silence_frames * frame_ms}ms)，准备停止...", flush=True)
                    # 检查是否有足够的语音
                    if speech_frames_recorded >= min_speech_frames * frames_per_buffer:
                         recording_complete.set() # 发送停止信号
                    else:
                         print(f"语音时长 ({speech_frames_recorded * 1000 / samplerate:.0f}ms) 不足最小要求 ({min_speech_duration_ms}ms)，继续监听...", flush=True)
                         # 重置状态，就像从未说过话一样 (或选择丢弃 buffer)
                         is_speaking = False
                         consecutive_silence_frames = 0
                         speech_frames_recorded = 0
                         # 清空 buffer 避免保存无效录音
                         while not audio_buffer.empty():
                             try: audio_buffer.get_nowait()
                             except queue.Empty: break

            # else: # 尚未开始说话，忽略静音
            #    pass

        # 检查是否达到最大录音时长
        if total_frames_recorded >= max_total_frames:
            print("\n达到最大录音时长，强制停止...", flush=True)
            if not is_speaking or speech_frames_recorded < min_speech_frames * frames_per_buffer:
                 print("最大时长内未录得有效语音。", flush=True)
                 # 清空 buffer，标记失败
                 while not audio_buffer.empty():
                      try: audio_buffer.get_nowait()
                      except queue.Empty: break
                 # 这里可以设置一个额外的失败标志，或者让后续检查 buffer 为空来判断
            recording_complete.set()

    try:
        print("\n>>> 请说话... (VAD 监听中) <<<", flush=True)
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype=dtype,
                           device=actual_device, blocksize=frames_per_buffer, # blocksize 要匹配 VAD 帧
                           callback=audio_callback):
            # 等待录音完成信号，可以设置超时
            recording_complete.wait(timeout=max_recording_s + 5) # 等待时间比最大录音长一点

            # 检查是否因为超时而退出（而不是正常VAD停止）
            if not recording_complete.is_set():
                 print("\n等待录音完成超时！可能是没有检测到语音或VAD逻辑问题。", flush=True)
                 # 确保流已停止 (尽管 with 语句会处理)
                 # sd.stop() # 一般不需要在 with 语句中手动调用
                 return False, "", None

    except sd.PortAudioError as e:
        print(f"\n音频流错误: {e}")
        list_audio_devices()
        return False, "", None
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        return False, "", None

    # --- 录音结束后处理 ---
    print("录音流已停止。处理音频数据...")
    if audio_buffer.empty():
        print("错误: 没有录制到音频数据。")
        print(f"--- VAD 录音失败 ---")
        return False, "", None

    # 从队列中取出所有数据并合并
    recorded_data_bytes = b"".join(list(audio_buffer.queue))
    final_recording = np.frombuffer(recorded_data_bytes, dtype=dtype)

    actual_duration_sec = len(final_recording) / samplerate
    print(f"实际录音时长: {actual_duration_sec:.2f} 秒")

    if actual_duration_sec < (min_speech_duration_ms / 1000.0):
         print(f"警告: 最终录音时长 ({actual_duration_sec:.2f}s) 小于最小语音时长 ({min_speech_duration_ms}ms)，可能无效。")
         # 可以选择返回 False
         # return False, "", actual_duration_sec

    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        # 保存为 WAV 文件
        sf.write(filename, final_recording, samplerate)
        saved_filepath = os.path.abspath(filename)
        print(f"VAD 录音成功保存至: '{saved_filepath}'")
        print(f"--- VAD 录音结束 ---")
        return True, saved_filepath, actual_duration_sec
    except Exception as e:
        print(f"\n保存 VAD 录音文件时出错: {e}")
        return False, "", None


# --- 主函数用于测试 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio from microphone (fixed duration or VAD).")
    parser.add_argument(
        "-f", "--filename", type=str, default=DEFAULT_FILENAME,
        help=f"Output WAV filename (default: {DEFAULT_FILENAME})"
    )
    parser.add_argument(
        "--mode", type=str, default="vad", choices=["vad", "fixed"],
        help="Recording mode: 'vad' (Voice Activity Detection) or 'fixed' duration (default: vad)"
    )

    # 固定时长参数组
    fixed_group = parser.add_argument_group('Fixed Duration Mode (--mode fixed)')
    fixed_group.add_argument(
        "-d", "--duration", type=int, default=DEFAULT_DURATION,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION})"
    )

    # VAD 参数组
    vad_group = parser.add_argument_group('VAD Mode (--mode vad)')
    vad_group.add_argument(
        "--vad-aggressiveness", type=int, default=DEFAULT_VAD_AGGRESSIVENESS, choices=[0, 1, 2, 3],
        help=f"VAD aggressiveness (0=least, 3=most sensitive, default: {DEFAULT_VAD_AGGRESSIVENESS})"
    )
    vad_group.add_argument(
        "--vad-frame-ms", type=int, default=DEFAULT_VAD_FRAME_MS, choices=[10, 20, 30],
        help=f"VAD frame duration in ms (default: {DEFAULT_VAD_FRAME_MS})"
    )
    vad_group.add_argument(
        "--vad-silence-ms", type=int, default=DEFAULT_VAD_SILENCE_TIMEOUT_MS,
        help=f"Silence duration after speech to stop recording in ms (default: {DEFAULT_VAD_SILENCE_TIMEOUT_MS})"
    )
    vad_group.add_argument(
        "--vad-min-speech-ms", type=int, default=DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
        help=f"Minimum duration of speech required to be considered valid in ms (default: {DEFAULT_VAD_MIN_SPEECH_DURATION_MS})"
    )
    vad_group.add_argument(
        "--vad-max-sec", type=int, default=DEFAULT_VAD_MAX_RECORDING_S,
        help=f"Maximum recording duration in VAD mode in seconds (default: {DEFAULT_VAD_MAX_RECORDING_S})"
    )

    # 通用参数
    parser.add_argument(
        "-r", "--rate", type=int, default=DEFAULT_SAMPLERATE,
        help=f"Sample rate in Hz (VAD requires 8k, 16k, 32k, or 48k; default: {DEFAULT_SAMPLERATE})"
    )
    parser.add_argument(
        "-c", "--channels", type=int, default=DEFAULT_CHANNELS, choices=[1], # VAD 当前限制为 1
        help=f"Number of channels (VAD requires 1; default: {DEFAULT_CHANNELS})"
    )
    parser.add_argument(
        "-t", "--type", type=str, default=DEFAULT_DTYPE, choices=['int16'], # VAD 当前限制为 int16
        help=f"Data type (VAD requires 'int16'; default: {DEFAULT_DTYPE})"
    )
    parser.add_argument(
        "-l", "--list-devices", action="store_true",
        help="List available audio devices and exit."
    )
    parser.add_argument(
        "-i", "--index", type=int, default=None,
        help="Index of the audio input device to use (use --list-devices to see options)."
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Playback the recorded audio immediately after recording for verification."
    )


    args = parser.parse_args()

    if not check_audio_dependencies():
        exit(1)

    if args.list_devices:
        list_audio_devices()
        exit(0)

    # 强制 VAD 模式下的参数符合要求
    if args.mode == 'vad':
        if args.rate not in [8000, 16000, 32000, 48000]:
            print(f"错误: --mode vad 需要 --rate 为 8000, 16000, 32000, 或 48000。当前: {args.rate}")
            exit(1)
        if args.channels != 1:
            print(f"错误: --mode vad 需要 --channels 为 1。当前: {args.channels}")
            exit(1)
        if args.type != 'int16':
            print(f"错误: --mode vad 需要 --type 为 'int16'。当前: {args.type}")
            exit(1)
        if args.vad_frame_ms not in [10, 20, 30]:
             print(f"错误: --vad-frame-ms 必须是 10, 20 或 30。当前: {args.vad_frame_ms}")
             exit(1)


    print("=" * 30)
    print(f"开始音频录制测试 (模式: {args.mode.upper()})")
    print("=" * 30)

    success = False
    saved_filepath = ""
    actual_duration = None

    if args.mode == 'vad':
        success, saved_filepath, actual_duration = record_with_vad(
            filename=args.filename,
            samplerate=args.rate,
            channels=args.channels,
            dtype=args.type,
            device_index=args.index,
            vad_aggressiveness=args.vad_aggressiveness,
            frame_ms=args.vad_frame_ms,
            silence_timeout_ms=args.vad_silence_ms,
            min_speech_duration_ms=args.vad_min_speech_ms,
            max_recording_s=args.vad_max_sec
        )
    elif args.mode == 'fixed':
        success, saved_filepath = record_audio(
            filename=args.filename,
            duration=args.duration,
            samplerate=args.rate,
            channels=args.channels,
            dtype=args.type,
            device_index=args.index
        )
        if success: actual_duration = args.duration # 固定模式下时长就是设定的值

    if success:
        print(f"\n测试成功。音频保存至: {saved_filepath}")
        if actual_duration is not None:
            print(f"录音时长约: {actual_duration:.2f} 秒")

        # 可选：立即回放录制的音频进行验证
        if args.play:
            if os.path.exists(saved_filepath):
                print("\n尝试播放录制的音频...")
                try:
                    # 读取时要用正确的 dtype，这里我们强制了 int16
                    data, fs = sf.read(saved_filepath, dtype=args.type)
                    print(f"正在播放 '{saved_filepath}' (采样率: {fs} Hz)...")
                    sd.play(data, fs, blocking=True)
                    print("播放完毕。")
                except Exception as e:
                    print(f"无法播放音频: {e}")
            else:
                print(f"错误: 文件 '{saved_filepath}' 未找到，无法播放。")
    else:
        print("\n测试失败。未能录制音频。")
        exit(1)

    print("\n音频捕获模块测试完成。")

# --- END OF FILE audio_capture.py (VAD Enhanced) ---