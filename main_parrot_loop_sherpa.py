# --- START OF FILE main_parrot_loop_sherpa.py (v3.3 - Text I/O Mode) ---

import argparse
import queue
import threading
import time
import os
import platform
import datetime
import signal
import re  # 用于语言猜测
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# from visualizer_gui import VisualizerGUI # 导入 GUI 类
# --- 导入自定义模块 ---
# (导入部分保持不变，假设之前的导入是成功的)
print("--- 开始加载模块 ---")
_all_imports_ok = True
try:
    from audio_capture import (record_with_vad, list_audio_devices,
                               check_audio_dependencies as check_audio_deps,
                               DEFAULT_SAMPLERATE, DEFAULT_CHANNELS, DEFAULT_DTYPE,
                               DEFAULT_VAD_AGGRESSIVENESS, DEFAULT_VAD_FRAME_MS,
                               DEFAULT_VAD_SILENCE_TIMEOUT_MS,
                               DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
                               DEFAULT_VAD_MAX_RECORDING_S)

    print("[OK] 成功导入 audio_capture (VAD 版本)")
except ImportError as e:
    print(f"[错误] 导入 audio_capture 失败: {e}"); _all_imports_ok = False
except Exception as e:
    print(f"[错误] 导入 audio_capture 时发生意外错误: {e}"); _all_imports_ok = False

try:
    from stt import (transcribe_audio, check_stt_dependencies as check_stt_deps,
                     DEFAULT_LOCAL_MODEL_DIR as STT_DEFAULT_MODEL_DIR,
                     DEFAULT_ASR_PROVIDER as STT_DEFAULT_PROVIDER,
                     DEFAULT_ASR_NUM_THREADS as STT_DEFAULT_NUM_THREADS,
                     DEFAULT_ASR_DECODING_METHOD as STT_DEFAULT_DECODING_METHOD)

    print("[OK] 成功导入 stt (Sherpa-ONNX Paraformer 版本)")
except ImportError as e:
    print(f"[错误] 导入 stt 失败: {e}"); _all_imports_ok = False
except AttributeError as e:
    print(f"[错误] stt.py 缺少必要的定义 (例如默认值): {e}"); _all_imports_ok = False
except Exception as e:
    print(f"[错误] 导入 stt 时发生意外错误: {e}"); _all_imports_ok = False

try:
    from llm_interaction import get_llm_response, _openai_available, _qianfan_available, load_llm_config

    print("[OK] 成功导入 llm_interaction (v2.2+)")
except ImportError as e:
    print(f"[错误] 导入 llm_interaction 失败: {e}"); _all_imports_ok = False
except AttributeError as e:
    print(f"[错误] llm_interaction.py 缺少必要的定义: {e}"); _all_imports_ok = False
except Exception as e:
    print(f"[错误] 导入 llm_interaction 时发生意外错误: {e}"); _all_imports_ok = False

try:
    from tts_sherpa_onnx import (speak_text, check_tts_dependencies as check_tts_deps,
                                 download_and_extract_model,
                                 DEFAULT_MODEL_DOWNLOAD_DIR as TTS_DEFAULT_DOWNLOAD_DIR,
                                 EN_MODEL_URL as TTS_EN_URL,
                                 ZH_MODEL_URL as TTS_ZH_URL,
                                 EN_MODEL_EXPECTED_SUBDIR as TTS_EN_SUBDIR,
                                 ZH_MODEL_EXPECTED_SUBDIR as TTS_ZH_SUBDIR,
                                 DEFAULT_SPEAKER_ID_SENTINEL, DEFAULT_SPEED_SENTINEL,
                                 DEFAULT_ZH_SPEAKER_ID, DEFAULT_EN_SPEAKER_ID,
                                 DEFAULT_ZH_SPEED, DEFAULT_EN_SPEED)

    try:
        from tts_sherpa_onnx import DEFAULT_TTS_PROVIDER, DEFAULT_TTS_NUM_THREADS, DEFAULT_TTS_SILENCE_MS
    except ImportError:
        print("警告: tts_sherpa_onnx.py 可能缺少 Provider/Threads/Silence 默认值，使用内部回退。")
        DEFAULT_TTS_PROVIDER = "cpu"
        DEFAULT_TTS_NUM_THREADS = 1
        DEFAULT_TTS_SILENCE_MS = 200
    print("[OK] 成功导入 tts_sherpa_onnx")
except ImportError as e:
    print(f"[错误] 导入 tts_sherpa_onnx 失败: {e}"); _all_imports_ok = False
except AttributeError as e:
    print(f"[错误] tts_sherpa_onnx.py 缺少必要的定义 ({e})"); _all_imports_ok = False
except Exception as e:
    print(f"[错误] 导入 tts_sherpa_onnx 时发生意外错误: {e}"); _all_imports_ok = False

print("--- 模块加载结束 ---")
if not _all_imports_ok:
    print("\n由于导入错误，程序无法继续。请检查上述错误信息并确保所有模块文件 (.py) 与此脚本在同一目录或 Python 路径中。")
    exit(1)

# --- 全局常量和配置 ---
AUDIO_QUEUE_MAX_SIZE = 1
STT_QUEUE_MAX_SIZE = 1  # Repurposed for text input if input_mode is 'text'
TTS_QUEUE_MAX_SIZE = 1
audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAX_SIZE)
stt_queue = queue.Queue(maxsize=STT_QUEUE_MAX_SIZE)
tts_queue = queue.Queue(maxsize=TTS_QUEUE_MAX_SIZE)
stop_event = threading.Event()
TEMP_AUDIO_PREFIX = "temp_vad_audio_"
processing_lock = threading.Lock()


# --- 工具函数 ---
def timestamp():
    """返回当前时间戳字符串 YYYY-MM-DD HH:MM:SS.ms"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def cleanup_temp_files(prefix=TEMP_AUDIO_PREFIX):
    """清理程序目录下所有匹配前缀的 .wav 文件"""
    print(f"{timestamp()} [清理] 开始清理前缀为 '{prefix}' 的 .wav 文件...")
    count = 0
    script_dir = Path(__file__).parent.resolve()
    try:
        for item in script_dir.iterdir():
            try:
                if item.exists() and item.is_file() and item.name.startswith(prefix) and item.name.endswith(".wav"):
                    try:
                        item.unlink()
                        print(f"{timestamp()} [清理] 删除了: {item.name}")
                        count += 1
                    except OSError as e:
                        print(f"{timestamp()} [清理] 删除 {item.name} 出错: {e}")
                    except FileNotFoundError:
                        print(f"{timestamp()} [清理] 文件 {item.name} 在尝试删除时未找到。")
            except OSError as list_e:
                print(f"{timestamp()} [清理] 检查文件 {item.name} 时出错: {list_e}")
                continue
    except FileNotFoundError:
        print(f"{timestamp()} [清理] 错误: 脚本目录 '{script_dir}' 未找到。")
    except Exception as e:
        print(f"{timestamp()} [清理] 遍历目录 '{script_dir}' 出错: {e}")
    if count == 0:
        print(f"{timestamp()} [清理] 未找到需清理文件。")
    else:
        print(f"{timestamp()} [清理] 清理完成，删除 {count} 个文件。")


def guess_language(text: str) -> Optional[str]:
    """
    根据文本内容简单猜测是中文 ('zh')、英文 ('en') 还是混合 ('mix')。
    """
    if not text: return None
    text_cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)
    if not text_cleaned: return None
    cjk_chars = 0;
    latin_chars = 0
    for char in text_cleaned:
        if '\u4e00' <= char <= '\u9fff':
            cjk_chars += 1
        elif 'a' <= char.lower() <= 'z':
            latin_chars += 1
    has_cjk = cjk_chars > 0;
    has_latin = latin_chars > 0
    if has_cjk and has_latin:
        return 'mix'
    elif has_cjk:
        return 'zh'
    elif has_latin:
        return 'en'
    else:
        return None


# --- 线程目标函数 ---

# ★★★ NEW Thread Function for Text Input ★★★
def text_input_thread(args: argparse.Namespace):
    """线程 1 (Text Mode): 从命令行获取文本输入，放入 stt_queue (repurposed)"""
    print(
        f"{timestamp()} [文本输入线程] 启动 (默认语言: {args.text_input_lang})。 输入 'exit', 'quit' 或 '退出' 来结束程序。")
    while not stop_event.is_set():
        if not processing_lock.acquire(blocking=False):
            # print(f"{timestamp()} [文本输入线程] 等待处理锁...") # 调试用
            time.sleep(0.1)  # Wait for lock
            if stop_event.is_set(): break
            continue

        # print(f"{timestamp()} [文本输入线程] 已获取处理锁。") # 调试用
        try:
            time.sleep(0.05)  # 短暂延迟，避免在文本输出模式下提示过快出现

            prompt_text = f"\n{timestamp()} [你 ({args.text_input_lang})]: "
            user_input = input(prompt_text).strip()

            if stop_event.is_set():  # 再次检查，因为 input() 可能阻塞
                if processing_lock.locked():
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                break

            if not user_input:
                print(f"{timestamp()} [文本输入线程] 输入为空，跳过。")
                if processing_lock.locked():
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                    # print(f"{timestamp()} [文本输入线程] 已释放处理锁 (因输入为空)。") # 调试用
                continue

            if user_input.lower() in ["exit", "quit", "退出"]:
                print(f"{timestamp()} [文本输入线程] 检测到退出命令。")
                stop_event.set()  # Signal all threads to stop
                if processing_lock.locked():
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                break  # Exit this thread

            guessed_lang = guess_language(user_input) or args.text_input_lang
            print(f"{timestamp()} [文本输入线程] 收到文本: '{user_input[:50]}...', 判定语言: {guessed_lang}")

            try:
                stt_queue.put((user_input, guessed_lang), block=True, timeout=2.0)
                # 锁不由本线程在此处释放，由流水线的末端释放
            except queue.Full:
                print(f"{timestamp()} [文本输入线程] 警告: stt_queue 满或超时，丢弃输入。")
                if processing_lock.locked():
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                    # print(f"{timestamp()} [文本输入线程] 已释放处理锁 (因stt_queue满)。") # 调试用

        except EOFError:  # Ctrl+D
            print(f"\n{timestamp()} [文本输入线程] 检测到 EOF (Ctrl+D)，停止。")
            stop_event.set()
            if processing_lock.locked():
                try:
                    processing_lock.release()
                except RuntimeError:
                    pass
            break
        except KeyboardInterrupt:  # 通常由主信号处理器捕获
            print(f"\n{timestamp()} [文本输入线程] 检测到 KeyboardInterrupt，设置停止事件。")
            stop_event.set()
            if processing_lock.locked():
                try:
                    processing_lock.release()
                except RuntimeError:
                    pass
            break
        except Exception as e:
            print(f"{timestamp()} [文本输入线程] 发生意外错误: {e.__class__.__name__}: {e}")
            import traceback;
            traceback.print_exc()
            if processing_lock.locked():
                try:
                    processing_lock.release()
                except RuntimeError:
                    pass
            time.sleep(1)  # 避免错误后快速循环

    print(f"{timestamp()} [文本输入线程] 停止。")
    if processing_lock.locked():  # 确保退出时释放锁
        try:
            processing_lock.release()
        except RuntimeError:
            pass


def recorder_thread(args: argparse.Namespace):
    print(
        f"{timestamp()} [录音线程] 启动 (设备: {'默认' if args.audio_device_index is None else args.audio_device_index})...")
    cycle_count = 0
    while not stop_event.is_set():
        if not processing_lock.acquire(blocking=False):
            # print(f"{timestamp()} [录音线程] 等待处理锁...") # 调试用
            time.sleep(0.1)
            if stop_event.is_set(): break
            continue

        # print(f"{timestamp()} [录音线程] 已获取处理锁。") # 调试用
        cycle_count += 1
        temp_filename = Path(f"{TEMP_AUDIO_PREFIX}{int(time.time() * 1000)}_{cycle_count}.wav").resolve()
        saved_filepath_str = None
        actual_duration = None

        print(
            f"\n{timestamp()} [录音线程] 等待语音输入 (文件: {temp_filename.name})... (VAD静音超时: {args.vad_silence_ms}ms)")

        try:
            success, saved_filepath_str, actual_duration = record_with_vad(
                filename=str(temp_filename),
                samplerate=args.samplerate, channels=args.channels, dtype=args.dtype,
                device_index=args.audio_device_index,
                vad_aggressiveness=args.vad_aggressiveness,
                frame_ms=args.vad_frame_ms,
                silence_timeout_ms=args.vad_silence_ms,
                min_speech_duration_ms=args.vad_min_speech_ms,
                max_recording_s=args.vad_max_sec
            )

            if stop_event.is_set():
                print(f"{timestamp()} [录音线程] 录音期间收到停止信号。")
                if saved_filepath_str and not args.no_cleanup:
                    try:
                        Path(saved_filepath_str).unlink(missing_ok=True)
                    except OSError:
                        pass
                if processing_lock.locked():
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                break

            if success and saved_filepath_str:
                saved_filepath = Path(saved_filepath_str)
                print(f"{timestamp()} [录音线程] VAD 录音成功: {saved_filepath.name} (时长: {actual_duration:.2f}s)")
                try:
                    audio_queue.put(str(saved_filepath), block=True, timeout=2.0)
                    # 锁不由本线程在此处释放
                except queue.Full:
                    print(f"{timestamp()} [录音线程] 警告: audio_queue 满或超时，丢弃录音。")
                    if processing_lock.locked():
                        try:
                            processing_lock.release()
                        except RuntimeError:
                            pass
                        # print(f"{timestamp()} [录音线程] 已释放处理锁 (因audio_queue满)。") # 调试用
                    if saved_filepath and not args.no_cleanup and saved_filepath.exists():
                        try:
                            saved_filepath.unlink(missing_ok=True)
                        except OSError as e:
                            print(f"{timestamp()} [录音线程] 清理文件 {saved_filepath.name} 时出错: {e}")
            else:
                print(f"{timestamp()} [录音线程] VAD 录音失败或未录得有效语音。")
                if processing_lock.locked():
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                    # print(f"{timestamp()} [录音线程] 已释放处理锁 (因录音失败)。") # 调试用
                time.sleep(0.2)

        except Exception as e:
            print(f"{timestamp()} [录音线程] 发生意外错误: {e.__class__.__name__}: {e}")
            import traceback;
            traceback.print_exc()
            if processing_lock.locked():
                try:
                    processing_lock.release()
                except RuntimeError:
                    pass
            # print(f"{timestamp()} [录音线程] 已释放处理锁 (因异常)。") # 调试用
            if saved_filepath_str and Path(saved_filepath_str).exists() and not args.no_cleanup:
                try:
                    Path(saved_filepath_str).unlink(missing_ok=True)
                except OSError:
                    pass
            time.sleep(1)

    print(f"{timestamp()} [录音线程] 停止。")
    if processing_lock.locked():  # 确保退出时释放锁
        try:
            processing_lock.release()
        except RuntimeError:
            pass


def stt_processor_thread(args: argparse.Namespace):
    print(f"{timestamp()} [STT线程] 启动 (模型: {args.stt_model_dir})...")
    while True:
        audio_path_str = None
        audio_path = None
        try:
            audio_path_str = audio_queue.get(block=True, timeout=0.5)

            if audio_path_str is None and stop_event.is_set():
                print(f"{timestamp()} [STT线程] 收到停止信号 (None)，退出。")
                break
            if not audio_path_str:
                continue

            audio_path = Path(audio_path_str)
            print(f"\n{timestamp()} [STT线程] 从 audio_queue 收到: {audio_path.name}")

            start_time = time.time()
            transcribed_text, detected_lang_reported = transcribe_audio(
                audio_path=str(audio_path),
                model_dir=args.stt_model_dir,
                provider=args.stt_provider,
                num_threads=args.stt_num_threads,
                decoding_method=args.stt_decoding_method
            )
            end_time = time.time()

            if stop_event.is_set(): break

            if transcribed_text is not None:
                print(
                    f"{timestamp()} [STT线程] 转录成功 (耗时 {end_time - start_time:.2f}s): '{transcribed_text}', Reported Lang: {detected_lang_reported or '未知'}")
                try:
                    stt_queue.put((transcribed_text, detected_lang_reported), block=True, timeout=2.0)
                    # 锁不由本线程在此处释放
                except queue.Full:
                    print(f"{timestamp()} [STT线程] 警告: stt_queue 满或超时，丢弃结果。")
                    if processing_lock.locked():  # 如果无法入队，流水线中断，释放锁
                        try:
                            processing_lock.release()
                        except RuntimeError:
                            pass
                        # print(f"{timestamp()} [STT线程] 已释放处理锁 (因 stt_queue 满)。") # 调试用
            else:
                print(f"{timestamp()} [STT线程] 转录失败或结果为空 (耗时 {end_time - start_time:.2f}s)。")
                if processing_lock.locked():  # STT失败，流水线中断，释放锁
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                    # print(f"{timestamp()} [STT线程] 已释放处理锁 (因 STT 失败)。") # 调试用

            audio_queue.task_done()
            if audio_path and not args.no_cleanup and audio_path.exists():
                try:
                    audio_path.unlink(missing_ok=True)
                except OSError as e:
                    print(f"{timestamp()} [STT线程] 清理文件 {audio_path.name} 时出错: {e}")

        except queue.Empty:
            if stop_event.is_set():
                print(f"{timestamp()} [STT线程] 收到停止信号且 audio_queue 为空，退出。")
                break
            continue
        except Exception as e:
            print(f"{timestamp()} [STT线程] 发生意外错误: {e.__class__.__name__}: {e}")
            import traceback;
            traceback.print_exc()
            try:
                if audio_path_str: audio_queue.task_done()
            except (ValueError, RuntimeError):
                pass
            if processing_lock.locked():  # 异常，流水线中断，释放锁
                try:
                    processing_lock.release()
                except RuntimeError:
                    pass
            # print(f"{timestamp()} [STT线程] 已释放处理锁 (因异常)。") # 调试用
            if audio_path and audio_path.exists() and not args.no_cleanup:
                try:
                    audio_path.unlink(missing_ok=True)
                except OSError:
                    pass
            time.sleep(1)

    print(f"{timestamp()} [STT线程] 停止。")


# ★★★ MODIFIED llm_processor_thread 函数 ★★★
def llm_processor_thread(args: argparse.Namespace):
    """线程 3: LLM 处理线程，从 stt_queue 获取文本，调用 LLM API。
       如果 output_mode=='text'，则打印响应并释放锁。
       否则，放入 tts_queue。
    """
    print(f"{timestamp()} [LLM线程] 启动 (模式: {args.llm_mode}, 输出模式: {args.output_mode})...")
    while True:
        stt_result = None
        try:
            stt_result = stt_queue.get(block=True, timeout=0.5)

            if stt_result is None and stop_event.is_set():
                print(f"{timestamp()} [LLM线程] 收到停止信号 (None)，退出。")
                break
            if not stt_result and not stop_event.is_set():
                continue

            transcribed_text, detected_lang_reported = stt_result
            print(
                f"\n{timestamp()} [LLM线程] 从 stt_queue 收到: '{transcribed_text[:50]}...', Reported Lang: {detected_lang_reported}")

            llm_provider_arg = None
            if args.llm_mode == 'local':
                llm_provider_arg = 'local'
            elif args.llm_mode == 'official':
                llm_provider_arg = args.llm_provider
            elif args.llm_mode == 'echo':
                llm_provider_arg = None

            print(
                f"{timestamp()} [LLM线程] 调用 LLM: mode='{args.llm_mode}', provider='{llm_provider_arg}', model='{args.llm_model_name or '默认'}'")
            start_time = time.time()
            response_text = get_llm_response(
                prompt=transcribed_text, mode=args.llm_mode,
                provider=llm_provider_arg, model_name=args.llm_model_name
            )
            end_time = time.time()

            if stop_event.is_set(): break

            if response_text is None or response_text.startswith("错误:"):
                print(f"{timestamp()} [LLM线程] 获取响应失败: {response_text}")
                if processing_lock.locked():  # LLM失败，流水线中断，释放锁
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                    # print(f"{timestamp()} [LLM线程] 已释放处理锁 (因 LLM 失败)。") # 调试用
                stt_queue.task_done()
                continue

            print(
                f"{timestamp()} [LLM线程] 获取响应成功 (耗时 {end_time - start_time:.2f}s): '{response_text[:50]}...'")

            text_to_process = response_text
            if args.llm_mode == "echo" and text_to_process.startswith("Echo: "):
                text_to_process = text_to_process[len("Echo: "):].strip()

            if not text_to_process:
                print(f"{timestamp()} [LLM线程] 响应为空文本。")
                if processing_lock.locked():  # 空响应，流水线中断，释放锁
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                    # print(f"{timestamp()} [LLM线程] 已释放处理锁 (因 LLM 空响应)。") # 调试用
                stt_queue.task_done()
                continue

            # ★★★ MODIFIED: Conditional output ★★★
            if args.output_mode == 'text':
                print(f"\n{timestamp()} [LLM输出]: {text_to_process}")
                # 这是文本输出模式的结束点，释放锁
                if processing_lock.locked():
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                    # print(f"{timestamp()} [LLM线程] 已释放处理锁 (文本输出模式)。") # 调试用
                else:
                    print(f"{timestamp()} [LLM线程] 警告: 文本输出模式结束时处理锁未被持有！")

            elif args.output_mode == 'voice':
                try:
                    tts_queue.put((text_to_process, detected_lang_reported), block=True, timeout=2.0)
                    # 锁不由本线程在此处释放 (语音输出模式)
                except queue.Full:
                    print(f"{timestamp()} [LLM线程] 警告: tts_queue 满或超时，丢弃任务。")
                    if processing_lock.locked():  # 无法入队，流水线中断，释放锁
                        try:
                            processing_lock.release()
                        except RuntimeError:
                            pass
                        # print(f"{timestamp()} [LLM线程] 已释放处理锁 (因 TTS 队列满)。") # 调试用

            stt_queue.task_done()

        except queue.Empty:
            if stop_event.is_set():
                print(f"{timestamp()} [LLM线程] 收到停止信号且 stt_queue 为空，退出。")
                break
            continue
        except Exception as e:
            print(f"{timestamp()} [LLM线程] 发生意外错误: {e.__class__.__name__}: {e}")
            import traceback;
            traceback.print_exc()
            try:
                if stt_result: stt_queue.task_done()
            except (ValueError, RuntimeError):
                pass
            if processing_lock.locked():  # 异常，流水线中断，释放锁
                try:
                    processing_lock.release()
                except RuntimeError:
                    pass
            # print(f"{timestamp()} [LLM线程] 已释放处理锁 (因异常)。") # 调试用
            time.sleep(1)

    print(f"{timestamp()} [LLM线程] 停止。")


def tts_processor_thread(args: argparse.Namespace, tts_models: Dict[str, Optional[str]]):
    """线程 4: TTS 处理与播放线程，从 tts_queue 获取文本，并负责释放处理锁"""
    print(f"{timestamp()} [TTS线程] 启动...")
    tts_model_dir_en = tts_models.get('en')
    tts_model_dir_zh = tts_models.get('zh')
    print(
        f"{timestamp()} [TTS线程] 可用模型 - EN: {'是' if tts_model_dir_en else '否'}, ZH/MIX: {'是' if tts_model_dir_zh else '否'}")

    while True:
        tts_task = None
        try:
            tts_task = tts_queue.get(block=True, timeout=0.5)

            if tts_task is None and stop_event.is_set():
                print(f"{timestamp()} [TTS线程] 收到停止信号 (None)，退出。")
                break
            if not tts_task and not stop_event.is_set(): continue

            text_to_speak, detected_lang_reported = tts_task
            print(
                f"\n{timestamp()} [TTS线程] 从 tts_queue 收到: '{text_to_speak[:50]}...', Reported Lang: {detected_lang_reported}")

            guessed_lang = guess_language(text_to_speak)
            print(f"{timestamp()} [TTS线程] 猜测语言 (基于响应文本): {guessed_lang or '不确定'}")
            selected_model_dir = None;
            selected_lang_desc = "未知"

            if guessed_lang == 'mix':
                selected_lang_desc = "混合"
                if tts_model_dir_zh:
                    selected_model_dir = tts_model_dir_zh; selected_lang_desc = "混合 (使用中英模型)"
                elif tts_model_dir_en:
                    selected_model_dir = tts_model_dir_en; selected_lang_desc = "英文 (混合内容，但无中英模型，回退)"
            elif guessed_lang == 'zh':
                selected_lang_desc = "中文"
                if tts_model_dir_zh:
                    selected_model_dir = tts_model_dir_zh
                elif tts_model_dir_en:
                    selected_model_dir = tts_model_dir_en; selected_lang_desc = "英文 (中文内容，但无中文模型，回退)"
            elif guessed_lang == 'en':
                selected_lang_desc = "英文"
                if tts_model_dir_en:
                    selected_model_dir = tts_model_dir_en
                elif tts_model_dir_zh:
                    selected_model_dir = tts_model_dir_zh; selected_lang_desc = "中英 (英文内容，但无英文模型，回退)"
            else:
                print(f"{timestamp()} [TTS线程] 无法根据内容猜测语言，尝试使用 STT 报告语言: {detected_lang_reported}")
                fallback_lang = None
                if detected_lang_reported and detected_lang_reported.lower().startswith('zh'):
                    fallback_lang = 'zh'
                elif detected_lang_reported and detected_lang_reported.lower().startswith('en'):
                    fallback_lang = 'en'
                if fallback_lang == 'zh':
                    selected_lang_desc = "中文 (基于 STT 回退)"
                    if tts_model_dir_zh:
                        selected_model_dir = tts_model_dir_zh
                    elif tts_model_dir_en:
                        selected_model_dir = tts_model_dir_en
                elif fallback_lang == 'en':
                    selected_lang_desc = "英文 (基于 STT 回退)"
                    if tts_model_dir_en:
                        selected_model_dir = tts_model_dir_en
                    elif tts_model_dir_zh:
                        selected_model_dir = tts_model_dir_zh
                else:
                    print(f"{timestamp()} [TTS线程] STT 报告语言也无效，最终默认尝试中文模型。")
                    selected_lang_desc = "中文 (默认)"
                    if tts_model_dir_zh:
                        selected_model_dir = tts_model_dir_zh
                    elif tts_model_dir_en:
                        selected_model_dir = tts_model_dir_en
            if selected_model_dir: print(
                f"{timestamp()} [TTS线程] 选择模型 ({selected_lang_desc}): {Path(selected_model_dir).name}")

            if not selected_model_dir:
                print(f"{timestamp()} [TTS线程] 错误: 最终无法选择有效的 TTS 模型来处理 '{selected_lang_desc}' 内容。")
                tts_queue.task_done()
                if processing_lock.locked():  # 无法TTS，流水线中断，释放锁
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
                    # print(f"{timestamp()} [TTS线程] 已释放处理锁 (因无 TTS 模型)。") # 调试用
                continue

            final_speaker_id = args.tts_speaker_id;
            final_speed = args.tts_speed
            print(
                f"{timestamp()} [TTS线程] 最终调用 TTS: 模型='{Path(selected_model_dir).name}', 语言判定='{selected_lang_desc}', SID={final_speaker_id}, Speed={final_speed}")

            start_time = time.time()
            speak_success = speak_text(
                text=text_to_speak, model_dir=selected_model_dir, speaker_id=final_speaker_id, speed=final_speed,
                provider=args.tts_provider, num_threads=args.tts_num_threads, add_silence_ms=args.tts_silence_ms
            )
            end_time = time.time()

            if stop_event.is_set(): break

            if speak_success:
                print(f"{timestamp()} [TTS线程] 朗读成功 (耗时 {end_time - start_time:.2f}s)")
            else:
                print(f"{timestamp()} [TTS线程] 朗读失败 (耗时 {end_time - start_time:.2f}s)")

            tts_queue.task_done()

            # 这是语音输出模式的结束点，释放锁
            if processing_lock.locked():
                try:
                    processing_lock.release();  # print(f"{timestamp()} [TTS线程] 已释放处理锁 (TTS 完成)。") # 调试用
                except RuntimeError:
                    print(f"{timestamp()} [TTS线程] 警告: 尝试释放锁时发现未被持有。")
            else:
                print(f"{timestamp()} [TTS线程] 警告: TTS 完成时处理锁未被持有！")

        except queue.Empty:
            if stop_event.is_set():
                print(f"{timestamp()} [TTS线程] 收到停止信号且 tts_queue 为空，退出。")
                break
            continue
        except Exception as e:
            print(f"{timestamp()} [TTS线程] 发生意外错误: {e.__class__.__name__}: {e}")
            import traceback;
            traceback.print_exc()
            try:
                if tts_task: tts_queue.task_done()
            except (ValueError, RuntimeError):
                pass
            if processing_lock.locked():  # 异常，流水线中断，释放锁
                try:
                    processing_lock.release()
                except RuntimeError:
                    pass
            # print(f"{timestamp()} [TTS线程] 已释放处理锁 (因异常)。") # 调试用
            time.sleep(1)

    print(f"{timestamp()} [TTS线程] 停止。")
    if processing_lock.locked():  # 确保退出时释放锁
        try:
            processing_lock.release()
        except RuntimeError:
            pass


# --- 信号处理函数 ---
# ★★★ MODIFIED signal_handler ★★★
def signal_handler(sig, frame, active_queues: List[queue.Queue]):
    print(f"\n{timestamp()} [信号处理] 收到信号 {signal.Signals(sig).name}，设置停止事件...")
    if not stop_event.is_set():
        stop_event.set()
        print(f"{timestamp()} [信号处理] 尝试向活动队列发送停止信号 (None)...")
        for q in active_queues:
            try:
                q.put_nowait(None)
            except queue.Full:
                print(f"{timestamp()} [信号处理] 队列 {q} 已满，无法放入 None。")
            except Exception as e:
                print(f"{timestamp()} [信号处理] 放入队列 {q} 时出错: {e}")


# --- 主函数 ---
if __name__ == "__main__":
    # --- 参数定义 ---
    parser = argparse.ArgumentParser(
        description="并发执行 VAD录音/文本输入 -> Sherpa-ONNX STT -> LLM/Echo -> Sherpa-ONNX TTS/文本输出 的流程 (v3.3)。",
        # ★★★ MODIFIED ★★★
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    audio_group = parser.add_argument_group('音频录制 (VAD) 参数 (仅当 --input-mode=voice)')
    stt_group = parser.add_argument_group('STT (Sherpa-ONNX Paraformer) 参数 (仅当 --input-mode=voice)')
    llm_group = parser.add_argument_group('LLM/Echo 参数')
    tts_group = parser.add_argument_group('TTS (Sherpa-ONNX) 参数 (仅当 --output-mode=voice)')
    # ★★★ NEW ★★★
    control_flow_group = parser.add_argument_group('控制流参数')
    general_group = parser.add_argument_group('通用参数')

    # Audio Args
    audio_group.add_argument("--list-audio-devices", action="store_true", help="列出音频输入设备并退出。")
    audio_group.add_argument("--audio-device-index", type=int, default=None,
                             help="指定音频输入设备索引 (默认自动选择)。")
    audio_group.add_argument("--samplerate", type=int, default=DEFAULT_SAMPLERATE,
                             help=f"录音采样率 (Hz)。VAD/STT 通常要求 {DEFAULT_SAMPLERATE}。")
    audio_group.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, choices=[1],
                             help="录音声道数 (VAD/STT 通常要求 1)。")
    audio_group.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=['int16'],
                             help="录音数据类型 (VAD/STT 通常要求 'int16')。")
    audio_group.add_argument("--vad-aggressiveness", type=int, default=DEFAULT_VAD_AGGRESSIVENESS, choices=[0, 1, 2, 3],
                             help="VAD 敏感度 (0-3)。")
    audio_group.add_argument("--vad-frame-ms", type=int, default=DEFAULT_VAD_FRAME_MS, choices=[10, 20, 30],
                             help="VAD 帧时长 (ms)。")
    audio_group.add_argument("--vad-silence-ms", type=int, default=DEFAULT_VAD_SILENCE_TIMEOUT_MS,
                             help="语音后静音多久停止 (ms)。")
    audio_group.add_argument("--vad-min-speech-ms", type=int, default=DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
                             help="最短有效语音时长 (ms)。")
    audio_group.add_argument("--vad-max-sec", type=int, default=DEFAULT_VAD_MAX_RECORDING_S,
                             help="VAD 最长录音时间 (s)。")

    # STT Args
    stt_group.add_argument("--stt-model-dir", type=str, default=STT_DEFAULT_MODEL_DIR,
                           help="【必需】指定 Sherpa-ONNX STT (Paraformer) 模型目录路径。")
    stt_group.add_argument("--stt-provider", type=str, default=STT_DEFAULT_PROVIDER, choices=["cpu", "cuda", "coreml"],
                           help="STT ONNX Runtime provider。")
    stt_group.add_argument("--stt-num-threads", type=int, default=STT_DEFAULT_NUM_THREADS, help="STT CPU 推理线程数。")
    stt_group.add_argument("--stt-decoding-method", type=str, default=STT_DEFAULT_DECODING_METHOD,
                           help="STT 解码方法 (通常为 greedy_search)。")

    # LLM Args
    llm_group.add_argument("--llm-mode", type=str, default="echo", choices=["local", "official", "echo"],
                           help="LLM 交互模式 ('local' 使用本地兼容API, 'official' 使用云服务API, 'echo' 复述输入)。")
    llm_group.add_argument("--llm-provider", type=str, default=None,
                           help="当 --llm-mode=official 时【必需】，指定 API 提供商 (如 'baidu', 'deepseek', 'zhipu', 'moonshot', 'openai')。")
    llm_group.add_argument("--llm-model-name", type=str, default=None,
                           help="指定 LLM 模型名称 (覆盖配置文件中的默认模型)。适用于 'local' 和 'official' 模式。")

    # TTS Args
    tts_group.add_argument("--tts-model-dir-en", type=str, default=None,
                           help="【可选】指定英文 TTS 模型目录路径。若不指定，将尝试下载/查找默认模型。")
    tts_group.add_argument("--tts-model-dir-zh", type=str, default=None,
                           help="【可选】指定中文 TTS 模型(推荐 vits-melo-tts-zh_en 用于中英混合)目录路径。若不指定，将尝试下载/查找默认模型。")
    tts_group.add_argument("--tts-download-dir", type=str, default=TTS_DEFAULT_DOWNLOAD_DIR,
                           help="TTS 模型下载和查找的根目录。")
    tts_group.add_argument("--tts-speaker-id", type=int, default=DEFAULT_SPEAKER_ID_SENTINEL,
                           help=f"TTS 说话人 ID (-1 为自动选择默认)。MeloTTS中英模型 ID 可能不同(如中0,英1)。")
    tts_group.add_argument("--tts-speed", type=float, default=DEFAULT_SPEED_SENTINEL,
                           help=f"TTS 语速 (-1.0 为自动选择默认)。建议范围 0.5-2.0。")
    tts_group.add_argument("--tts-provider", type=str, default=DEFAULT_TTS_PROVIDER, choices=["cpu", "cuda", "coreml"],
                           help="TTS ONNX Runtime provider。")
    tts_group.add_argument("--tts-num-threads", type=int, default=DEFAULT_TTS_NUM_THREADS, help="TTS CPU 推理线程数。")
    tts_group.add_argument("--tts-silence-ms", type=int, default=DEFAULT_TTS_SILENCE_MS,
                           help="TTS 合成语音末尾添加的静音时长 (ms)。0 为不添加。")

    # ★★★ NEW Control Flow Arguments ★★★
    control_flow_group.add_argument(
        "--input-mode", type=str, default="voice", choices=["voice", "text"],
        help="输入模式: 'voice' (麦克风输入) 或 'text' (命令行文本输入)。"
    )
    control_flow_group.add_argument(
        "--output-mode", type=str, default="voice", choices=["voice", "text"],
        help="输出模式: 'voice' (语音合成播放) 或 'text' (命令行文本输出)。"
    )
    control_flow_group.add_argument(
        "--text-input-lang", type=str, default="zh", choices=["zh", "en", "mix"],
        help="当 --input-mode=text 时，如果 guess_language 失败，默认使用的语言 (主要影响 TTS 模型选择)。"
    )

    # General Args
    general_group.add_argument("--no-cleanup", action="store_true", help="不删除录音产生的临时 .wav 文件。")
    general_group.add_argument("--skip-checks", action="store_true", help="跳过启动时的依赖检查（不推荐）。")

    args = parser.parse_args()

    if args.llm_mode == 'official' and not args.llm_provider:
        parser.error("--llm-mode=official 时，必须指定 --llm-provider")

    # --- 依赖检查 (★★★ MODIFIED ★★★) ---
    if not args.skip_checks:
        print("\n--- 检查所有依赖 ---")
        all_deps_ok = True
        if args.input_mode == 'voice':
            print("检查语音输入依赖...")
            if not check_audio_deps(): all_deps_ok = False
            if not check_stt_deps(): all_deps_ok = False

        print("检查 LLM 依赖...")
        print(f"  'openai' 库: {'可用' if _openai_available else '未找到 (需要用于 local/openai兼容模式)'}")
        print(f"  'qianfan' 库: {'可用' if _qianfan_available else '未找到 (需要用于 baidu 模式)'}")

        if args.output_mode == 'voice':
            print("检查语音输出依赖...")
            if not check_tts_deps(): all_deps_ok = False

        if args.llm_mode == 'local' and not _openai_available:
            print("警告: LLM 模式为 'local' 但 'openai' 库不可用。'local' 模式将无法工作。")

        if not all_deps_ok:
            print("\n依赖检查失败或存在警告，请检查安装。程序退出。")
            exit(1)
        print("--- 所有必要依赖检查通过 (根据所选模式) ---")
    else:
        print("\n--- 跳过依赖检查 ---")

    # --- 处理音频设备列表 (★★★ MODIFIED ★★★) ---
    if args.input_mode == 'voice' and args.list_audio_devices:
        list_audio_devices()
        exit(0)

    # --- 准备 STT 模型目录 (★★★ MODIFIED ★★★) ---
    stt_model_path = None
    if args.input_mode == 'voice':
        stt_model_path = Path(args.stt_model_dir)
        if not stt_model_path.is_dir():
            print(f"错误: 指定的 STT 模型目录不存在或无效: {args.stt_model_dir}")
            exit(1)
        print(f"确认 STT 模型目录: {stt_model_path.resolve()}")

    # --- 准备 TTS 模型目录 (★★★ MODIFIED ★★★) ---
    tts_models_found: Dict[str, Optional[str]] = {'en': None, 'zh': None}
    if args.output_mode == 'voice':
        tts_download_base_path = Path(args.tts_download_dir)
        if args.tts_model_dir_en:
            en_path = Path(args.tts_model_dir_en)
            if en_path.is_dir():
                tts_models_found['en'] = str(en_path.resolve()); print(
                    f"使用指定英文 TTS 模型: {tts_models_found['en']}")
            else:
                print(
                    f"警告: 指定的英文 TTS 目录无效: {args.tts_model_dir_en}，将尝试查找或下载默认模型。"); args.tts_model_dir_en = None
        if not args.tts_model_dir_en and TTS_EN_URL and TTS_EN_SUBDIR:
            expected_en_path = tts_download_base_path / TTS_EN_SUBDIR
            if expected_en_path.is_dir():
                tts_models_found['en'] = str(expected_en_path.resolve()); print(
                    f"自动找到默认英文 TTS 模型: {tts_models_found['en']}")
            else:
                print(f"未找到默认英文模型 '{TTS_EN_SUBDIR}'，尝试下载 ({TTS_EN_URL})...")
                downloaded_path = download_and_extract_model(TTS_EN_URL, args.tts_download_dir, TTS_EN_SUBDIR)
                if downloaded_path:
                    tts_models_found['en'] = str(downloaded_path.resolve())
                else:
                    print(f"下载默认英文 TTS 模型失败。英文 TTS 将不可用。")
        if args.tts_model_dir_zh:
            zh_path = Path(args.tts_model_dir_zh)
            if zh_path.is_dir():
                tts_models_found['zh'] = str(zh_path.resolve()); print(
                    f"使用指定中文/混合 TTS 模型: {tts_models_found['zh']}")
            else:
                print(
                    f"警告: 指定的中文/混合 TTS 目录无效: {args.tts_model_dir_zh}，将尝试查找或下载默认模型。"); args.tts_model_dir_zh = None
        if not args.tts_model_dir_zh and TTS_ZH_URL and TTS_ZH_SUBDIR:
            expected_zh_path = tts_download_base_path / TTS_ZH_SUBDIR
            if expected_zh_path.is_dir():
                tts_models_found['zh'] = str(expected_zh_path.resolve()); print(
                    f"自动找到默认中文/混合 TTS 模型: {tts_models_found['zh']}")
            else:
                print(f"未找到默认中文/混合模型 '{TTS_ZH_SUBDIR}'，尝试下载 ({TTS_ZH_URL})...")
                downloaded_path = download_and_extract_model(TTS_ZH_URL, args.tts_download_dir, TTS_ZH_SUBDIR)
                if downloaded_path:
                    tts_models_found['zh'] = str(downloaded_path.resolve())
                else:
                    print(f"下载默认中文/混合 TTS 模型失败。中文/混合 TTS 将不可用。")
        if not tts_models_found['en'] and not tts_models_found['zh']:
            print("\n错误：未能准备任何有效的 TTS 模型（英文或中文/混合）。程序退出。")
            exit(1)

    # --- 打印最终配置摘要 (★★★ MODIFIED ★★★) ---
    print(f"\n{'=' * 15} 并发语音助手配置 (v3.3 - Text I/O) {'=' * 15}")
    print(f"  平台: {platform.system()} ({platform.machine()})")
    print(f"  输入模式: {args.input_mode}" + (
        f" (文本输入默认语言: {args.text_input_lang})" if args.input_mode == 'text' else ""))
    print(f"  输出模式: {args.output_mode}")

    if args.input_mode == 'voice':
        print(f"  [录音 VAD]")
        print(
            f"    设备: {'默认' if args.audio_device_index is None else args.audio_device_index}, 采样率: {args.samplerate}, 声道: {args.channels}, 类型: {args.dtype}")
        print(
            f"    VAD 参数: aggr={args.vad_aggressiveness}, frame={args.vad_frame_ms}ms, silence={args.vad_silence_ms}ms, min_speech={args.vad_min_speech_ms}ms, max_rec={args.vad_max_sec}s")
        print(f"  [STT Sherpa-ONNX Paraformer]")
        print(f"    模型目录: {args.stt_model_dir}")
        print(f"    Provider: {args.stt_provider}, 线程: {args.stt_num_threads}, 解码: {args.stt_decoding_method}")

    print(f"  [LLM/Echo]")
    print(f"    模式: {args.llm_mode}")
    if args.llm_mode == 'local':
        llm_config = load_llm_config().get('local', {})
        model_display = args.llm_model_name or llm_config.get('default_model', '未在配置中指定')
        base_url_display = llm_config.get('base_url') or os.environ.get("OPENAI_BASE_URL", "未设置")
        print(f"    本地模型名 (命令行 > 配置): {model_display}")
        print(f"    API Base URL (配置 > 环境变量): {base_url_display}")
        if not base_url_display or base_url_display == "未设置": print(
            "    警告: LLM local 模式通常需要配置 API Base URL!")
        print(f"    依赖: 'openai' 库 {'可用' if _openai_available else '不可用!'}")
    elif args.llm_mode == 'official':
        llm_config = load_llm_config().get(args.llm_provider, {})
        model_display = args.llm_model_name or llm_config.get('default_model', '未在配置中指定')
        print(f"    Provider: {args.llm_provider}")
        print(f"    模型名 (命令行 > 配置): {model_display}")
        if args.llm_provider == 'baidu':
            print(f"    依赖: 'qianfan' 库 {'可用' if _qianfan_available else '不可用!'}")
        else:
            print(f"    依赖: 'openai' 库 {'可用' if _openai_available else '不可用!'}")

    if args.output_mode == 'voice':
        print(f"  [TTS Sherpa-ONNX]")
        print(f"    英文模型: {tts_models_found['en'] or '未配置/不可用'}")
        print(f"    中文/混合模型: {tts_models_found['zh'] or '未配置/不可用'}")
        print(f"    * TTS 语言选择将基于响应文本猜测 (优先处理混合) *")
        print(
            f"    Speaker ID: {args.tts_speaker_id if args.tts_speaker_id != DEFAULT_SPEAKER_ID_SENTINEL else '自动'}")
        print(f"    语速: {args.tts_speed if args.tts_speed != DEFAULT_SPEED_SENTINEL else '自动'}")
        print(f"    Provider: {args.tts_provider}, 线程: {args.tts_num_threads}, 末尾静音: {args.tts_silence_ms}ms")

    print(f"  [通用]")
    print(f"    清理临时文件: {'否' if args.no_cleanup else ('是' if args.input_mode == 'voice' else '不适用')}")
    print("=" * 60)

    # --- 注册信号处理器 (★★★ MODIFIED ★★★) ---
    # We need to know which queues are active to send stop signals
    active_queues_for_signal: List[queue.Queue] = []
    if args.input_mode == 'voice': active_queues_for_signal.append(audio_queue)
    active_queues_for_signal.append(stt_queue)  # Always active (either for STT or text input)
    if args.output_mode == 'voice': active_queues_for_signal.append(tts_queue)

    # Use a lambda to pass active_queues to the handler
    custom_signal_handler = lambda sig, frame: signal_handler(sig, frame, active_queues_for_signal)
    signal.signal(signal.SIGINT, custom_signal_handler)
    signal.signal(signal.SIGTERM, custom_signal_handler)
    if platform.system() == "Windows":
        try:
            signal.signal(signal.SIGBREAK, custom_signal_handler)  # type: ignore
        except AttributeError:
            pass

    # --- 创建并启动线程 (★★★ MODIFIED ★★★) ---
    threads: List[threading.Thread] = []
    all_threads_started = False
    try:
        print(f"{timestamp()} [主线程] 准备启动所有工作线程...")
        thread_definitions: List[Tuple[Any, str, Tuple]] = []

        if args.input_mode == 'voice':
            thread_definitions.append((recorder_thread, "RecorderThread", (args,)))
            thread_definitions.append((stt_processor_thread, "STTThread", (args,)))
        elif args.input_mode == 'text':
            thread_definitions.append((text_input_thread, "TextInputThread", (args,)))

        thread_definitions.append((llm_processor_thread, "LLMThread", (args,)))

        if args.output_mode == 'voice':
            thread_definitions.append((tts_processor_thread, "TTSThread", (args, tts_models_found)))

        for target_func, name, target_args in thread_definitions:
            t = threading.Thread(target=target_func, args=target_args, name=name, daemon=True)
            threads.append(t)  # Add to our list of active threads
            t.start()
            print(f"{timestamp()} [主线程] 已启动线程: {t.name}")

        all_threads_started = True  # This flag is mostly for historical reasons now
        print(f"\n{timestamp()} [主线程] 所有选定模式的线程已启动。流水线开始运行...")
        if args.input_mode == 'voice':
            print(f"{timestamp()} [主线程] （语音输入模式）请说话...")
        else:  # text input mode
            print(f"{timestamp()} [主线程] （文本输入模式）请在下方提示符后输入文本并按回车。")
        print(f"{timestamp()} [主线程] 按 Ctrl+C 或发送 SIGTERM/SIGBREAK 信号来优雅地停止程序。")
        if args.input_mode == 'text':
            print(f"{timestamp()} [主线程] 在文本输入提示符后输入 'exit', 'quit' 或 '退出' 也可以停止程序。")

        # 主循环等待停止信号或线程异常退出
        while not stop_event.is_set():
            if threads and any(not t.is_alive() for t in threads):  # Check if any of *our active* threads died
                print(f"\n{timestamp()} [主线程] 警告: 检测到有工作线程意外退出！将停止所有线程。")
                if not stop_event.is_set():
                    custom_signal_handler(signal.SIGTERM, None)  # Trigger shutdown
                break
            time.sleep(0.5)

        print(f"\n{timestamp()} [主线程] stop_event 已设置或检测到线程退出，开始关闭流程...")

    except KeyboardInterrupt:
        print(f"\n{timestamp()} [主线程] 检测到 KeyboardInterrupt，开始关闭流程...")
        if not stop_event.is_set():
            custom_signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"\n{timestamp()} [主线程] 发生意外错误: {e}")
        import traceback;

        traceback.print_exc()
        if not stop_event.is_set():
            custom_signal_handler(signal.SIGTERM, None)
    finally:
        print(f"{timestamp()} [主线程] 开始清理和关闭...")
        if not stop_event.is_set():  # Ensure stop_event is set and queues get None
            stop_event.set()
            for q_to_signal in active_queues_for_signal:
                try:
                    q_to_signal.put_nowait(None)
                except Exception:
                    pass

        # Join only the threads that were actually started
        if threads:
            timeout_seconds = 10
            start_join = time.time()
            print(f"{timestamp()} [主线程] 等待工作线程结束 (总超时: {timeout_seconds}s)...")
            active_threads_before_join = [t for t in threads if t.is_alive()]
            if active_threads_before_join:
                print(f"{timestamp()} [主线程] 仍在活动的线程: {[t.name for t in active_threads_before_join]}")
                for t in active_threads_before_join:
                    join_timeout = max(0.1, timeout_seconds - (time.time() - start_join))
                    if join_timeout <= 0.1:  # Check with a small positive value
                        print(f"{timestamp()} [主线程] 等待超时，不再等待线程 {t.name}。")
                        break
                    try:
                        t.join(timeout=join_timeout)
                    except Exception as join_e:
                        print(f"{timestamp()} [主线程] 等待线程 '{t.name}' 时出错: {join_e}")
                remaining_threads = [t.name for t in threads if t.is_alive()]
                if remaining_threads:
                    print(
                        f"{timestamp()} [主线程] 警告: 以下线程在最终超时 ({timeout_seconds}s) 后仍未结束: {', '.join(remaining_threads)}")
                else:
                    print(f"{timestamp()} [主线程] 所有工作线程均已结束。")
            else:
                print(f"{timestamp()} [主线程] 所有工作线程在等待开始前均已结束。")
        else:
            print(f"{timestamp()} [主线程] 没有活动的工作线程需要等待。")

        # 文件清理 (★★★ MODIFIED ★★★)
        if args.input_mode == 'voice' and not args.no_cleanup:
            cleanup_temp_files(prefix=TEMP_AUDIO_PREFIX)
        else:
            print(f"{timestamp()} [主线程] 跳过临时文件清理 (非语音输入模式或 --no-cleanup)。")

        if processing_lock.locked():
            print(f"{timestamp()} [主线程] 警告: 程序退出时处理锁仍被持有，强制释放。")
            try:
                processing_lock.release()
            except RuntimeError:
                pass

        print(f"\n{timestamp()} [主线程] 程序退出。")
        os._exit(0)

# --- END OF FILE main_parrot_loop_sherpa.py (v3.3 - Text I/O Mode) ---