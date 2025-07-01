# --- START OF FILE stt.py (v3.0 - Correctly Use from_paraformer) ---

# stt.py
"""
Handles Speech-to-Text (STT) using a local Sherpa-ONNX Offline ASR (Paraformer).
v3.0:
- Correctly uses the OfflineRecognizer.from_paraformer() class method for loading.
- Requires model to be present locally.
- Requires 'sherpa-onnx', 'soundfile'. 'librosa' is optional.
"""
import os
import argparse
import platform
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

# --- 依赖项导入 ---
_sherpa_onnx_available = False; _soundfile_available = False; _librosa_available = False
OfflineRecognizer = None # We get the class itself
# Config classes are not needed for this loading method
sf = None; librosa = None
try:
    import sherpa_onnx
    OfflineRecognizer = sherpa_onnx.OfflineRecognizer
    print("成功导入 sherpa-onnx。"); _sherpa_onnx_available = True
except ImportError: print("警告: 未找到 'sherpa-onnx' 库。"); _sherpa_onnx_available = False
except Exception as e: print(f"导入 sherpa-onnx 时出错: {e}"); _sherpa_onnx_available = False
try: import soundfile as sf; print("成功导入 soundfile。"); _soundfile_available = True
except ImportError: print("警告: 未找到 'soundfile' 库。")
try: import librosa; print("成功导入 librosa。"); _librosa_available = True
except ImportError: print("提示: 未找到 'librosa' 库。")

# --- 配置默认值 ---
DEFAULT_LOCAL_MODEL_DIR = r"C:\ELF\sherpa_onnx_paraformer\sherpa-onnx-paraformer-zh-2023-09-14"
DEFAULT_ASR_NUM_THREADS = 1
DEFAULT_ASR_PROVIDER = "cpu"
DEFAULT_ASR_DECODING_METHOD = "greedy_search"
print(f"STT (Sherpa-ONNX) 配置: Provider='{DEFAULT_ASR_PROVIDER}', Threads={DEFAULT_ASR_NUM_THREADS}")
print(f"提示: 默认使用模型目录 '{DEFAULT_LOCAL_MODEL_DIR}'。可用 --model-dir 覆盖。")

# --- Recognizer 缓存 ---
_recognizer_cache: Dict[str, Optional[OfflineRecognizer]] = {}

# --- Dependency Check ---
def check_stt_dependencies():
    """检查 Sherpa-ONNX ASR 及其相关依赖是否可用。"""
    print("\nChecking STT (Sherpa-ONNX ASR) dependencies...")
    ok = True
    if not _sherpa_onnx_available: print("错误: 'sherpa-onnx' 未安装或导入失败。"); ok = False
    else: print("'sherpa-onnx' 可用。")
    if not _soundfile_available: print("错误: 'soundfile' 未安装或导入失败。"); ok = False
    else: print("'soundfile' 可用。")
    if not _librosa_available: print("提示: 'librosa' (可选, 用于重采样) 未安装。")
    else: print("'librosa' 可用。")
    print("-" * 20)
    return ok

# --- Model Loading (使用 from_paraformer 类方法) ---
def _load_asr_recognizer(
    model_dir: str,
    num_threads: int = DEFAULT_ASR_NUM_THREADS,
    provider: str = DEFAULT_ASR_PROVIDER,
    decoding_method: str = DEFAULT_ASR_DECODING_METHOD
) -> Optional[OfflineRecognizer]:
    """加载 Sherpa-ONNX OfflineRecognizer (使用 from_paraformer)。"""
    global _recognizer_cache
    model_path = Path(model_dir)
    # Cache key should include relevant parameters used by from_paraformer
    cache_key = f"{model_dir}_{provider}_{num_threads}_{decoding_method}"

    if cache_key in _recognizer_cache:
        cached = _recognizer_cache[cache_key]
        if cached: print(f"从缓存加载 ASR Recognizer: {cache_key}"); return cached
        else: print(f"缓存记录 ASR Recognizer '{cache_key}' 加载失败。"); return None
    if not _sherpa_onnx_available: print("错误: Sherpa-ONNX 库不可用。"); return None
    if not model_path.is_dir(): print(f"错误: ASR 模型目录 '{model_dir}' 不存在。"); return None

    print(f"\n开始加载 Sherpa-ONNX ASR Recognizer: {model_dir}")
    print(f"  Provider: {provider}, Threads: {num_threads}, Decoding: {decoding_method}")
    load_start_time = time.time()

    paraformer_model_file = "model.int8.onnx"; tokens_file = "tokens.txt"
    paraformer_model_path = model_path / paraformer_model_file; tokens_path = model_path / tokens_file
    missing_files = []
    if not paraformer_model_path.is_file(): missing_files.append(paraformer_model_file)
    if not tokens_path.is_file(): missing_files.append(tokens_file)
    if missing_files: print(f"错误: 模型目录 '{model_dir}' 缺少文件: {', '.join(missing_files)}"); _recognizer_cache[cache_key] = None; return None
    print(f"  找到模型文件: Model='{paraformer_model_path.name}', Tokens='{tokens_path.name}'")

    try:
        # --- ★★★ 调用 OfflineRecognizer.from_paraformer ★★★
        recognizer = OfflineRecognizer.from_paraformer(
            paraformer=str(paraformer_model_path), # 传递 ONNX 文件路径
            tokens=str(tokens_path),             # 传递 tokens 文件路径
            num_threads=num_threads,
            sample_rate=16000,                  # 根据文档默认值设置
            feature_dim=80,                     # 根据文档默认值设置
            decoding_method=decoding_method,
            debug=False,
            provider=provider
            # rule_fsts="", # 其他可选参数可以根据需要添加
            # rule_fars="",
        )
        load_end_time = time.time()
        print(f"ASR Recognizer 加载成功 (耗时: {load_end_time - load_start_time:.2f} 秒)")
        _recognizer_cache[cache_key] = recognizer; return recognizer
    except TypeError as te:
         print(f"\n错误: 调用 from_paraformer 时参数错误 (TypeError): {te}")
         print(f"提示: 请再次核对 from_paraformer 的参数列表与您的 sherpa-onnx 版本 ({sherpa_onnx.__version__}) 是否匹配。")
         _recognizer_cache[cache_key] = None; return None
    except Exception as e:
        print(f"\n错误: 加载 ASR Recognizer 时出错: {e.__class__.__name__}: {e}"); import traceback; traceback.print_exc();
        _recognizer_cache[cache_key] = None; return None

# --- Transcription Function (保持不变) ---
def transcribe_audio(
    audio_path: str,
    model_dir: str,
    num_threads: int = DEFAULT_ASR_NUM_THREADS,
    provider: str = DEFAULT_ASR_PROVIDER,
    decoding_method: str = DEFAULT_ASR_DECODING_METHOD
) -> Tuple[Optional[str], Optional[str]]:
    # ... (函数体不变，确保内部调用 _load_asr_recognizer) ...
    print(f"\n--- 开始语音转文本 (Sherpa-ONNX ASR) ---")
    print(f"输入文件: {audio_path}"); print(f"使用模型目录: {model_dir}")
    if not _sherpa_onnx_available or not _soundfile_available: print("错误: 依赖库未加载。"); return None, None
    if not os.path.exists(audio_path): print(f"错误: 音频文件未找到: {audio_path}"); return None, None

    recognizer = _load_asr_recognizer(model_dir, num_threads, provider, decoding_method)
    if recognizer is None: print("错误: ASR Recognizer 加载失败。"); return None, None

    transcribed_text = None; detected_language = 'zh'
    transcribe_start_time = time.time()
    try:
        samples, sample_rate = sf.read(audio_path, dtype='float32', always_2d=False)
        print(f"读取音频: SR={sample_rate}Hz, Samples={len(samples)}")
        expected_sr = 16000 # from_paraformer 默认需要 16k
        if sample_rate != expected_sr:
            if not _librosa_available: print(f"错误: 采样率 ({sample_rate}Hz) 不符且缺少 librosa。期望 {expected_sr}Hz。"); return None, None
            print(f"警告: 采样率 ({sample_rate}Hz) 与期望 ({expected_sr}Hz) 不符，重采样...")
            try:
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=expected_sr)
                sample_rate = expected_sr; print(f"重采样完成: SR={sample_rate}Hz, Samples={len(samples)}")
            except Exception as resample_e: print(f"错误: 重采样失败: {resample_e}"); return None, None
        if len(samples) == 0: print("错误：音频样本为空。"); return None, None

        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        # stream.input_finished() # 可能需要

        recognizer.decode_stream(stream)
        result = stream.result; transcribed_text = result.text.strip()
        transcribe_end_time = time.time()
        print(f"ASR 转录完成 (耗时: {transcribe_end_time - transcribe_start_time:.2f} 秒)")
        if not transcribed_text: print("警告: 转录结果为空。"); transcribed_text = None
        if transcribed_text: print(f"转录结果: '{transcribed_text}'")
        print(f"--- STT (Sherpa-ONNX ASR) 结束 ---"); return transcribed_text, detected_language
    except sf.SoundFileError as sfe: print(f"\n错误: 读取音频文件失败: {sfe}"); return None, None
    except AttributeError as ae: # 捕捉 create_stream 等方法不存在的错误
         print(f"\n错误: 调用识别方法时出错: {ae}"); print("提示: v1.11.4 的识别流程可能不同。"); return None, None
    except Exception as e:
        print(f"\n错误: ASR 转录过程中出错: {e.__class__.__name__}: {e}"); import traceback; traceback.print_exc();
        print(f"--- STT (Sherpa-ONNX ASR) 失败 ---"); return None, None


# --- 用于测试的主函数 (保持不变) ---
if __name__ == "__main__":
    # ... (主函数代码不变) ...
    parser = argparse.ArgumentParser(description="使用本地 Sherpa-ONNX ASR 模型转录音频文件。", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_file", type=str, help="要转录的音频文件路径 (WAV, 16kHz 最佳)。")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_LOCAL_MODEL_DIR, help=f"【必需】指定本地 ASR 模型目录路径。")
    parser.add_argument("--provider", type=str, default=DEFAULT_ASR_PROVIDER, choices=["cpu", "cuda", "coreml"], help="ONNX Runtime provider。")
    parser.add_argument("--num-threads", type=int, default=DEFAULT_ASR_NUM_THREADS, help="CPU 推理线程数。")
    parser.add_argument("--decoding-method", type=str, default=DEFAULT_ASR_DECODING_METHOD, help="解码方法。")
    args = parser.parse_args()

    if not check_stt_dependencies(): exit(1)
    if not os.path.exists(args.audio_file): print(f"错误: 输入音频文件不存在: {args.audio_file}"); exit(1)
    if not Path(args.model_dir).is_dir(): print(f"错误: 指定的模型目录不存在或不是目录: {args.model_dir}"); exit(1)

    print("=" * 30); print("开始 STT (Sherpa-ONNX ASR) 模块测试"); print("=" * 30)
    print(f"平台: {platform.system()} ({platform.machine()})"); print(f"转录文件: {args.audio_file}"); print(f"模型目录: {args.model_dir}")
    print(f"Provider: {args.provider}, 线程: {args.num_threads}, 解码: {args.decoding_method}")

    test_start_time = time.time()
    text, lang = transcribe_audio(audio_path=args.audio_file, model_dir=args.model_dir,
                               provider=args.provider, num_threads=args.num_threads,
                               decoding_method=args.decoding_method)
    test_end_time = time.time()

    print("\n--- 测试结果 ---")
    if text is not None:
        print(f"  识别语言 (假定): {lang}"); print(f"  转录文本: '{text}'")
        print(f"  总耗时: {test_end_time - test_start_time:.2f} 秒")
    else: print("  ASR 转录失败或结果为空。"); print(f"  总耗时: {test_end_time - test_start_time:.2f} 秒")
    print("=" * 30); print("\n提示:"); print(f" - 确认模型路径和文件正确。"); print(f" - 确保音频为 16kHz WAV。")
    print(f" - 如果加载失败, 尝试核对 from_paraformer 参数。")
    exit(0 if text is not None else 1)


# --- END OF FILE stt.py (v3.0 - Correctly Use from_paraformer) ---