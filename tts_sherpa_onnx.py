# --- START OF FILE tts_sherpa_onnx.py (v2.24.4 - Update Chinese Model to MeloTTS) ---

# tts_sherpa_onnx.py
"""
使用 Sherpa-ONNX 实现离线文本转语音功能。
v2.24.4:
- 基于 v2.24.3。
- 更新默认中文模型为 vits-melo-tts-zh_en (中英双语)。
- 更新相应的 URL 和子目录名。
- 调整 VITS 加载逻辑：对中文模型，查找 dict 目录但不再强制要求其存在，
  仅打印警告（因为 melo-tts 可能不需要），但仍会传递 dict_dir 参数。
"""
import os
import argparse
import platform
import time
import logging
from pathlib import Path
import urllib.request
import tarfile
import zipfile
import shutil
from typing import Union, Optional, Dict, List

# --- 依赖项导入 (保持不变) ---
_sherpa_onnx_available = False
_playback_libs_available = False
OfflineTts = None
OfflineTtsModelConfig = None
OfflineTtsVitsModelConfig = None
OfflineTtsMatchaModelConfig = None
OfflineTtsConfig = None
sd = None
np = None

try:
    import sherpa_onnx
    OfflineTts = sherpa_onnx.OfflineTts
    OfflineTtsModelConfig = sherpa_onnx.OfflineTtsModelConfig
    OfflineTtsVitsModelConfig = sherpa_onnx.OfflineTtsVitsModelConfig
    OfflineTtsMatchaModelConfig = sherpa_onnx.OfflineTtsMatchaModelConfig
    OfflineTtsConfig = sherpa_onnx.OfflineTtsConfig
    print("成功导入 sherpa-onnx。")
    _sherpa_onnx_available = True
except ImportError: print("警告: 未找到 'sherpa-onnx' 库。TTS 功能不可用。"); _sherpa_onnx_available = False
except Exception as e: print(f"导入 sherpa-onnx 时出错: {e}"); _sherpa_onnx_available = False

try:
    import sounddevice as sd
    import numpy as np
    print("成功导入 sounddevice, numpy (用于音频播放)。")
    _playback_libs_available = True
except ImportError: print("警告: 未找到 'sounddevice' 或 'numpy'。音频播放功能不可用。"); _playback_libs_available = False

# --- 配置日志 (保持不变) ---
logging.getLogger('sherpa_onnx').setLevel(logging.WARNING)

# --- 全局 TTS 对象缓存 (保持不变) ---
_tts_cache: Dict[str, Optional[OfflineTts]] = {}

# --- 默认模型配置 (★★★ 更新中文模型 ★★★) ---
DEFAULT_MODEL_DOWNLOAD_DIR = "./sherpa_onnx_models"
# 英文模型: matcha (保持不变)
EN_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2"
EN_MODEL_EXPECTED_SUBDIR = "matcha-icefall-en_US-ljspeech"
# 中文模型: vits-melo-tts-zh_en
ZH_MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"
ZH_MODEL_EXPECTED_SUBDIR = "vits-melo-tts-zh_en" # <--- 更新

# --- 默认值定义 (中文 Speaker ID 可能需要注意) ---
DEFAULT_SPEAKER_ID_SENTINEL = -1
# MeloTTS zh_en 通常默认中文是 0，英文是 1，但具体看模型。暂时保留 0。
DEFAULT_ZH_SPEAKER_ID = 0
DEFAULT_EN_SPEAKER_ID = 0 # Matcha LJSpeech 是 0
DEFAULT_SPEED_SENTINEL = -1.0
DEFAULT_ZH_SPEED = 1.0
DEFAULT_EN_SPEED = 1.0
DEFAULT_VITS_NOISE_SCALE = 0.667
DEFAULT_VITS_NOISE_SCALE_W = 0.8
DEFAULT_VITS_LENGTH_SCALE = 1.0
DEFAULT_MATCHA_NOISE_SCALE = 0.667
DEFAULT_MATCHA_LENGTH_SCALE = 1.0

DEFAULT_TTS_PROVIDER = "cpu"
DEFAULT_TTS_NUM_THREADS = 1
DEFAULT_TTS_SILENCE_MS = 200


# --- 依赖检查 (保持不变) ---
def check_tts_dependencies():
    print("\n检查 TTS (Sherpa-ONNX) 依赖...")
    ok = True
    if not _sherpa_onnx_available: print("错误: 'sherpa-onnx' 未安装或导入失败。"); ok = False
    else: print("'sherpa-onnx' 可用。")
    if not _playback_libs_available: print("错误: 播放库 ('sounddevice', 'numpy') 未安装或导入失败。"); ok = False
    else: print("播放库 ('sounddevice', 'numpy') 可用。")
    print("-" * 20)
    return ok

# --- 模型下载和解压函数 (保持不变) ---
def _download_file_with_progress(url: str, dest_path: str) -> bool:
    # (代码省略，与 v2.24.3 相同)
    try:
        print(f"开始下载: {url}")
        with urllib.request.urlopen(url) as response, open(dest_path, 'wb') as out_file:
            total_size_in_bytes = response.length
            if total_size_in_bytes: print(f"文件大小: {total_size_in_bytes / (1024*1024):.2f} MB")
            else: print("文件大小未知...")
            block_size = 8192; downloaded_size = 0; start_time = time.time()
            while True:
                chunk = response.read(block_size);
                if not chunk: break
                out_file.write(chunk); downloaded_size += len(chunk)
                if total_size_in_bytes:
                    percent = downloaded_size * 100 / total_size_in_bytes; elapsed_time = time.time() - start_time
                    speed = (downloaded_size / 1024) / elapsed_time if elapsed_time > 0 else 0
                    bar_len = 30; filled_len = int(bar_len * downloaded_size / total_size_in_bytes)
                    bar = '=' * filled_len + ' ' * (bar_len - filled_len)
                    print(f"\r[{bar}] {percent:.1f}% ({downloaded_size/1024/1024:.1f}/{total_size_in_bytes/1024/1024:.1f}MB @{speed:.1f}KB/s)", end='')
                else: print(f"\r已下载: {downloaded_size / 1024:.1f} KB", end='')
            print("\n下载完成。")
        return True
    except Exception as e:
        print(f"\n下载失败: {e}");
        if os.path.exists(dest_path):
            try: os.remove(dest_path)
            except OSError: pass
        return False

def _extract_archive(archive_path: str, dest_dir: str) -> bool:
    # (代码省略，与 v2.24.3 相同)
    print(f"开始解压: {archive_path} 到 {dest_dir}")
    try:
        if archive_path.lower().endswith((".tar.bz2", ".tbz2")):
            with tarfile.open(archive_path, "r:bz2") as tar: tar.extractall(path=dest_dir); print("解压 tar.bz2 完成。"); return True
        elif archive_path.lower().endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref: zip_ref.extractall(dest_dir); print("解压 zip 完成。"); return True
        else: print(f"错误: 不支持的压缩格式: {archive_path}"); return False
    except tarfile.ReadError: print(f"错误: 解压 tar 文件失败: {archive_path}"); return False
    except zipfile.BadZipFile: print(f"错误: 解压 zip 文件失败: {archive_path}"); return False
    except Exception as e: print(f"解压失败: {e.__class__.__name__}: {e}"); return False

def download_and_extract_model(model_url: str, download_base_dir: str, expected_subdir: str) -> Optional[Path]:
    # (代码省略，与 v2.24.3 相同)
    download_base_path = Path(download_base_dir); target_model_path = download_base_path / expected_subdir
    archive_filename = Path(model_url).name; download_archive_path = download_base_path / archive_filename
    if target_model_path.is_dir(): print(f"模型目录已存在: {target_model_path}"); return target_model_path
    print("-" * 10, f"准备 TTS 模型: {expected_subdir}", "-" * 10)
    download_base_path.mkdir(parents=True, exist_ok=True)
    if not _download_file_with_progress(model_url, str(download_archive_path)): return None
    extract_success = _extract_archive(str(download_archive_path), str(download_base_path))
    try: print(f"清理压缩包: {download_archive_path}"); os.remove(download_archive_path)
    except OSError as e: print(f"警告: 清理压缩包失败: {e}")
    if not extract_success or not target_model_path.is_dir():
        print(f"错误: 解压后未找到模型目录: {target_model_path}"); return None
    print(f"模型成功准备于: {target_model_path}"); print("-" * (30 + len(expected_subdir))); return target_model_path


# --- TTS 模型加载 (★★★ 调整中文 VITS dict_dir 处理逻辑 ★★★) ---
def load_tts_engine(model_dir: str, provider: str = 'cpu', num_threads: int = 1) -> Optional[OfflineTts]:
    """加载 Sherpa-ONNX TTS 引擎。区分 VITS 和 Matcha 的加载方式。"""
    global _tts_cache
    engine_key = f"{model_dir}_{provider}_{num_threads}"
    if engine_key in _tts_cache:
        cached_engine = _tts_cache[engine_key]
        if cached_engine: print(f"从缓存加载引擎: {model_dir}"); return cached_engine
        else: print(f"缓存记录引擎加载失败: {engine_key}"); return None

    if not _sherpa_onnx_available or OfflineTts is None: print("错误: sherpa-onnx 库不可用。"); return None
    model_path = Path(model_dir);
    if not model_path.is_dir(): print(f"错误: 模型目录 '{model_dir}' 不存在。"); return None

    print(f"开始加载 Sherpa-ONNX TTS 引擎: {model_dir}")
    try:
        # --- 通用文件查找 ---
        tokens_path = model_path / "tokens.txt";
        if not tokens_path.exists(): print(f"错误: 未找到 'tokens.txt' 文件于 {model_path}"); return None
        lexicon_path = model_path / "lexicon.txt"; lexicon_str = str(lexicon_path) if lexicon_path.exists() else ""

        # --- 模型类型判断 ---
        model_dir_lower = model_dir.lower()
        # 使用更新后的中文模型子目录名判断
        is_chinese_model = ZH_MODEL_EXPECTED_SUBDIR.lower() in model_dir_lower
        is_vits = "vits" in model_dir_lower
        is_matcha = "matcha" in model_dir_lower

        model_type = "unknown"
        if is_vits: model_type = "vits"
        elif is_matcha: model_type = "matcha"
        else: # 尝试推断
             onnx_files_generic = list(model_path.glob("*.onnx"))
             if any("vits" in f.name.lower() for f in onnx_files_generic): model_type="vits"
             elif any("matcha" in f.name.lower() for f in onnx_files_generic): model_type="matcha"
             elif (model_path / "config.json").exists(): model_type = "vits"

        print(f"  检测到/假定模型类型: {model_type}")
        if is_chinese_model and model_type == "vits":
            print("  检测到中文 VITS 模型，将查找 data/dict 目录。")

        # --- 配置顶层模型和 TTS Config ---
        top_level_model_config = OfflineTtsModelConfig()
        top_level_model_config.provider = provider
        top_level_model_config.num_threads = num_threads
        try: top_level_model_config.debug = False
        except AttributeError: pass
        print(f"  模型配置: provider={provider}, num_threads={num_threads}")

        # --- 根据模型类型配置具体模型 ---
        if model_type == "vits":
            # VITS: 查找 onnx, data, dict
            onnx_files = list(model_path.glob("*.onnx"))
            if not onnx_files: print(f"错误: VITS 模型在 '{model_dir}' 中未找到 .onnx 文件。"); return None
            # 选择 VITS onnx 文件
            actual_model_path_obj = None
            non_int8 = [f for f in onnx_files if 'int8' not in f.name.lower()]
            int8_files = [f for f in onnx_files if 'int8' in f.name.lower()]
            if non_int8: actual_model_path_obj = non_int8[0]
            elif int8_files: actual_model_path_obj = int8_files[0]
            else: actual_model_path_obj = onnx_files[0]
            actual_model_path_str = str(actual_model_path_obj)
            print(f"  使用 VITS 模型文件: {actual_model_path_obj.name}")

            # 明确查找 data 和 dict 目录
            data_path = model_path / "data"; data_dir_str = str(data_path) if data_path.is_dir() else ""
            dict_path = model_path / "dict"; dict_dir_str = str(dict_path) if dict_path.is_dir() else ""

            if data_dir_str: print(f"  找到 VITS data 目录: {data_dir_str}")
            if dict_dir_str: print(f"  找到 VITS dict 目录: {dict_dir_str}")

            # ★★★ 对中文模型，如果 dict 目录未找到，打印警告而非错误 ★★★
            if is_chinese_model and not dict_dir_str:
                print(f"警告: 中文 VITS 模型 '{model_dir}' 未找到 'dict' 目录。如果此特定模型需要（例如使用jieba），可能会加载失败。")
                # 不再返回 None，继续尝试加载

            # 配置 VITS 模型，仍然同时传递 data_dir 和 dict_dir
            vits_config = OfflineTtsVitsModelConfig(
                 model=actual_model_path_str,
                 lexicon=lexicon_str,
                 tokens=str(tokens_path),
                 data_dir=data_dir_str,
                 dict_dir=dict_dir_str, # 即使为空也传递，由 C++ 端决定是否需要
                 noise_scale=DEFAULT_VITS_NOISE_SCALE,
                 noise_scale_w=DEFAULT_VITS_NOISE_SCALE_W,
                 length_scale=DEFAULT_VITS_LENGTH_SCALE)
            print(f"  应用 VITS 参数: noise={vits_config.noise_scale:.3f}, noise_w={vits_config.noise_scale_w:.3f}, length={vits_config.length_scale:.3f}, dict_dir='{dict_dir_str}', data_dir='{data_dir_str}'")
            top_level_model_config.vits = vits_config

        elif model_type == "matcha":
            # Matcha 加载逻辑 (保持 v2.24.3 不变)
            acoustic_model_path_obj = None; vocoder_path_obj = None
            acoustic_candidates = list(model_path.glob("*steps*.onnx")) + list(model_path.glob("*acoustic*.onnx"))
            vocoder_candidates = list(model_path.glob("*vocos*.onnx")) + list(model_path.glob("*vocoder*.onnx")) + list(model_path.glob("*univ*.onnx"))
            if acoustic_candidates: acoustic_model_path_obj = acoustic_candidates[0]
            if vocoder_candidates: vocoder_path_obj = vocoder_candidates[0]
            if not acoustic_model_path_obj: print(f"错误: Matcha 模型在 '{model_dir}' 中未找到声学模型 onnx 文件。"); return None
            if not vocoder_path_obj: print(f"错误: Matcha 模型在 '{model_dir}' 中未找到声码器 onnx 文件。"); return None
            acoustic_model_path_str = str(acoustic_model_path_obj); vocoder_path_str = str(vocoder_path_obj)
            print(f"  使用 Matcha 声学模型: {acoustic_model_path_obj.name}")
            print(f"  使用 Matcha 声码器: {vocoder_path_obj.name}")
            matcha_data_dir_path = model_path / "espeak-ng-data"
            matcha_data_dir_str = str(matcha_data_dir_path) if matcha_data_dir_path.is_dir() else ""
            if matcha_data_dir_str: print(f"  使用 Matcha data 目录: {matcha_data_dir_str}")
            else: print(f"警告: Matcha 模型 '{model_dir}' 未找到预期的 'espeak-ng-data' 目录。")
            matcha_config = OfflineTtsMatchaModelConfig(
                  acoustic_model=acoustic_model_path_str, vocoder=vocoder_path_str,
                  lexicon=lexicon_str, tokens=str(tokens_path), data_dir=matcha_data_dir_str,
                  noise_scale=DEFAULT_MATCHA_NOISE_SCALE, length_scale=DEFAULT_MATCHA_LENGTH_SCALE )
            print(f"  应用 Matcha 参数: noise_scale={matcha_config.noise_scale:.3f}, length_scale={matcha_config.length_scale:.3f}")
            top_level_model_config.matcha = matcha_config
        else:
             print(f"错误: 无法识别或不支持的模型类型: {model_dir}。");
             _tts_cache[engine_key] = None; return None

        # --- 初始化最终的 TTS 配置 ---
        tts_config = OfflineTtsConfig( model=top_level_model_config, rule_fsts="", max_num_sentences=1 )

        tts_engine = OfflineTts(tts_config)
        _tts_cache[engine_key] = tts_engine; print(f"Sherpa-ONNX TTS 引擎加载成功。")
        return tts_engine
    except Exception as e:
        print(f"\n错误: 加载引擎时出错: {e.__class__.__name__}: {e}")
        import traceback; traceback.print_exc(); _tts_cache[engine_key] = None; return None

# --- 核心 TTS 功能 (保持不变) ---
def speak_text(text: str,
               model_dir: Optional[str] = None,
               speaker_id: int = DEFAULT_SPEAKER_ID_SENTINEL,
               speed: float = DEFAULT_SPEED_SENTINEL,
               provider: str = 'cpu',
               num_threads: int = 1,
               add_silence_ms: int = 200):
    # (代码省略，与 v2.24.3 相同)
    print(f"\n--- 文本转语音 (Sherpa-ONNX) ---")
    print(f"原始请求: text='{text[:50]}...', model='{model_dir}', sid={speaker_id}, speed={speed}")

    if not text or not text.strip(): print("错误: 输入文本为空。"); return False
    if not _sherpa_onnx_available or not _playback_libs_available: print("错误: 必要的库未加载。"); return False
    if model_dir is None or not Path(model_dir).is_dir(): print("错误：无效的模型目录。"); return False

    final_speaker_id = speaker_id; final_speed = speed
    model_dir_lower = model_dir.lower()
    # 使用更新后的中文模型子目录名判断
    is_effective_zh = ZH_MODEL_EXPECTED_SUBDIR.lower() in model_dir_lower
    is_effective_en = EN_MODEL_EXPECTED_SUBDIR.lower() in model_dir_lower

    if final_speaker_id == DEFAULT_SPEAKER_ID_SENTINEL:
        if is_effective_zh: final_speaker_id = DEFAULT_ZH_SPEAKER_ID # 使用新的中文默认 ID
        elif is_effective_en: final_speaker_id = DEFAULT_EN_SPEAKER_ID
        else: final_speaker_id = 0
        print(f"  应用默认 Speaker ID: {final_speaker_id}")
    # 注意：MeloTTS 可能有多个中文/英文 Speaker ID，用户可能需要手动指定非0的ID
    if is_effective_zh and final_speaker_id != 0:
        print(f"  提示: MeloTTS 中英模型使用 Speaker ID {final_speaker_id}。请确认此 ID 对该模型有效。")


    if final_speed == DEFAULT_SPEED_SENTINEL:
        if is_effective_en: final_speed = DEFAULT_EN_SPEED
        elif is_effective_zh: final_speed = DEFAULT_ZH_SPEED
        else: final_speed = 1.0
        print(f"  应用默认语速: {final_speed:.1f}")
    elif not (0.5 <= final_speed <= 2.0):
         print(f"  警告: 语速 {final_speed} 超出建议范围 (0.5-2.0)。"); final_speed = max(0.5, min(final_speed, 2.0)); print(f"  语速已约束到: {final_speed:.1f}")

    print(f"最终参数: Speaker ID={final_speaker_id}, 语速={final_speed:.1f}, 末尾静音={add_silence_ms}ms")

    tts_engine = load_tts_engine(model_dir, provider, num_threads);
    if tts_engine is None: return False

    print("开始合成音频..."); start_synth_time = time.time()
    try:
        audio_result = tts_engine.generate(text=text, sid=final_speaker_id, speed=final_speed)
        synth_duration = time.time() - start_synth_time
        if audio_result is None or not hasattr(audio_result, 'samples') or audio_result.samples is None or not hasattr(audio_result, 'sample_rate'):
             print(f"错误: 合成失败或返回无效结果 (耗时 {synth_duration:.2f} 秒)。"); return False
        print(f"合成完成，耗时 {synth_duration:.2f} 秒。采样率: {audio_result.sample_rate} Hz。")

        samples_data: Optional[np.ndarray] = None
        if isinstance(audio_result.samples, (list, np.ndarray)):
            if len(audio_result.samples) == 0: print("错误: 合成结果样本为空。"); return False
            try: samples_data = np.array(audio_result.samples, dtype=np.float32)
            except ValueError as ve: print(f"错误: 转换样本为 numpy 数组失败: {ve}"); return False
        else: print(f"错误: 未知 samples 类型: {type(audio_result.samples)}"); return False
        if samples_data is None or samples_data.size == 0: print("错误: 未能获取有效音频数据。"); return False

        if add_silence_ms > 0:
            silence_samples = int(audio_result.sample_rate * (add_silence_ms / 1000.0))
            if silence_samples > 0:
                silence_array = np.zeros(silence_samples, dtype=samples_data.dtype)
                samples_data = np.concatenate((samples_data, silence_array))
                print(f"在末尾添加了 {add_silence_ms} ms 的静音。")

        print("开始播放音频..."); start_play_time = time.time()
        try:
            sd.play(samples_data, samplerate=audio_result.sample_rate, blocking=True)
            play_duration = time.time() - start_play_time
            print(f"播放结束，耗时 {play_duration:.2f} 秒。"); print("------------------------------"); return True
        except Exception as play_err: print(f"播放音频时出错: {play_err}"); return False
    except RuntimeError as re:
        err_msg = str(re);
        if "This model contains only" in err_msg and "speakers" in err_msg: print(f"合成错误提示: {err_msg.strip()}.")
        elif "matcha" in err_msg.lower(): print(f"Matcha 合成运行时错误: {re}")
        else: print(f"合成运行时错误: {re}"); import traceback; traceback.print_exc();
        return False
    except Exception as synth_err:
        print(f"合成或处理错误: {synth_err}"); import traceback; traceback.print_exc(); return False

# --- 用于测试的主函数 (★★★ 更新中文模型信息 ★★★) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="使用 Sherpa-ONNX 合成并播放文本。", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("text", type=str, nargs='?', default=None, help="要朗读的文本。对于中英模型，可包含中英文。")
    parser.add_argument("--model-dir", type=str, default=None, help="TTS 模型目录路径。")
    parser.add_argument("--download-dir", type=str, default=DEFAULT_MODEL_DOWNLOAD_DIR, help="模型下载/查找目录。")
    # 更新 Speaker ID 帮助信息，提示 MeloTTS 可能有不同 ID
    parser.add_argument("--speaker-id", type=int, default=DEFAULT_SPEAKER_ID_SENTINEL, help=f"说话人 ID (默认自动: 中={DEFAULT_ZH_SPEAKER_ID}, 英={DEFAULT_EN_SPEAKER_ID})。MeloTTS中英模型 ID 可能不同，请查阅模型说明。")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED_SENTINEL, help=f"语速 (默认自动: 中={DEFAULT_ZH_SPEED:.1f}, 英={DEFAULT_EN_SPEED:.1f})。")
    parser.add_argument("--provider", type=str, default="cpu", help="ONNX Runtime provider。")
    parser.add_argument("--num-threads", type=int, default=1, help="CPU 线程数。")
    parser.add_argument("--silence-ms", type=int, default=200, help="在末尾添加的静音毫秒数 (0=不添加)。")
    parser.add_argument("--list-models-info", action="store_true", help="显示默认模型信息和下载地址。")
    parser.add_argument("--test-en", action="store_true", help=f"运行默认英文 ({EN_MODEL_EXPECTED_SUBDIR}) 测试。")
    # 更新中文测试帮助文本
    parser.add_argument("--test-zh", action="store_true", help=f"运行默认中文 ({ZH_MODEL_EXPECTED_SUBDIR}) 测试。")
    args = parser.parse_args()

    if not check_tts_dependencies(): exit(1)
    if args.list_models_info:
        print("\n--- Sherpa-ONNX TTS 默认模型信息 ---")
        print(f"  英文模型 URL: {EN_MODEL_URL}")
        print(f"     -> Subdir: {EN_MODEL_EXPECTED_SUBDIR}, Default SID: {DEFAULT_EN_SPEAKER_ID} (Matcha)")
        # 更新中文模型信息
        print(f"  中文模型 URL: {ZH_MODEL_URL}")
        print(f"     -> Subdir: {ZH_MODEL_EXPECTED_SUBDIR}, Default SID: {DEFAULT_ZH_SPEAKER_ID} (VITS - MeloTTS, 中英双语)")
        print("\n模型下载发布页: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models")
        print("\nMeloTTS Speaker ID 说明: 通常中文 SID=0, 英文 SID=1。具体请查看模型文档。")
        print("\n示例:")
        print(f"  测试中文: python {os.path.basename(__file__)} --test-zh")
        print(f"  测试英文: python {os.path.basename(__file__)} --test-en")
        print(f"  使用 MeloTTS 读中文: python {os.path.basename(__file__)} --model-dir ./sherpa_onnx_models/{ZH_MODEL_EXPECTED_SUBDIR} --speaker-id 0 \"你好世界\"")
        print(f"  使用 MeloTTS 读英文: python {os.path.basename(__file__)} --model-dir ./sherpa_onnx_models/{ZH_MODEL_EXPECTED_SUBDIR} --speaker-id 1 \"Hello world.\"")
        print(f"  使用 Matcha 读英文: python {os.path.basename(__file__)} --model-dir ./sherpa_onnx_models/{EN_MODEL_EXPECTED_SUBDIR} --speaker-id 0 --speed 0.9 \"Hello world.\"")
        print(f"  控制末尾静音: python {os.path.basename(__file__)} --test-zh --silence-ms 0")
        exit(0)

    print("=" * 30); print("开始文本转语音 (Sherpa-ONNX) 测试"); print("=" * 30)

    effective_model_dir = args.model_dir
    if args.test_zh or args.test_en:
        if effective_model_dir: print(f"警告: --test-xx 优先于 --model-dir。")
        # 使用更新后的中文 URL 和 Subdir
        model_url = ZH_MODEL_URL if args.test_zh else EN_MODEL_URL
        subdir = ZH_MODEL_EXPECTED_SUBDIR if args.test_zh else EN_MODEL_EXPECTED_SUBDIR
        lang_name = "中文" if args.test_zh else "英文"; print(f"准备测试默认 {lang_name} 模型 ({subdir})...")
        downloaded_path = download_and_extract_model(model_url, args.download_dir, subdir)
        if downloaded_path: effective_model_dir = str(downloaded_path)
        else: print(f"错误: 准备默认 {lang_name} 模型失败。"); exit(1)
    elif not effective_model_dir:
        print("提示: 未指定模型，尝试自动查找...");
        # 使用更新后的中文 Subdir 查找
        zh_path = Path(args.download_dir) / ZH_MODEL_EXPECTED_SUBDIR;
        en_path = Path(args.download_dir) / EN_MODEL_EXPECTED_SUBDIR
        if zh_path.is_dir(): effective_model_dir = str(zh_path); print(f"  找到中文模型: {effective_model_dir}")
        elif en_path.is_dir(): effective_model_dir = str(en_path); print(f"  找到英文模型: {effective_model_dir}")
        else: print(f"错误: 在 '{args.download_dir}' 未找到默认模型 ({ZH_MODEL_EXPECTED_SUBDIR} 或 {EN_MODEL_EXPECTED_SUBDIR})。请使用 --model-dir 或 --test-xx。"); exit(1)

    if not effective_model_dir or not Path(effective_model_dir).is_dir(): print(f"错误: 无效模型目录: '{effective_model_dir}'。"); exit(1)
    print(f"最终使用模型目录: {effective_model_dir}")

    text_to_speak = args.text
    if not text_to_speak:
        if args.test_en: text_to_speak = "This is a test using the Matcha Icefall English model from Sherpa O N N X."; print("使用默认英文测试句。")
        # 更新中文测试句
        elif args.test_zh: text_to_speak = "你好，这是使用 VITS Melo T T S 中英双语模型进行的语音合成测试。"; print("使用默认中文测试句 (SID=0)。")
        else: print(f"错误: 请提供要朗读的文本或使用 --test-xx。"); exit(1)

    # 确定 speaker_id (如果使用 --test-zh 且未指定 --speaker-id，确保用中文默认 ID)
    final_sid_for_test = args.speaker_id
    if args.test_zh and final_sid_for_test == DEFAULT_SPEAKER_ID_SENTINEL:
        final_sid_for_test = DEFAULT_ZH_SPEAKER_ID

    success = speak_text(
        text=text_to_speak,
        model_dir=effective_model_dir,
        speaker_id=final_sid_for_test, # 使用确定后的 ID
        speed=args.speed,
        provider=args.provider,
        num_threads=args.num_threads,
        add_silence_ms=args.silence_ms
    )

    print("\n--- 测试总结 ---"); print("TTS 测试成功完成。" if success else "TTS 测试失败。"); print("=" * 30)
    exit(0 if success else 1)

# --- END OF FILE tts_sherpa_onnx.py (v2.24.4 - Update Chinese Model to MeloTTS) ---