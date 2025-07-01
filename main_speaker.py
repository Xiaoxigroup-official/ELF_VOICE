# --- START OF FILE main_speaker.py ---
import sys
import os
import time
import datetime
import threading
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from PyQt6.QtCore import (QObject, pyqtSignal, QThread, Qt, pyqtSlot, QTimer, QMetaObject,
                          QCoreApplication, Q_ARG)
from PyQt6.QtWidgets import QApplication, QCheckBox

# --- Dynamic sys.path ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAKER_GUI_MODULE_PARENT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "gui", "speak_gui")
if not os.path.isdir(SPEAKER_GUI_MODULE_PARENT_DIR):
    print(f"错误: UI模块目录未找到: {SPEAKER_GUI_MODULE_PARENT_DIR}");
    sys.exit(1)
sys.path.insert(0, SPEAKER_GUI_MODULE_PARENT_DIR)

try:
    from speaker_gui import VoiceAssistantUI, ICONS_DIR as GUI_ICONS_DIR
except ImportError as e:
    print(f"错误: 无法从 speaker_gui.py 导入: {e}");
    sys.exit(1)
except NameError:
    GUI_ICONS_DIR = os.path.join(SPEAKER_GUI_MODULE_PARENT_DIR, "icons");
    print(
        f"警告: speaker_gui.GUI_ICONS_DIR 未找到, 使用回退路径: {GUI_ICONS_DIR}")

# --- Import pipeline components ---
try:
    from audio_capture import (record_with_vad, list_audio_devices, DEFAULT_SAMPLERATE, DEFAULT_CHANNELS, DEFAULT_DTYPE,
                               DEFAULT_VAD_AGGRESSIVENESS, DEFAULT_VAD_FRAME_MS, DEFAULT_VAD_SILENCE_TIMEOUT_MS,
                               DEFAULT_VAD_MIN_SPEECH_DURATION_MS, DEFAULT_VAD_MAX_RECORDING_S)
    from stt import (transcribe_audio, _load_asr_recognizer, DEFAULT_LOCAL_MODEL_DIR as STT_DEFAULT_MODEL_DIR,
                     DEFAULT_ASR_PROVIDER as STT_DEFAULT_PROVIDER, DEFAULT_ASR_NUM_THREADS as STT_DEFAULT_NUM_THREADS,
                     DEFAULT_ASR_DECODING_METHOD as STT_DEFAULT_DECODING_METHOD)
    from llm_interaction import get_llm_response, load_llm_config
    from tts_sherpa_onnx import (speak_text, download_and_extract_model, load_tts_engine,
                                 DEFAULT_MODEL_DOWNLOAD_DIR as TTS_DEFAULT_DOWNLOAD_DIR, EN_MODEL_URL as TTS_EN_URL,
                                 ZH_MODEL_URL as TTS_ZH_URL, EN_MODEL_EXPECTED_SUBDIR as TTS_EN_SUBDIR,
                                 ZH_MODEL_EXPECTED_SUBDIR as TTS_ZH_SUBDIR, DEFAULT_SPEAKER_ID_SENTINEL,
                                 DEFAULT_SPEED_SENTINEL, DEFAULT_ZH_SPEAKER_ID, DEFAULT_EN_SPEAKER_ID, DEFAULT_ZH_SPEED,
                                 DEFAULT_EN_SPEED, DEFAULT_TTS_PROVIDER, DEFAULT_TTS_NUM_THREADS,
                                 DEFAULT_TTS_SILENCE_MS)
except ImportError as e:
    print(f"关键组件导入失败: {e}. 请确保所有 .py 文件在同一目录或PYTHONPATH中。");
    sys.exit(1)

TEMP_AUDIO_DIR = os.path.join(CURRENT_SCRIPT_DIR, "temp_audio");
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
PRELOAD_LLM_PROMPT_ZH = "你好";
PRELOAD_LLM_PROMPT_EN = "Hello";
PRELOAD_TTS_TEXT_ZH = "你好";
PRELOAD_TTS_TEXT_EN = "Hello"


def log_message(message): print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} MainSpeaker] {message}",
                                flush=True)


def is_text_predominantly_english(text_segment: str) -> bool:
    if not text_segment: return False
    english_alpha_count = 0;
    cjk_char_count = 0
    for char_val in text_segment:
        if 'a' <= char_val.lower() <= 'z':
            english_alpha_count += 1
        elif '\u4e00' <= char_val <= '\u9fff':
            cjk_char_count += 1
    if english_alpha_count > 0 and (english_alpha_count * 0.8 > cjk_char_count or cjk_char_count == 0): return True
    if cjk_char_count > 0 and cjk_char_count >= english_alpha_count: return False
    return False


class VADWorker(QObject):
    speech_segment_recorded = pyqtSignal(str);
    vad_error = pyqtSignal(str);
    status_update = pyqtSignal(str)

    def __init__(self, audio_params, vad_params, temp_dir):
        super().__init__();
        self.audio_params = audio_params;
        self.vad_params = vad_params;
        self.temp_dir = temp_dir
        self._is_listening_active = False;
        self._stop_loop_requested = threading.Event();
        self.is_processing_downstream = threading.Event()
        log_message("VADWorker: Initialized.")  # Kept from previous version

    @pyqtSlot()
    def start_active_listening(self):
        # Logic from working code: if already active and not stopped, clear downstream and return.
        if self._is_listening_active and not self._stop_loop_requested.is_set():
            log_message(
                "VADWorker: start_active_listening: Already actively listening and not stopped. Clearing downstream flag.")
            self.is_processing_downstream.clear()
            return
        log_message(
            f"VADWorker: start_active_listening called. Setting states. Prev: _is_listening_active={self._is_listening_active}, _stop_loop_requested={self._stop_loop_requested.is_set()}")
        self._is_listening_active = True;
        self._stop_loop_requested.clear();
        self.is_processing_downstream.clear()
        log_message(
            "VADWorker: start_active_listening: States set: _is_listening_active=True, _stop_loop_requested=False, is_processing_downstream=False.")

    @pyqtSlot()
    def run_listening_loop(self):
        log_message("VADWorker: run_listening_loop started by thread.");
        cycle_count = 0
        while not self._stop_loop_requested.is_set():
            if not self._is_listening_active:
                # log_message("VADWorker: Loop: Not active. Waiting.") # Can be noisy
                self._stop_loop_requested.wait(0.1);
                continue
            if self.is_processing_downstream.is_set():
                # log_message("VADWorker: Loop: Downstream busy. Waiting.") # Can be noisy
                self._stop_loop_requested.wait(0.1);
                continue

            cycle_count += 1;
            temp_filename = Path(self.temp_dir) / f"vad_{int(time.time() * 1000)}_{cycle_count}.wav"
            try:
                success, saved_filepath_str, actual_duration = record_with_vad(filename=str(temp_filename),
                                                                               **self.audio_params, **self.vad_params)
                if self._stop_loop_requested.is_set():  # Check after blocking call
                    log_message("VADWorker: Loop: Stop requested during/after record_with_vad. Breaking.")
                    break
                if not self._is_listening_active:  # Also check if VAD was stopped by another means
                    log_message("VADWorker: Loop: No longer active after record_with_vad. Breaking.")
                    break

                if success and saved_filepath_str:
                    log_message(
                        f"VADWorker: Speech detected: {saved_filepath_str} ({actual_duration:.2f}s). Setting downstream flag & emitting.")
                    self.is_processing_downstream.set();
                    self.speech_segment_recorded.emit(
                        saved_filepath_str)
                elif not success and saved_filepath_str is None:
                    pass  # Normal VAD timeout
                elif not success:
                    log_message("VADWorker: VAD recording failed (returned False but no specific error).")

                if not actual_duration or actual_duration < 0.1:  # Short pause if no/short audio
                    if self._is_listening_active and not self._stop_loop_requested.is_set():
                        self._stop_loop_requested.wait(0.05)
            except Exception as e:
                log_message(f"VADWorker: Exception in record_with_vad: {e}")
                self.vad_error.emit(f"VAD exception: {e}")
                if self._stop_loop_requested.is_set(): break  # Exit if stop requested during exception handling
                self._stop_loop_requested.wait(0.5)  # Pause after error
        log_message(f"VADWorker: run_listening_loop ended. _stop_loop_requested={self._stop_loop_requested.is_set()}")

    @pyqtSlot()
    def stop_active_listening(self):
        log_message(
            f"VADWorker: stop_active_listening called. Setting _is_listening_active=False. Prev: {self._is_listening_active}")
        self._is_listening_active = False

    @pyqtSlot()
    def fully_stop_worker_and_loop(self):
        log_message(
            "VADWorker: fully_stop_worker_and_loop called. Setting _is_listening_active=False, _stop_loop_requested=True.");
        self._is_listening_active = False;
        self._stop_loop_requested.set()


class STTWorker(QObject):
    transcription_ready = pyqtSignal(str, str);
    stt_error = pyqtSignal(str)

    def __init__(self, default_stt_model_dir_str: str, stt_params: dict):
        super().__init__();
        self.current_stt_model_dir = default_stt_model_dir_str;
        self.stt_params = stt_params
        self.num_threads = stt_params.get("num_threads", STT_DEFAULT_NUM_THREADS);
        self.provider = stt_params.get("provider", STT_DEFAULT_PROVIDER);
        self.decoding_method = stt_params.get("decoding_method", STT_DEFAULT_DECODING_METHOD)
        self._stop_requested = False  # Kept for consistency if needed later

    @pyqtSlot()
    def fully_stop_worker_and_loop(self):  # Added for consistency
        log_message("STTWorker: fully_stop_worker_and_loop called (sets _stop_requested).")
        self._stop_requested = True

    @pyqtSlot(str)
    def transcribe(self, audio_path: str):
        if self._stop_requested: log_message("STTWorker: Transcription aborted due to stop request."); return
        log_message(f"STTWorker: transcribe called for {audio_path}");
        effective_model_dir = self.current_stt_model_dir
        if not effective_model_dir or not Path(
                effective_model_dir).is_dir(): err = f"STT model directory not found or invalid: '{effective_model_dir}'"; log_message(
            f"STTWorker: Error - {err}"); self.stt_error.emit(err); return
        try:
            text, lang = transcribe_audio(audio_path=audio_path, model_dir=effective_model_dir, **self.stt_params)
            if self._stop_requested: log_message(
                "STTWorker: Transcription finished but stop was requested, discarding."); return
            try:
                if Path(audio_path).exists(): Path(audio_path).unlink(missing_ok=True); log_message(
                    f"STTWorker: Deleted temp audio file {audio_path}")
            except OSError as e_del:
                log_message(f"STTWorker: Error deleting temp audio file {audio_path}: {e_del}")

            if text:
                self.transcription_ready.emit(text, lang or "zh")
            else:
                self.stt_error.emit("Transcription failed or no speech recognized.")
        except Exception as e:
            log_message(f"STTWorker: Exception in transcribe_audio: {e}")
            self.stt_error.emit(f"STT transcription error: {e}")

    @pyqtSlot(str)
    def preload_stt_model(self, model_dir_to_preload: str):
        if self._stop_requested: log_message("STTWorker: Preload aborted due to stop request."); return
        log_message(f"STTWorker: Attempting to preload STT model from: {model_dir_to_preload}")
        if not model_dir_to_preload or not Path(model_dir_to_preload).is_dir(): log_message(
            f"STTWorker: Preload failed - Invalid model directory: {model_dir_to_preload}"); return
        try:
            recognizer = _load_asr_recognizer(model_dir=model_dir_to_preload, num_threads=self.num_threads,
                                              provider=self.provider, decoding_method=self.decoding_method)
            if self._stop_requested: log_message("STTWorker: Preload finished but stop was requested."); return
            if recognizer:
                log_message(f"STTWorker: Model at {model_dir_to_preload} preloaded/cached.")
            else:
                log_message(f"STTWorker: Preloading {model_dir_to_preload} returned None (possibly failed).")
        except Exception as e:
            log_message(f"STTWorker: Exception during STT preload for {model_dir_to_preload}: {e}")


class LLMWorker(QObject):
    response_ready = pyqtSignal(str, str);
    llm_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._stop_requested = False  # Kept for consistency

    @pyqtSlot()
    def fully_stop_worker_and_loop(self):  # Added for consistency
        log_message("LLMWorker: fully_stop_worker_and_loop called (sets _stop_requested).")
        self._stop_requested = True

    @pyqtSlot(str, dict, str)
    def process_text(self, prompt: str, llm_config_params: dict, input_lang_hint: str):
        if self._stop_requested: log_message("LLMWorker: Processing aborted due to stop request."); return
        log_message(f"LLMWorker: process_text with prompt '{prompt[:50]}...' and params {llm_config_params}")
        final_prompt = prompt;
        is_for_tts = llm_config_params.get("_is_for_tts_output_", False)
        is_preload_prompt = prompt == PRELOAD_LLM_PROMPT_EN or prompt == PRELOAD_LLM_PROMPT_ZH
        if is_for_tts and not is_preload_prompt:
            tts_instruction_zh = " 请直接提供回复内容，确保文本适合语音朗读：避免使用markdown、表格、复杂列表、过多或不常见的标点符号，也不要包含引导性语句如“以下是答案:”。"
            tts_instruction_en = " Please provide the reply content directly, ensuring the text is suitable for voice synthesis: avoid markdown, tables, complex lists, excessive or uncommon punctuation, and do not include introductory phrases like 'Here is the answer:'."
            if any('\u4e00' <= char <= '\u9fff' for char in prompt[:30]):
                final_prompt = prompt + tts_instruction_zh
            else:
                final_prompt = prompt + tts_instruction_en
            log_message(f"LLMWorker: Modified prompt for TTS: '{final_prompt[:120]}...'")

        actual_llm_params = {k: v for k, v in llm_config_params.items() if k != "_is_for_tts_output_"}

        try:
            response = get_llm_response(prompt=final_prompt, mode=actual_llm_params.get("mode", "echo"),
                                        provider=actual_llm_params.get("provider"),
                                        model_name=actual_llm_params.get("model_name"))
            if self._stop_requested: log_message(
                "LLMWorker: Processing finished but stop was requested, discarding."); return
            if response and not response.startswith("错误:"):
                if is_for_tts and not is_preload_prompt and actual_llm_params.get("mode", "echo") != "echo":
                    prefixes_to_strip = [r"好的，以下是.*：", r"当然，这是.*：", r"以下是一些.*：", r"以下是为您生成的文本：",
                                         r"Okay, here is.*:", r"Sure, here's.*:", r"Here is some text suitable for.*:"]
                    for prefix_pattern in prefixes_to_strip:
                        match = re.match(prefix_pattern, response, re.IGNORECASE)
                        if match: response = response[match.end():].lstrip(); log_message(
                            f"LLMWorker: Stripped prefix matching '{prefix_pattern}', new response starts: '{response[:50]}...'"); break
                    response = re.sub(r"^\s*([0-9]+|[一二三四五六七八九十]+)\s*[\.。、]\s*", "", response).lstrip()
                self.response_ready.emit(response, input_lang_hint)
            else:
                self.llm_error.emit(response or "LLM returned empty or null response.")
        except Exception as e:
            log_message(f"LLMWorker: Exception in get_llm_response: {e}")
            self.llm_error.emit(f"LLM API error: {e}")


class TTSWorker(QObject):
    tts_finished = pyqtSignal(bool);
    tts_error = pyqtSignal(str)

    def __init__(self, tts_params: dict):
        super().__init__();
        self.tts_params = tts_params
        self._stop_requested = False  # Kept for consistency

    @pyqtSlot()
    def fully_stop_worker_and_loop(self):  # Added for consistency
        log_message("TTSWorker: fully_stop_worker_and_loop called (sets _stop_requested).")
        self._stop_requested = True

    @pyqtSlot(str, dict, str, int, float)
    def speak(self, text_to_speak: str, tts_model_paths: Dict[str, Optional[str]], lang_hint: str, speaker_id: int,
              speed: float):
        if self._stop_requested: log_message("TTSWorker: Speak aborted due to stop request."); self.tts_finished.emit(
            False); return
        log_message(f"TTSWorker: speak text '{text_to_speak[:50]}...' (lang hint for model choice: {lang_hint})")
        model_dir_to_use = None
        if lang_hint and lang_hint.lower().startswith('en') and tts_model_paths.get('en'):
            model_dir_to_use = tts_model_paths['en']
        elif lang_hint and lang_hint.lower().startswith('zh') and tts_model_paths.get('zh'):
            model_dir_to_use = tts_model_paths['zh']
        elif tts_model_paths.get('zh'):
            model_dir_to_use = tts_model_paths['zh']
        elif tts_model_paths.get('en'):
            model_dir_to_use = tts_model_paths['en']

        if not model_dir_to_use: log_message(
            "TTSWorker: No suitable TTS model found based on hint or availability."); self.tts_error.emit(
            "No suitable TTS model for language."); self.tts_finished.emit(False); return

        log_message(f"TTSWorker: Selected model_dir: {model_dir_to_use}")
        try:
            success = speak_text(text=text_to_speak, model_dir=model_dir_to_use, speaker_id=speaker_id, speed=speed,
                                 provider=self.tts_params.get("provider", DEFAULT_TTS_PROVIDER),
                                 num_threads=self.tts_params.get("num_threads", DEFAULT_TTS_NUM_THREADS),
                                 add_silence_ms=self.tts_params.get("silence_ms", DEFAULT_TTS_SILENCE_MS))
            if self._stop_requested: log_message(
                "TTSWorker: Speak finished but stop was requested."); self.tts_finished.emit(False); return
            if not success: self.tts_error.emit("TTS synthesis failed (returned False).")
            self.tts_finished.emit(success)
        except Exception as e:
            log_message(f"TTSWorker: Exception in speak_text: {e}")
            self.tts_error.emit(f"TTS synthesis error: {e}")
            self.tts_finished.emit(False)

    @pyqtSlot(str)
    def preload_tts_model(self, model_dir_to_preload: str):
        if self._stop_requested: log_message("TTSWorker: Preload aborted due to stop request."); return
        log_message(f"TTSWorker: Attempting to preload TTS model from: {model_dir_to_preload}")
        if not model_dir_to_preload or not Path(model_dir_to_preload).is_dir(): log_message(
            f"TTSWorker: Preload failed - Invalid model directory: {model_dir_to_preload}"); return
        try:
            engine = load_tts_engine(model_dir=model_dir_to_preload,
                                     provider=self.tts_params.get("provider", DEFAULT_TTS_PROVIDER),
                                     num_threads=self.tts_params.get("num_threads", DEFAULT_TTS_NUM_THREADS))
            if self._stop_requested: log_message("TTSWorker: Preload finished but stop was requested."); return
            if engine:
                log_message(f"TTSWorker: Model at {model_dir_to_preload} preloaded/cached.")
            else:
                log_message(f"TTSWorker: Preloading {model_dir_to_preload} returned None (possibly failed).")
        except Exception as e:
            log_message(f"TTSWorker: Exception during TTS preload for {model_dir_to_preload}: {e}")


class MainApplication(QApplication):
    request_preload_stt = pyqtSignal(str)
    request_preload_llm = pyqtSignal(str, dict, str)
    request_preload_tts = pyqtSignal(str)

    def __init__(self, argv):
        super().__init__(argv);
        self._is_shutting_down = False
        log_message("MainApplication __init__ 开始");
        self.ui = VoiceAssistantUI()
        self.is_processing_user_request = False;
        self.is_processing_preload = False
        self.active_input_mode = "text";
        self.active_output_mode = "text";
        self.current_llm_params: Dict = {}
        self.tts_model_paths: Dict[str, Optional[str]] = {'en': None, 'zh': None};
        self.stt_default_model_path_str = ""
        self.preload_timer = QTimer(self);
        self.preload_timer.setSingleShot(True);
        self.preload_timer.setInterval(750);
        self.preload_timer.timeout.connect(self.execute_pending_preloads)
        self.pending_preload_tasks: List[Tuple[str, tuple]] = [];
        self.show_english_checkbox: Optional[QCheckBox] = None

        self._setup_paths_and_dirs();
        self._prepare_tts_models();
        self._setup_workers_and_threads();
        self._connect_gui_signals()
        self.ui.show();
        self._update_gui_and_worker_states();
        self.ui.status_label.setText("已准备就绪");
        log_message("MainApplication __init__ 结束")

    def _setup_paths_and_dirs(self):
        if GUI_ICONS_DIR and not os.path.exists(GUI_ICONS_DIR):
            try:
                os.makedirs(GUI_ICONS_DIR);
                log_message(f"创建了UI图标目录: {GUI_ICONS_DIR}")
            except OSError as e:
                log_message(f"错误: 创建UI图标目录 '{GUI_ICONS_DIR}' 失败: {e}")

        stt_model_path_obj = Path(STT_DEFAULT_MODEL_DIR)
        if not stt_model_path_obj.is_dir():
            log_message(
                f"警告: 默认STT模型目录 '{STT_DEFAULT_MODEL_DIR}' 未找到.");
            self.stt_default_model_path_str = ""
        else:
            self.stt_default_model_path_str = str(stt_model_path_obj.resolve());
            log_message(
                f"默认STT模型目录 '{self.stt_default_model_path_str}' 存在.")

    def _prepare_tts_models(self):
        log_message("Preparing TTS models...");
        tts_download_base_path = Path(TTS_DEFAULT_DOWNLOAD_DIR)
        if TTS_EN_URL and TTS_EN_SUBDIR:
            expected_en_path = tts_download_base_path / TTS_EN_SUBDIR
            if expected_en_path.is_dir():
                self.tts_model_paths['en'] = str(expected_en_path.resolve())
            else:
                downloaded_en_path = download_and_extract_model(TTS_EN_URL, str(tts_download_base_path), TTS_EN_SUBDIR);
                self.tts_model_paths['en'] = str(downloaded_en_path.resolve()) if downloaded_en_path else None
        if TTS_ZH_URL and TTS_ZH_SUBDIR:
            expected_zh_path = tts_download_base_path / TTS_ZH_SUBDIR
            if expected_zh_path.is_dir():
                self.tts_model_paths['zh'] = str(expected_zh_path.resolve())
            else:
                downloaded_zh_path = download_and_extract_model(TTS_ZH_URL, str(tts_download_base_path), TTS_ZH_SUBDIR);
                self.tts_model_paths['zh'] = str(downloaded_zh_path.resolve()) if downloaded_zh_path else None
        log_message(f"TTS Models prepared: EN='{self.tts_model_paths['en']}', ZH='{self.tts_model_paths['zh']}'")

    def _setup_workers_and_threads(self):
        log_message("Setting up workers and threads...");
        audio_params = {'samplerate': DEFAULT_SAMPLERATE, 'channels': DEFAULT_CHANNELS, 'dtype': DEFAULT_DTYPE}
        vad_params = {'vad_aggressiveness': DEFAULT_VAD_AGGRESSIVENESS, 'frame_ms': DEFAULT_VAD_FRAME_MS,
                      'silence_timeout_ms': DEFAULT_VAD_SILENCE_TIMEOUT_MS,
                      'min_speech_duration_ms': DEFAULT_VAD_MIN_SPEECH_DURATION_MS,
                      'max_recording_s': DEFAULT_VAD_MAX_RECORDING_S}
        self.vad_worker = VADWorker(audio_params, vad_params, TEMP_AUDIO_DIR);
        self.vad_thread = QThread(self);
        self.vad_thread.setObjectName("VADThread")
        self.vad_worker.moveToThread(self.vad_thread);
        self.vad_thread.started.connect(self.vad_worker.run_listening_loop);
        self.vad_worker.speech_segment_recorded.connect(self.on_speech_segment_recorded)
        self.vad_worker.vad_error.connect(lambda err_msg: self.on_pipeline_error("VAD", err_msg));
        self.vad_worker.status_update.connect(
            lambda s: self.ui.status_label.setText(s) if hasattr(self.ui, 'status_label') else None);
        self.vad_thread.start()

        stt_params = {'provider': STT_DEFAULT_PROVIDER, 'num_threads': STT_DEFAULT_NUM_THREADS,
                      'decoding_method': STT_DEFAULT_DECODING_METHOD}
        self.stt_worker = STTWorker(self.stt_default_model_path_str, stt_params);
        self.stt_thread = QThread(self);
        self.stt_thread.setObjectName("STTThread")
        self.stt_worker.moveToThread(self.stt_thread);
        self.stt_worker.transcription_ready.connect(self.on_stt_result);
        self.stt_worker.stt_error.connect(lambda err: self.on_pipeline_error("STT", err))
        self.request_preload_stt.connect(self.stt_worker.preload_stt_model);
        self.stt_thread.start()

        self.llm_worker = LLMWorker();
        self.llm_thread = QThread(self);
        self.llm_thread.setObjectName("LLMThread")
        self.llm_worker.moveToThread(self.llm_thread);
        self.llm_worker.response_ready.connect(self.on_llm_response);
        self.llm_worker.llm_error.connect(lambda err: self.on_pipeline_error("LLM", err))
        self.request_preload_llm.connect(self.llm_worker.process_text);  # Modified for direct connection
        self.llm_thread.start()

        tts_params = {"provider": DEFAULT_TTS_PROVIDER, "num_threads": DEFAULT_TTS_NUM_THREADS,
                      "silence_ms": DEFAULT_TTS_SILENCE_MS}
        self.tts_worker = TTSWorker(tts_params);
        self.tts_thread = QThread(self);
        self.tts_thread.setObjectName("TTSThread")
        self.tts_worker.moveToThread(self.tts_thread);
        self.tts_worker.tts_finished.connect(self.on_tts_finished);
        self.tts_worker.tts_error.connect(lambda err: self.on_pipeline_error("TTS", err))
        self.request_preload_tts.connect(self.tts_worker.preload_tts_model);
        self.tts_thread.start()
        log_message("Workers and threads setup complete and started.")

    def _connect_gui_signals(self):
        log_message("Connecting GUI signals...");
        self.ui.userInputEntered.connect(self.handle_user_text_input);
        self.ui.mic_btn.clicked.connect(self.handle_mic_button_click)
        if hasattr(self.ui, 'interaction_mode_group'): self.ui.interaction_mode_group.buttonClicked.connect(
            self.on_interaction_mode_changed)
        if hasattr(self.ui, 'model_button_group'): self.ui.model_button_group.buttonClicked.connect(
            self.on_llm_model_changed)

        if hasattr(self.ui, 'cb_show_english'):
            self.show_english_checkbox = self.ui.cb_show_english
            if self.show_english_checkbox:
                log_message("Found '显示英文' checkbox via direct attribute 'self.ui.cb_show_english'.")
        elif hasattr(self.ui, 'findChild'):
            self.show_english_checkbox = self.ui.findChild(QCheckBox, "showEnglishCheckbox")
            if self.show_english_checkbox:
                log_message("Found '显示英文' checkbox by objectName 'showEnglishCheckbox'.")

        if not self.show_english_checkbox:
            log_message("警告: 未能找到 '显示英文' 复选框。英文显示控制将默认显示英文。")

        self._update_current_llm_params_from_gui();
        log_message("GUI signals connected.")

    def schedule_preload(self, task_type: str, *args):
        self.pending_preload_tasks = [t for t in self.pending_preload_tasks if t[0] != task_type]
        self.pending_preload_tasks.append((task_type, args))
        if not self.is_processing_user_request and not self.is_processing_preload:
            self.preload_timer.start()

    def execute_pending_preloads(self):
        if self.is_processing_user_request: log_message(
            "Skipping preloads as system is busy processing user request."); return
        if self.is_processing_preload: log_message("Skipping preloads, another preload batch is in progress."); return

        if not self.pending_preload_tasks: return

        log_message(f"Executing pending preloads: {self.pending_preload_tasks}");
        self.is_processing_preload = True
        self._set_ui_interaction_enabled_based_on_state()

        for task_type, args in self.pending_preload_tasks:
            if task_type == "stt":
                model_path_to_preload = args[0]
                if model_path_to_preload and Path(model_path_to_preload).is_dir():
                    self.request_preload_stt.emit(model_path_to_preload)
                else:
                    log_message(f"Skipping STT preload, invalid path: {model_path_to_preload}")
            elif task_type == "llm":
                llm_params_arg, dummy_prompt_arg, dummy_hint_arg = args[0], args[1], args[2]
                log_message(f"Requesting LLM preload via signal: {llm_params_arg} with prompt '{dummy_prompt_arg}'")
                self.request_preload_llm.emit(dummy_prompt_arg, llm_params_arg, dummy_hint_arg)
            elif task_type == "tts":
                model_path, _ = args[0], args[1]
                if model_path and Path(model_path).is_dir():
                    log_message(f"Requesting TTS model preload via signal: {model_path}")
                    self.request_preload_tts.emit(model_path)
                else:
                    log_message(f"Skipping TTS preload, invalid path: {model_path}")

        self.pending_preload_tasks.clear();
        QTimer.singleShot(500, self._finish_preload_batch_ui_reset)

    def _finish_preload_batch_ui_reset(self):
        self.is_processing_preload = False
        if not self.is_processing_user_request:
            self._set_ui_interaction_enabled_based_on_state()
        log_message("Finished preload batch UI reset.")

    def on_interaction_mode_changed(self):
        log_message(
            f"Interaction mode change requested. System busy (user request): {self.is_processing_user_request}, (preload): {self.is_processing_preload}")
        if self.is_processing_user_request or self.is_processing_preload:
            log_message("Interaction mode change deferred (system busy).");
            return
        self._update_gui_and_worker_states()

    def on_llm_model_changed(self):
        log_message(
            f"LLM model change requested. System busy (user request): {self.is_processing_user_request}, (preload): {self.is_processing_preload}")
        if self.is_processing_user_request or self.is_processing_preload:
            log_message("LLM model change deferred (system busy).");
            return
        self._update_current_llm_params_from_gui()
        if self.current_llm_params and self.current_llm_params.get("mode") != "echo":
            is_kimi_or_deepseek = "moonshot" in self.current_llm_params.get("provider",
                                                                            "").lower() or "siliconflow" in self.current_llm_params.get(
                "provider", "").lower()
            preload_prompt = PRELOAD_LLM_PROMPT_ZH if is_kimi_or_deepseek else PRELOAD_LLM_PROMPT_EN
            preload_lang_hint = "zh" if is_kimi_or_deepseek else "en"
            self.schedule_preload("llm", self.current_llm_params.copy(), preload_prompt,
                                  preload_lang_hint)  # Pass a copy

    def _update_gui_and_worker_states(self):
        # This method is called on mode changes or after pipeline completion.
        # It re-evaluates the VAD state based on the current active_input_mode.
        log_message(
            f"Updating GUI and worker states. Current mode: Input='{self.active_input_mode}', Output='{self.active_output_mode}'. System busy (user request): {self.is_processing_user_request}, (preload): {self.is_processing_preload}");
        if self.is_processing_user_request or self.is_processing_preload:
            log_message("Mode update deferred as system is busy.");
            return

        selected_interaction_button = self.ui.interaction_mode_group.checkedButton()
        if selected_interaction_button:
            mode_text = selected_interaction_button.text()
            mode_map = {"语音输入语音输出": ("voice", "voice"), "文本输入语音输出": ("text", "voice"),
                        "语音输入文本输出": ("voice", "text"), "纯文本交互": ("text", "text")}
            self.active_input_mode, self.active_output_mode = mode_map.get(mode_text, ("text", "text"))
        else:  # Fallback if no button is checked (should not happen in normal operation)
            self.active_input_mode, self.active_output_mode = "text", "text"

        log_message(
            f"Active modes newly set/confirmed: Input='{self.active_input_mode}', Output='{self.active_output_mode}'");
        self._set_ui_interaction_enabled_based_on_state()  # Update UI enable/disable state

        if self.active_input_mode == "voice":
            if not self.stt_default_model_path_str:
                self.ui.status_label.setText("警告: STT模型未配置!");
                if self.vad_worker:
                    log_message(
                        "_update_gui_and_worker_states (voice mode): STT not configured, calling stop_active_listening.")
                    # Direct call as this is from main thread UI interaction
                    self.vad_worker.stop_active_listening()
            else:
                if self.vad_worker:
                    log_message(
                        "_update_gui_and_worker_states (voice mode): STT configured, calling start_active_listening.")
                    # Direct call
                    self.vad_worker.start_active_listening()
                self.ui.status_label.setText("语音模式: 请说话...");
                self.schedule_preload("stt", self.stt_default_model_path_str)
        else:  # Text input mode
            if self.vad_worker:
                log_message("_update_gui_and_worker_states (text mode): Calling stop_active_listening.")
                # Direct call
                self.vad_worker.stop_active_listening()
            self.ui.status_label.setText("文本模式: 请输入...")

        if self.active_output_mode == "voice":
            preferred_tts_model_path = self.tts_model_paths.get('zh') or self.tts_model_paths.get('en')
            if preferred_tts_model_path:
                dummy_tts_text = PRELOAD_TTS_TEXT_ZH if self.tts_model_paths.get(
                    'zh') == preferred_tts_model_path else PRELOAD_TTS_TEXT_EN
                self.schedule_preload("tts", preferred_tts_model_path, dummy_tts_text)

        # No need to call _update_current_llm_params_from_gui() here
        # as it's called by on_llm_model_changed or during init.

    def _update_current_llm_params_from_gui(self):
        selected_button = self.ui.model_button_group.checkedButton();
        new_params = {}
        if not selected_button:
            new_params = {"mode": "echo"}
        else:
            model_text = selected_button.text()
            if model_text == "本地模型":
                new_params = {"mode": "local", "provider": "local"}
            elif model_text == "在线模型 - Kimi":
                new_params = {"mode": "official", "provider": "moonshot", "model_name": "moonshot-v1-8k"}
            elif model_text == "在线模型 - DeepSeek":  # Use the model name from your working code
                new_params = {"mode": "official", "provider": "siliconflow",
                              "model_name": "Qwen/QwQ-32B"}  # This was in your working code
            else:
                new_params = {"mode": "echo"}

        if new_params != self.current_llm_params:
            self.current_llm_params = new_params;
            log_message(f"LLM config updated: {self.current_llm_params}")

    def _pipeline_step_start(self):
        log_message("Pipeline step START: Setting is_processing_user_request = True")
        self.is_processing_user_request = True
        self._set_ui_interaction_enabled_based_on_state()

    def _pipeline_step_complete(self, error: bool = False):
        log_message(f"Pipeline step COMPLETE (Error: {error}). Setting is_processing_user_request = False");
        was_main_processing = self.is_processing_user_request  # Capture state before reset
        self.is_processing_user_request = False

        # This is crucial: clear VAD's downstream processing flag.
        if self.vad_worker and hasattr(self.vad_worker, 'is_processing_downstream'):
            log_message("Pipeline step COMPLETE: Clearing VAD is_processing_downstream flag.")
            self.vad_worker.is_processing_downstream.clear()

        self._set_ui_interaction_enabled_based_on_state()  # Update UI immediately

        if not error:
            # Don't set status label here if it will be reset by _update_gui_and_worker_states
            # self.ui.status_label.setText("已准备就绪") # Temporarily remove
            pass
        else:
            self.ui.status_label.setText("处理出错，已准备")  # Keep error status until next action

        # According to working code, call _update_gui_and_worker_states after processing.
        # This will re-evaluate VAD state (e.g., restart listening if still in voice mode).
        if was_main_processing:  # Only if it was a full user request cycle
            log_message("Pipeline step COMPLETE: Requesting _update_gui_and_worker_states to refresh VAD state.")
            self._update_gui_and_worker_states()

    def _set_ui_interaction_enabled_based_on_state(self):
        can_interact_overall = not self.is_processing_user_request and not self.is_processing_preload
        is_text_input_active_mode = self.active_input_mode == "text";
        is_voice_input_active_mode = self.active_input_mode == "voice"
        # log_message(f"_set_ui_interaction_enabled: can_interact={can_interact_overall}, text_active={is_text_input_active_mode}, voice_active={is_voice_input_active_mode}")

        self.ui.input_field.setEnabled(can_interact_overall and is_text_input_active_mode);
        self.ui.send_button.setEnabled(can_interact_overall and is_text_input_active_mode);
        self.ui.mic_btn.setEnabled(can_interact_overall and is_voice_input_active_mode)

        if hasattr(self.ui, 'interaction_mode_group'):
            for button in self.ui.interaction_mode_group.buttons(): button.setEnabled(can_interact_overall)
        if hasattr(self.ui, 'model_button_group'):
            for button in self.ui.model_button_group.buttons(): button.setEnabled(can_interact_overall)

        if hasattr(self.ui, 'cb_show_subtitles'):
            self.ui.cb_show_subtitles.setEnabled(can_interact_overall)
        if self.show_english_checkbox:
            self.show_english_checkbox.setEnabled(can_interact_overall)

    @pyqtSlot(str)
    def handle_user_text_input(self, user_text: str):
        if self.is_processing_user_request or self.is_processing_preload:
            log_message("Text input: System busy, ignoring.")
            self.ui.add_message("系统繁忙，请稍后再试。", "bot");
            self.ui.input_field.clear();
            return
        if not user_text.strip(): return

        self._pipeline_step_start()
        self.ui.add_message(user_text, "user");
        self.ui.input_field.clear();
        self.ui.status_label.setText("LLM 处理中...")

        llm_params_for_call = self.current_llm_params.copy()
        if self.active_output_mode == "voice": llm_params_for_call["_is_for_tts_output_"] = True

        input_lang_hint_for_llm = "en" if is_text_predominantly_english(user_text) else "zh"
        log_message(f"User text input, determined lang hint for LLM: {input_lang_hint_for_llm}")

        # Using QTimer.singleShot as in working code
        QTimer.singleShot(0,
                          lambda: self.llm_worker.process_text(user_text, llm_params_for_call, input_lang_hint_for_llm))

    @pyqtSlot()
    def handle_mic_button_click(self):
        # Logic based on the working code provided
        log_message(
            f"Mic button clicked. Active input: {self.active_input_mode}, Preloading: {self.is_processing_preload}, User request: {self.is_processing_user_request}")
        if self.active_input_mode != "voice":
            log_message("Mic button: Ignored, not in voice mode.");
            return

        if self.is_processing_user_request or self.is_processing_preload:
            log_message("Mic button: Ignored, system busy (processing user request or preloading).")
            self.ui.status_label.setText("处理中...");  # Or more specific status
            return

        if not self.stt_default_model_path_str:
            self.ui.status_label.setText("错误: STT模型未配置!")
            self.ui.add_message("STT模型未配置，无法使用语音输入。", "bot")
            log_message("Mic button: STT model not configured.")
            return

        log_message("Mic button: Setting pipeline_step_start and activating VAD.")
        self._pipeline_step_start()  # Set busy flag as per working code
        self.ui.status_label.setText("语音模式: 请说话...")
        if self.vad_worker:
            # Direct call as this is from main thread UI interaction
            self.vad_worker.start_active_listening()

    @pyqtSlot(str)
    def on_speech_segment_recorded(self, audio_path: str):
        # Logic based on the working code provided
        log_message(
            f"Speech segment recorded: {audio_path}. Current is_processing_user_request: {self.is_processing_user_request}")
        if not self.is_processing_user_request:  # If not already busy from mic click (or auto VAD)
            log_message("on_speech_segment_recorded: Not already processing, calling _pipeline_step_start.")
            self._pipeline_step_start()
        else:
            log_message("on_speech_segment_recorded: Already processing, ensuring UI state is updated.")
            self._set_ui_interaction_enabled_based_on_state()  # Ensure UI reflects busy state

        # The working code does NOT stop VAD here. This means VAD might continue to listen
        # and potentially queue up more segments if user speaks continuously.
        # If this is desired, keep it. If single-utterance processing is preferred,
        # then `self.vad_worker.stop_active_listening()` should be called here.
        # For now, matching the working code:
        # if self.vad_worker:
        #     log_message("on_speech_segment_recorded: (Following working code) NOT stopping VAD active listening here.")
        #     # self.vad_worker.stop_active_listening() # This would be for single utterance

        log_message(f"Audio segment for STT: {audio_path}.")
        self.ui.status_label.setText("语音识别中...")
        # Using QTimer.singleShot as in working code
        QTimer.singleShot(0, lambda: self.stt_worker.transcribe(audio_path))

    @pyqtSlot(str, str)
    def on_stt_result(self, text: str, lang: str):
        log_message(f"STT Result: '{text}', Estimated Lang from STT: {lang}")

        stt_text_is_english = is_text_predominantly_english(text)
        should_display_stt = True

        if stt_text_is_english and self.show_english_checkbox and not self.show_english_checkbox.isChecked():
            log_message("STT is English, but '显示英文' is unchecked. Not adding to chat history directly.")
            should_display_stt = False

        if should_display_stt:
            self.ui.add_message(f"{text} (STT)", "user")
        else:
            log_message(f"STT (English, not displayed due to checkbox): {text}")

        self.ui.status_label.setText("LLM 处理中...")
        llm_params_for_call = self.current_llm_params.copy()
        if self.active_output_mode == "voice": llm_params_for_call["_is_for_tts_output_"] = True

        effective_lang_for_llm = "en" if stt_text_is_english else "zh"
        log_message(f"Determined lang hint for LLM based on STT content: {effective_lang_for_llm}")

        # Using QTimer.singleShot as in working code
        QTimer.singleShot(0, lambda: self.llm_worker.process_text(text, llm_params_for_call, effective_lang_for_llm))

    @pyqtSlot(str, str)
    def on_llm_response(self, response: str, input_lang_hint: str):
        log_message(f"LLM Response: '{response[:50]}...' (LLM was hinted with: {input_lang_hint})")

        response_is_english = is_text_predominantly_english(response)
        should_display_response = True

        if response_is_english:
            if self.show_english_checkbox:
                if not self.show_english_checkbox.isChecked():
                    log_message("LLM response is English, and '显示英文' is unchecked. Will not display in chat.")
                    should_display_response = False
            elif not self.show_english_checkbox:
                log_message("警告: '显示英文' checkbox对象未找到，英文文本将默认显示。")

        if should_display_response:
            self.ui.add_message(response, "bot")
        else:
            log_message(f"LLM Response (English, not displayed because '显示英文' is unchecked): {response[:100]}...")

        QTimer.singleShot(50, lambda: self._process_tts_after_display(response, input_lang_hint, response_is_english))

    def _process_tts_after_display(self, response: str, input_lang_hint_llm_received: str, response_is_english: bool):
        if self.active_output_mode == "voice":
            self.ui.status_label.setText("语音合成中...")
            chosen_lang_for_tts_output = "en" if response_is_english else "zh"
            log_message(f"Determined language for TTS output based on response content: {chosen_lang_for_tts_output}")

            speaker_id = DEFAULT_ZH_SPEAKER_ID;
            speed = DEFAULT_ZH_SPEED
            final_lang_hint_for_tts_worker = "zh"

            if chosen_lang_for_tts_output == 'en' and self.tts_model_paths.get('en'):
                speaker_id = DEFAULT_EN_SPEAKER_ID;
                speed = DEFAULT_EN_SPEED;
                final_lang_hint_for_tts_worker = "en"
                log_message(f"TTS decision: Using English model. Speaker: {speaker_id}, Speed: {speed}")
            elif chosen_lang_for_tts_output == 'zh' and self.tts_model_paths.get('zh'):
                log_message(f"TTS decision: Using Chinese model. Speaker: {speaker_id}, Speed: {speed}")
            elif self.tts_model_paths.get('zh'):
                log_message(f"TTS decision: Fallback to Chinese model. Speaker: {speaker_id}, Speed: {speed}")
            elif self.tts_model_paths.get('en'):
                speaker_id = DEFAULT_EN_SPEAKER_ID;
                speed = DEFAULT_EN_SPEED;
                final_lang_hint_for_tts_worker = "en"
                log_message(f"TTS decision: Fallback to English model. Speaker: {speaker_id}, Speed: {speed}")
            else:
                log_message("TTS decision: No suitable model path found. Skipping TTS.");
                self.on_pipeline_error("TTS", "No suitable model for TTS output.")  # Changed to call on_pipeline_error
                return

            QMetaObject.invokeMethod(
                self.tts_worker, "speak", Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, response), Q_ARG(dict, self.tts_model_paths),
                Q_ARG(str, final_lang_hint_for_tts_worker),
                Q_ARG(int, speaker_id), Q_ARG(float, speed)
            )
        else:
            self._pipeline_step_complete()

    @pyqtSlot(bool)
    def on_tts_finished(self, success: bool):
        log_message(f"TTS playback finished (Success: {success}).")
        if not success: self.ui.status_label.setText("TTS 错误")
        self._pipeline_step_complete(error=not success)

    @pyqtSlot(str, str)
    def on_pipeline_error(self, worker_name: str, error_message: str):
        log_message(f"Error from {worker_name}: {error_message}");
        self.ui.add_message(f"{worker_name} 错误: {error_message.splitlines()[0]}", "bot");
        self._pipeline_step_complete(error=True)

    def closeEvent(self, event):
        if self._is_shutting_down:
            if event: event.accept()
            return

        self._is_shutting_down = True
        log_message("Close event triggered. Shutting down application...");
        self.preload_timer.stop()

        worker_thread_pairs = [
            (self.vad_worker, self.vad_thread),
            (self.stt_worker, self.stt_thread),
            (self.llm_worker, self.llm_thread),
            (self.tts_worker, self.tts_thread)
        ]

        # Signal all workers to stop
        for worker, thread in worker_thread_pairs:
            if worker and hasattr(worker, 'fully_stop_worker_and_loop'):
                obj_name = thread.objectName() if thread else "worker_no_thread"
                log_message(f"Invoking fully_stop_worker_and_loop for {obj_name}")
                # For VADWorker, fully_stop_worker_and_loop sets the event which is thread-safe.
                # For others, it sets a flag. If their tasks are short, QueuedConnection is fine.
                # If tasks can be long and _stop_requested needs to be checked within the task,
                # this queued call might be delayed. However, for shutdown, this is usually acceptable.
                QMetaObject.invokeMethod(worker, "fully_stop_worker_and_loop", Qt.ConnectionType.QueuedConnection)
            elif worker and hasattr(worker, '_stop_requested'):  # Fallback
                worker._stop_requested = True

        threads_to_process = []
        for worker, thread in worker_thread_pairs:
            if thread and thread.isRunning():
                threads_to_process.append(thread)
            elif thread:
                log_message(f"Thread {thread.objectName()} already stopped or not started.")

        for thread in threads_to_process:
            log_message(f"Requesting quit for thread {thread.objectName()}...")
            thread.quit()

        for thread in threads_to_process:
            log_message(f"Waiting for thread {thread.objectName()} to finish...")
            if not thread.wait(1800):  # Timeout from previous version
                log_message(f"Thread {thread.objectName()} did not quit gracefully after 1.8s, terminating.");
                thread.terminate()
                if not thread.wait(500):
                    log_message(f"Thread {thread.objectName()} termination wait also timed out.")
            else:
                log_message(f"Thread {thread.objectName()} finished.")

        try:
            if os.path.exists(TEMP_AUDIO_DIR):
                cleaned_count = 0
                for item in Path(TEMP_AUDIO_DIR).glob("vad_*.wav"):
                    try:
                        item.unlink(missing_ok=True)
                        cleaned_count += 1
                    except OSError as e_file:
                        log_message(f"Error deleting temp file {item}: {e_file}")
                if cleaned_count > 0:
                    log_message(f"Cleaned {cleaned_count} temp audio file(s) in {TEMP_AUDIO_DIR}")
                else:
                    log_message(f"No temp audio files found to clean in {TEMP_AUDIO_DIR}")
        except Exception as e:
            log_message(f"Error during temp file cleanup: {e}")

        if event:
            log_message("Proceeding with super().closeEvent() for GUI close.")
            super().closeEvent(event)
        else:
            log_message("Cleanup complete (triggered by aboutToQuit). Application will now exit.")
            # Removed QApplication.instance().quit() as it was causing recursion
            # and is not needed when called from aboutToQuit.


if __name__ == '__main__':
    QCoreApplication.setApplicationName("VoiceAssistantMainApp")
    QCoreApplication.setOrganizationName("MyOrganization")
    QCoreApplication.setApplicationVersion("1.0")

    log_message("应用程序启动 (__main__)")
    app = MainApplication(sys.argv)
    app.aboutToQuit.connect(lambda: app.closeEvent(None))  # Ensures cleanup on quit
    exit_code = app.exec()
    log_message(f"Application finished with exit code: {exit_code}")
    sys.exit(exit_code)