# --- START OF FILE llm_interaction.py (v2.2 - 基于 v2.1, 无核心逻辑修改) ---

import os
import argparse
import platform
import sys
import json
from pathlib import Path
import traceback
from typing import Optional, Dict, Any

# --- 配置文件 ---
CONFIG_FILENAME = ".llm_config.json"
CONFIG_PATH = Path(__file__).parent / CONFIG_FILENAME

# --- 依赖导入 ---
_openai_available = False
_qianfan_available = False

# OpenAI (用于兼容 API)
try:
    from openai import OpenAI, APIConnectionError, APIError
    _openai_available = True
    print("OK: 导入 'openai' 库。")
except ImportError:
    print("警告: 'openai' 库未找到。无法使用 OpenAI 兼容 API (如 DeepSeek, Zhipu, Moonshot, OpenAI)。")
    class OpenAI: pass # 占位符
    class APIConnectionError(Exception): pass
    class APIError(Exception): pass

# Baidu Qianfan SDK
try:
    import qianfan
    _qianfan_available = True
    print("OK: 导入 'qianfan' 库。")
except ImportError:
    print("警告: 'qianfan' 库未找到。无法使用 'baidu' provider。请安装: pip install qianfan")
    qianfan = None # 避免 NameError

# --- 配置加载器 (与 v2.1 相同) ---
_config_cache: Optional[Dict[str, Any]] = None
def load_llm_config() -> Dict[str, Any]:
    """从文件加载 LLM 配置并缓存"""
    global _config_cache
    if _config_cache is not None: return _config_cache
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                if isinstance(config_data, dict):
                    print(f"OK: 从 '{CONFIG_FILENAME}' 加载配置。")
                    _config_cache = config_data; return _config_cache
                else: print(f"错误: 配置文件 '{CONFIG_FILENAME}' 格式错误。")
        except (json.JSONDecodeError, IOError) as e: print(f"错误: 读取/解析配置文件 '{CONFIG_FILENAME}' 失败: {e}")
    else: print(f"警告: 未找到 LLM 配置文件 '{CONFIG_FILENAME}'。")
    _config_cache = {}; return _config_cache

# --- 核心函数 (与 v2.1 相同) ---
def get_llm_response(prompt: str, provider: str, mode: str, model_name: Optional[str] = None) -> str:
    """根据输入提示获取响应，使用统一配置文件。"""
    print(f"\n--- LLM 交互 ---"); effective_mode = mode.lower()
    print(f"模式: {effective_mode}");
    if effective_mode != 'echo': print(f"Provider: {provider}")
    print(f"Prompt: '{prompt[:60]}...'")

    if effective_mode == "echo":
        response = f"Echo: {prompt}"; print(f"响应: {response[:60]}..."); print("-----------------------"); return response

    config = load_llm_config(); provider_config = config.get(provider, {})
    if not provider_config and effective_mode != 'echo':
         if effective_mode == 'local':
             print(f"警告: 配置中未找到 '{provider}'，尝试环境变量...");
             provider_config = {'base_url': os.environ.get("OPENAI_BASE_URL"), 'api_key': os.environ.get("OPENAI_API_KEY", "dummy-key"), 'default_model': os.environ.get("LLM_MODEL_NAME")}
             if not provider_config.get('base_url'): error_msg = "错误: 'local' 模式需 API Base URL。"; print(error_msg); print("-----------------------"); return error_msg
         else: error_msg = f"错误: 配置中未找到 Provider '{provider}'。请运行 configure_api.py。"; print(error_msg); print("-----------------------"); return error_msg

    final_model_name = model_name or provider_config.get('default_model')
    if not final_model_name and effective_mode != 'echo':
        error_msg = f"错误: 未确定模型名称 (命令行或配置 '{provider}' 中均未指定)。"; print(error_msg); print("-----------------------"); return error_msg
    print(f"模型: {final_model_name}")

    # === Baidu Qianfan ===
    if provider == 'baidu':
        if not _qianfan_available or qianfan is None: error_msg = "错误: 需 'qianfan' 库。请安装。"; print(error_msg); print("-----------------------"); return error_msg
        api_key = provider_config.get('api_key'); secret_key = provider_config.get('secret_key')
        if not api_key or not secret_key: error_msg = f"错误: 百度配置缺少 'api_key' 或 'secret_key'。"; print(error_msg); print("-----------------------"); return error_msg
        print(f"认证: AK/SK")
        try:
            chat_comp = qianfan.ChatCompletion(ak=api_key, sk=secret_key)
            print("请求: 正在发送到百度千帆...");
            resp = chat_comp.do(model=final_model_name, messages=[{"role": "user", "content": prompt}])
            if hasattr(resp, 'body') and isinstance(resp.body, dict) and 'result' in resp.body: response = resp.body['result']; print(f"响应: 成功接收。")
            elif isinstance(resp, dict) and 'result' in resp: response = resp['result']; print(f"响应: 成功接收 (字典格式)。")
            else: response = f"错误: 百度 API 返回未知结构。"; print(response); print(f"原始响应: {resp}")
            print(f"响应内容: '{response[:60]}...'"); print("-----------------------"); return response
        except qianfan.errors.QianfanError as e: error_msg = f"错误: 百度 API 错误: {e}"; print(error_msg); print("提示: 检查 AK/SK、账户、模型名、网络。"); print("-----------------------"); return error_msg
        except Exception as e: error_msg = f"错误: 百度交互意外错误: {e.__class__.__name__}: {e}"; print(error_msg); traceback.print_exc(); print("-----------------------"); return error_msg

    # === OpenAI 兼容 API ===
    else:
        if not _openai_available: error_msg = f"错误: 'openai' 库未找到 (Provider '{provider}')。"; print(error_msg); print("-----------------------"); return error_msg
        if OpenAI is None or not callable(OpenAI): error_msg = f"错误: OpenAI 类未初始化。"; print(error_msg); print("-----------------------"); return error_msg
        api_key = provider_config.get('api_key'); base_url = provider_config.get('base_url')
        if not api_key: error_msg = f"错误: Provider '{provider}' 配置缺少 'api_key'。"; print(error_msg); print("-----------------------"); return error_msg
        if provider != 'openai' and provider != 'local' and not base_url: error_msg = f"错误: Provider '{provider}' 配置缺少 'base_url'。"; print(error_msg); print("-----------------------"); return error_msg
        print(f"认证: API Key"); print(f"Base URL: {base_url or 'OpenAI 默认'}")
        try:
            client = OpenAI(base_url=base_url if base_url else None, api_key=api_key, timeout=90.0 if effective_mode == 'official' else 60.0)
            print(f"请求: 正在发送到 {provider}...");
            chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=final_model_name, stream=False)
            if chat_completion.choices and chat_completion.choices[0].message and chat_completion.choices[0].message.content: response = chat_completion.choices[0].message.content; print(f"响应: 成功接收。")
            else:
                finish_reason="未知";
                if chat_completion.choices and chat_completion.choices[0].finish_reason: finish_reason = chat_completion.choices[0].finish_reason
                response = f"错误: {provider} API 返回无效响应或空内容。原因: {finish_reason}"; print(response)
            print(f"响应内容: '{response[:60]}...'"); print("-----------------------"); return response
        except APIConnectionError as e: error_msg = f"错误: 连接 API ({base_url or '默认'}) 失败: {e}"; print(error_msg); print("提示: 检查网络、防火墙、Base URL。"); print("-----------------------"); return error_msg
        except APIError as e: error_body = getattr(e.response, 'text', str(e.response)); error_msg = f"错误: {provider} API 返回错误: Status={getattr(e, 'status_code', 'N/A')}, Body={error_body[:500]}"; print(error_msg); print("提示: 检查 API Key、账户、模型名、速率。"); print("-----------------------"); return error_msg
        except Exception as e: error_msg = f"错误: {provider} 交互意外错误: {e.__class__.__name__}: {e}"; print(error_msg); traceback.print_exc(); print("-----------------------"); return error_msg

# --- 用于测试的主函数 (与 v2.2 相同，示例不变) ---
if __name__ == "__main__":
    if not CONFIG_PATH.exists() and '--mode' in sys.argv and ('official' in sys.argv or 'local' in sys.argv):
        # ... (检查配置文件存在逻辑) ...
        is_local_or_official = False; # ... 省略检查代码 ...
        if is_local_or_official: print(f"错误: 未找到 '{CONFIG_FILENAME}'。请运行 configure_api.py。"); exit(1)

    parser = argparse.ArgumentParser(description="LLM 交互 v2.2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prompt", type=str, nargs='?', default="你好，请介绍一下你自己。", help="提示。")
    parser.add_argument("--mode", type=str, choices=["local", "official", "echo"], default="official", help="交互模式。")
    parser.add_argument("--provider", type=str, default=None, help="当 mode='official' 时【必需】，指定 API 提供商。")
    parser.add_argument("--model", type=str, default=None, help="指定模型名称 (覆盖默认)。")
    parser.add_argument("--check-deps", action="store_true", help="检查依赖库。")
    args = parser.parse_args()

    final_provider = args.provider;
    if args.mode == 'local': final_provider = 'local'
    elif args.mode == 'official' and not args.provider: parser.error("--mode=official 时，必须指定 --provider")
    elif args.mode == 'echo': final_provider = None

    if args.check_deps:
        print("\n--- 依赖检查 ---"); print(f"'openai' 库: {'OK' if _openai_available else '未安装!'}"); print(f"'qianfan' 库: {'OK' if _qianfan_available else '未安装!'}"); print("-----------------"); exit(0)

    print("=" * 30); print("LLM 交互测试 (v2.2)"); print("=" * 30)
    print(f"模式: {args.mode}");
    if final_provider: print(f"Provider: {final_provider}")
    config = load_llm_config(); provider_config = config.get(final_provider, {}) if final_provider else {}
    if final_provider:
        display_model = args.model or provider_config.get('default_model', '未配置默认'); print(f"模型 (命令行 > 配置): {display_model}")
        if final_provider == 'baidu': print("认证: AK/SK (从配置)")
        else: print("认证: API Key (从配置)"); print(f"Base URL: {provider_config.get('base_url', '未配置') or 'OpenAI 默认'}")
        if not provider_config: print(f"警告: 未在 '{CONFIG_FILENAME}' 找到 Provider '{final_provider}' 配置!")

    response = get_llm_response(prompt=args.prompt, provider=final_provider, mode=args.mode, model_name=args.model)
    print("\n--- 测试结果 ---"); print(f"最终响应: {response}"); print("=" * 30)

    # 示例部分保持不变，但现在 moonshot 可以通过 --provider moonshot 调用了
    print("\n运行示例:")
    print("  1. 配置 API: python configure_api.py")
    print("  2. 测试 Echo: python llm_interaction.py --mode echo \"...\"")
    print("  3. 测试 Local: python llm_interaction.py --mode local --model your-model \"...\"")
    print("  4. 测试 Baidu: python llm_interaction.py --mode official --provider baidu --model ERNIE-Bot-turbo \"...\"")
    print("  5. 测试 DeepSeek: python llm_interaction.py --mode official --provider deepseek --model deepseek-chat \"...\"")
    print("  6. 测试 Zhipu: python llm_interaction.py --mode official --provider zhipu --model glm-4 \"...\"")
    print("  7. 测试 Moonshot: python llm_interaction.py --mode official --provider moonshot --model moonshot-v1-8k \"...\"") # 添加 Kimi 示例
    print("  8. 检查依赖: python llm_interaction.py --check-deps")

# --- END OF FILE llm_interaction.py (v2.2 - 基于 v2.1, 无核心逻辑修改) ---