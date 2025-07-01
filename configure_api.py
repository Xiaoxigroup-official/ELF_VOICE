# --- START OF FILE configure_api.py (v1.6 - 更新 SiliconFlow 默认模型) ---

import json
import os
import platform
import sys
from pathlib import Path

CONFIG_FILENAME = ".llm_config.json"
CONFIG_PATH = Path(__file__).parent / CONFIG_FILENAME

# --- 参数检查 ---
# (保持不变)
if len(sys.argv) > 1:
    print("*" * 60)
    print("警告：检测到您在运行 configure_api.py 时提供了额外的命令行参数:")
    print(f"  参数列表: {sys.argv[1:]}")
    print("configure_api.py 脚本本身不接受任何命令行参数。")
    print("请直接运行 'python configure_api.py'。")
    print("如果您想使用 '--mode' 或 '--provider' 等参数，您可能需要运行:")
    print("  python llm_interaction.py [参数...]")
    print("  或者")
    print("  python main_parrot_loop_sherpa.py [参数...]")
    print("*" * 60)

def load_config():
    """加载现有的配置 (如果存在)"""
    # (保持不变)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                if isinstance(config_data, dict):
                    print(f"成功从 '{CONFIG_FILENAME}' 加载现有配置。")
                    return config_data
                else:
                    print(f"警告: 配置文件 '{CONFIG_FILENAME}' 格式不正确 (不是 JSON 对象)。将创建新配置。")
                    return {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 读取配置文件 '{CONFIG_FILENAME}' 失败: {e}. 将创建新配置。")
    return {}

def save_config(config_data):
    """保存配置到文件"""
    # (保持不变)
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print(f"配置已成功保存到: {CONFIG_PATH.resolve()}")
        try:
            os.chmod(CONFIG_PATH, 0o600)
            print("尝试设置文件权限为 600 (仅用户可读写)。")
        except OSError:
            print("提示: 未能设置文件权限 (可能是 Windows 系统或权限不足)。")
        except Exception as e_perm:
             print(f"设置文件权限时发生意外错误: {e_perm}")
    except IOError as e:
        print(f"错误: 保存配置文件失败: {e}")
    except Exception as e_save:
         print(f"保存配置文件时发生意外错误: {e_save}")

def get_input(prompt, default=None):
    """获取用户输入，支持默认值和空输入的处理"""
    # (保持不变)
    if default:
        prompt_with_default = f"{prompt} (当前: '{default}', 直接回车保持不变): "
    else:
        prompt_with_default = f"{prompt}: "
    value = input(prompt_with_default).strip()
    return default if not value and default is not None else value

def configure_provider(provider_name, current_config):
    """通用配置函数，处理不同 provider 的字段"""
    print(f"\n--- 配置 {provider_name} API ---")
    config_data = {}
    is_baidu = provider_name.lower() == 'baidu'

    # 1. API Key
    key_prompt = "请输入你的 API Key"
    if is_baidu: key_prompt += " (Access Key ID)"
    config_data['api_key'] = get_input(key_prompt, current_config.get('api_key'))
    if not config_data['api_key']: print("错误: API Key 不能为空。"); return None

    # 2. Secret Key (仅百度)
    if is_baidu:
        config_data['secret_key'] = get_input("请输入你的 Secret Key (Secret Access Key)", current_config.get('secret_key'))
        if not config_data['secret_key']: print("错误: 百度 API 需要 Secret Key。"); return None

    # 3. Base URL / Endpoint
    default_base_url = current_config.get('base_url')
    base_url_prompt = "请输入 API Endpoint (Base URL)"
    provider_lower = provider_name.lower()

    if provider_lower == 'baidu':
        if not default_base_url: default_base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
        base_url_prompt += " (例如 ERNIE-Bot-turbo/4 的地址)"
    elif provider_lower == 'openai':
        if not default_base_url: default_base_url = "https://api.openai.com/v1"
        base_url_prompt += " (官方 API 可留空使用默认)"
    elif provider_lower == 'deepseek':
        if not default_base_url: default_base_url = "https://api.deepseek.com/v1"
    elif provider_lower == 'zhipu':
        if not default_base_url: default_base_url = "https://open.bigmodel.cn/api/paas/v4/"
    elif provider_lower == 'moonshot':
        if not default_base_url: default_base_url = "https://api.moonshot.cn/v1"
    elif provider_lower == 'siliconflow':
        if not default_base_url: default_base_url = "https://api.siliconflow.cn/v1"
        base_url_prompt += " (例如 https://api.siliconflow.cn/v1)"

    config_data['base_url'] = get_input(base_url_prompt, default_base_url)
    if not config_data['base_url'] and provider_lower != 'openai':
        print(f"错误: {provider_name} API 需要 Endpoint (Base URL)。"); return None
    if provider_lower == 'openai' and not config_data['base_url']:
        config_data['base_url'] = None

    # 4. Default Model
    default_model_suggestion = current_config.get('default_model')
    if provider_lower == 'baidu' and not default_model_suggestion: default_model_suggestion = 'ERNIE-Bot-turbo'
    elif provider_lower == 'openai' and not default_model_suggestion: default_model_suggestion = 'gpt-3.5-turbo'
    elif provider_lower == 'deepseek' and not default_model_suggestion: default_model_suggestion = 'deepseek-chat'
    elif provider_lower == 'zhipu' and not default_model_suggestion: default_model_suggestion = 'glm-4'
    elif provider_lower == 'moonshot' and not default_model_suggestion: default_model_suggestion = 'moonshot-v1-8k'
    elif provider_lower == 'siliconflow':
        # --- ★★★ 修改点: 更新 SiliconFlow 的默认模型建议 ★★★ ---
        if not default_model_suggestion: default_model_suggestion = 'Qwen/QwQ-32B' # 已更新

    config_data['default_model'] = get_input("请输入默认使用的模型名称", default_model_suggestion)
    if not config_data['default_model']:
        config_data['default_model'] = None

    return config_data

def display_env_var_instructions(provider_data):
    """显示如何设置环境变量的提示 (可选)"""
    # (保持不变)
    print("\n--- (可选) 通过环境变量覆盖配置 ---")
    provider_name = provider_data.get('_provider_name', 'unknown')
    api_key_var = ""; secret_key_var = ""; base_url_var = ""; model_var = ""

    if provider_name == 'baidu':
        api_key_var = "QIANFAN_ACCESS_KEY"; secret_key_var = "QIANFAN_SECRET_KEY";
        base_url_var = "BAIDU_API_BASE_URL"; model_var = "BAIDU_DEFAULT_MODEL";
    else:
        provider_upper = provider_name.upper()
        api_key_var = f"{provider_upper}_API_KEY"
        base_url_var = f"{provider_upper}_BASE_URL"
        model_var = f"{provider_upper}_DEFAULT_MODEL"

    system = platform.system(); set_cmd = "set" if system == "Windows" else "export"; ps_cmd = "$env:" if system == "Windows" else ""
    print(f"你也可以设置以下环境变量来覆盖 '{provider_name}' 的配置文件或提供缺失值：")
    if api_key_var:
        print(f"  {api_key_var}='your_api_key'")
        print(f"    (例: CMD '{set_cmd} {api_key_var}=...' / Bash '{set_cmd} {api_key_var}=...' / PowerShell '{ps_cmd}{api_key_var}=...')")
    if secret_key_var:
        print(f"  {secret_key_var}='your_secret_key'")
        print(f"    (例: CMD '{set_cmd} {secret_key_var}=...' / Bash '{set_cmd} {secret_key_var}=...' / PowerShell '{ps_cmd}{secret_key_var}=...')")
    if base_url_var:
        if provider_name != 'baidu':
            print(f"  {base_url_var}='your_base_url'")
            print(f"    (例: CMD '{set_cmd} {base_url_var}=...' / Bash '{set_cmd} {base_url_var}=...' / PowerShell '{ps_cmd}{base_url_var}=...')")
    if model_var:
        print(f"  {model_var}='your_model_name'")
        print(f"    (例: CMD '{set_cmd} {model_var}=...' / Bash '{set_cmd} {model_var}=...' / PowerShell '{ps_cmd}{model_var}=...')")
    print("环境变量通常具有最高优先级。")


if __name__ == "__main__":
    # (参数检查保持不变)
    if len(sys.argv) > 1:
        print("\n" + "="*60)
        print("严重警告：configure_api.py 被调用时附加了命令行参数！")
        print(f"参数: {sys.argv[1:]}")
        print("此脚本不接受参数，请直接运行 'python configure_api.py'")
        print("如果看到关于 '--mode' 或 '--provider' 的错误，说明你运行了错误的脚本！")
        print("="*60 + "\n")

    known_providers = ['baidu', 'deepseek', 'zhipu', 'openai', 'moonshot', 'siliconflow']
    known_providers.sort()

    config = load_config()

    # --- ★★★ 修改点: 更新版本号 ★★★ ---
    print("欢迎使用 LLM API 配置工具！(v1.6 - 更新 SiliconFlow 默认模型)")
    print(f"配置文件: {CONFIG_PATH.resolve()}")


    while True:
        # (菜单显示逻辑保持不变)
        print("\n选择要配置/更新的 API 提供商:")
        provider_choices = {str(i+1): p for i, p in enumerate(known_providers)}
        for index_str, provider_name in provider_choices.items():
            status_indicator = "(已配置)" if provider_name in config else "(未配置)"
            print(f"  {index_str}. {provider_name.capitalize()} {status_indicator}")

        other_option_index = len(known_providers) + 1
        print(f"  {other_option_index}. other (自定义 OpenAI 兼容 API)")
        print("  d. 显示环境变量设置说明")
        print("  q. 退出")

        choice = input(f"输入选项 (1-{other_option_index}, d, q): ").strip().lower()

        provider_key_to_configure = None
        new_config_data_for_provider = None

        if choice == 'q':
            break
        elif choice == 'd':
            # (显示环境变量说明的逻辑不变)
            print("\n选择要显示环境变量设置说明的 Provider:")
            configured_providers = list(config.keys())
            if not configured_providers:
                print("尚未配置任何 Provider。"); continue
            display_choices = {str(i+1): p for i, p in enumerate(configured_providers)}
            for k, v in display_choices.items(): print(f"  {k}. {v.capitalize()}")
            display_choice = input(f"输入选项 (1-{len(configured_providers)}): ").strip()
            selected_provider_name = display_choices.get(display_choice)
            if selected_provider_name and selected_provider_name in config:
                provider_data_for_display = config[selected_provider_name].copy(); provider_data_for_display['_provider_name'] = selected_provider_name
                display_env_var_instructions(provider_data_for_display)
            else: print("无效选择。")
            continue

        # (处理用户选择 Provider 的逻辑保持不变)
        try:
            choice_int = int(choice)
            if 1 <= choice_int <= len(known_providers):
                provider_key_to_configure = provider_choices[choice]
            elif choice_int == other_option_index:
                custom_name = input("请输入自定义 Provider 名称 (例如 'my_local_llm'): ").strip().lower()
                if custom_name:
                    provider_key_to_configure = custom_name
                    print(f"将为 '{provider_key_to_configure}' 配置 OpenAI 兼容 API。")
                    if provider_key_to_configure in known_providers:
                        print(f"提示: 您输入的名称 '{provider_key_to_configure}' 与已知 Provider 相同。")
                else: print("自定义名称不能为空。"); continue
            else: print("无效的数字选项。"); continue
        except ValueError:
            print("无效输入。请输入数字选项、'd' 或 'q'。"); continue

        # (调用配置函数并保存的逻辑保持不变)
        if provider_key_to_configure:
            current_provider_config = config.get(provider_key_to_configure, {})
            new_config_data_for_provider = configure_provider(
                provider_key_to_configure,
                current_provider_config
            )
            if new_config_data_for_provider:
                config[provider_key_to_configure] = new_config_data_for_provider
                save_config(config)
            else:
                print(f"未能完成对 '{provider_key_to_configure}' 的配置。")

    print("\n配置完成或已退出。")

# --- END OF FILE configure_api.py (v1.6 - 更新 SiliconFlow 默认模型) ---