import os
import subprocess
import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import traceback
import select
import threading
import queue

# --- 配置区域 ---
LLM_DEMO_WORKING_DIR = "/home/elf/local/deepseek"
LLM_DEMO_EXECUTABLE_NAME = "./llm_demo"
MODEL_FILE_NAME = "DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm"
MODEL_ARGS = ["2048", "4096"]
DEFAULT_MODEL_ID = "deepseek-rk3588-qwen-1.5b"

# 超时设置
PROCESS_READY_TIMEOUT = 30  # 增加启动超时时间
REPLY_SILENCE_TIMEOUT = 3.0  # 增加回复超时时间
SELECT_TIMEOUT_PER_LOOP = 0.2

MODEL_READY_MARKER_STDERR = "I rkllm: prompt_postfix: <｜Assistant｜>"
# --- 配置区域结束 ---

app = FastAPI(
    title="RKLLM OpenAI-Compatible API (v0.6.0 - Multi-turn Chat)",
    description="支持多轮对话的RKLLM API服务",
    version="0.6.0",
)


def setup_environment():
    """设置环境：执行chmod和export命令"""
    try:
        # 切换到工作目录
        original_dir = os.getcwd()
        os.chdir(LLM_DEMO_WORKING_DIR)

        # 执行 chmod +x llm_demo
        executable_path = os.path.join(LLM_DEMO_WORKING_DIR, LLM_DEMO_EXECUTABLE_NAME)
        if os.path.exists(executable_path):
            subprocess.run(["chmod", "+x", "llm_demo"], check=True, cwd=LLM_DEMO_WORKING_DIR)
            print(f"Successfully set executable permission for {executable_path}")
        else:
            print(f"Warning: {executable_path} not found, skipping chmod")

        # 设置 LD_LIBRARY_PATH 环境变量
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if current_ld_path:
            new_ld_path = f"{current_ld_path}:{LLM_DEMO_WORKING_DIR}"
        else:
            new_ld_path = LLM_DEMO_WORKING_DIR

        os.environ['LD_LIBRARY_PATH'] = new_ld_path
        print(f"Updated LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

        # 恢复原始目录
        os.chdir(original_dir)

    except subprocess.CalledProcessError as e:
        print(f"Error executing chmod command: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set executable permission: {e}")
    except Exception as e:
        print(f"Error setting up environment: {e}")
        raise HTTPException(status_code=500, detail=f"Environment setup failed: {e}")


# 全局进程管理
class LLMProcessManager:
    def __init__(self):
        self.process = None
        self.lock = threading.Lock()
        self.ready = False
        self.environment_setup = False

    def get_process(self):
        with self.lock:
            # 首次调用时设置环境
            if not self.environment_setup:
                setup_environment()
                self.environment_setup = True

            if self.process is None or self.process.poll() is not None:
                self._start_process()
            return self.process

    def _start_process(self):
        """启动LLM进程并等待就绪"""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
                self.process.wait()

        model_file_path = os.path.join(LLM_DEMO_WORKING_DIR, MODEL_FILE_NAME)
        executable_path = os.path.join(LLM_DEMO_WORKING_DIR, LLM_DEMO_EXECUTABLE_NAME)

        if not os.path.isfile(executable_path):
            raise HTTPException(status_code=500, detail=f"Exe not found: {executable_path}")
        if not os.access(executable_path, os.X_OK):
            raise HTTPException(status_code=500, detail=f"No exe perm: {executable_path}")
        if not os.path.isfile(model_file_path):
            raise HTTPException(status_code=500, detail=f"Model not found: {model_file_path}")

        command = [executable_path, model_file_path] + MODEL_ARGS

        # 使用更新后的环境变量启动进程
        env = os.environ.copy()

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=LLM_DEMO_WORKING_DIR,
            bufsize=1,
            universal_newlines=True,
            env=env  # 传递环境变量
        )

        # 等待模型就绪
        self._wait_for_ready()

    def _wait_for_ready(self):
        """等待模型就绪标记"""
        deadline = time.time() + PROCESS_READY_TIMEOUT

        while time.time() < deadline:
            if self.process.poll() is not None:
                raise HTTPException(status_code=500, detail="LLM process exited during startup")

            ready_stderr, _, _ = select.select([self.process.stderr], [], [], SELECT_TIMEOUT_PER_LOOP)
            if ready_stderr:
                err_line = self.process.stderr.readline()
                if not err_line:
                    break

                if MODEL_READY_MARKER_STDERR in err_line:
                    self.ready = True
                    # 清理残留输出
                    time.sleep(0.5)
                    while select.select([self.process.stdout], [], [], 0.01)[0]:
                        self.process.stdout.readline()
                    while select.select([self.process.stderr], [], [], 0.01)[0]:
                        self.process.stderr.readline()
                    return

            # 清理stdout防止阻塞
            while select.select([self.process.stdout], [], [], 0.01)[0]:
                self.process.stdout.readline()

        if not self.ready:
            raise HTTPException(status_code=500, detail="LLM model not ready within timeout")

    def send_prompt_and_get_response(self, prompt: str) -> str:
        """发送提示并获取响应"""
        with self.lock:
            if not self.process or self.process.poll() is not None:
                raise HTTPException(status_code=500, detail="LLM process not available")

            try:
                # 发送提示
                self.process.stdin.write(prompt + "\n")
                self.process.stdin.flush()

                # 读取响应
                response_lines = []
                last_output_time = time.time()

                while True:
                    if self.process.poll() is not None:
                        break

                    ready_stdout, _, _ = select.select([self.process.stdout], [], [], SELECT_TIMEOUT_PER_LOOP)

                    if ready_stdout:
                        line = self.process.stdout.readline()
                        if not line:
                            break

                        last_output_time = time.time()
                        line_stripped = line.strip()

                        # 过滤输出，只保留robot的回复
                        if line_stripped.startswith("robot: "):
                            content = line_stripped[7:]  # 移除"robot: "前缀
                            response_lines.append(content)
                        elif line_stripped.startswith("user: robot: "):
                            content = line_stripped[13:]  # 移除"user: robot: "前缀
                            response_lines.append(content)
                        elif response_lines and line_stripped:  # 如果已经开始收集回复，保留后续非空行
                            response_lines.append(line_stripped)
                    else:
                        # 检查是否超时
                        if time.time() - last_output_time > REPLY_SILENCE_TIMEOUT:
                            break

                    # 清理stderr
                    while select.select([self.process.stderr], [], [], 0.01)[0]:
                        self.process.stderr.readline()

                return "\n".join(response_lines).strip()

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error communicating with LLM: {str(e)}")

    def cleanup(self):
        """清理进程"""
        with self.lock:
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                except:
                    self.process.kill()
                    self.process.wait()
            self.process = None
            self.ready = False


# 全局进程管理器
llm_manager = LLMProcessManager()


# --- OpenAI 兼容的数据模型 ---
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None


# --- 函数定义 ---
def prepare_prompt_for_llm_demo(messages: List[ChatMessage]) -> str:
    """将OpenAI格式的消息转换为LLM demo格式"""
    # 只取最后一条用户消息，实现简单的单轮对话
    # 如果需要真正的多轮对话，需要根据LLM的具体格式要求来构建完整对话历史
    last_user_message = ""
    for msg in reversed(messages):
        if msg.role == "user":
            last_user_message = msg.content
            break

    return last_user_message if last_user_message else "hello"


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported.")

    try:
        # 确保进程运行
        llm_manager.get_process()

        # 准备提示
        prompt = prepare_prompt_for_llm_demo(request.messages)

        # 获取响应
        response_text = llm_manager.send_prompt_and_get_response(prompt)

        if not response_text:
            response_text = "Sorry, I couldn't generate a response."

        # 构建响应
        response_message = ChatMessage(role="assistant", content=response_text)
        choice = Choice(index=0, message=response_message, finish_reason="stop")

        usage_stats = Usage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(prompt.split()) + len(response_text.split())
        )

        return ChatCompletionResponse(
            model=request.model or DEFAULT_MODEL_ID,
            choices=[choice],
            usage=usage_stats
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/")
async def read_root():
    return {"message": "RKLLM OpenAI-compatible API server (v0.6.0) is running."}


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        process = llm_manager.get_process()
        if process and process.poll() is None:
            return {"status": "healthy", "llm_process": "running"}
        else:
            return {"status": "unhealthy", "llm_process": "not running"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    llm_manager.cleanup()


if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=9000)
    finally:
        llm_manager.cleanup()