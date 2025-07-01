import subprocess
import threading
import queue

class LLMController:
    def __init__(self, model_path):
        self.process = subprocess.Popen(
            ["./llm_demo", model_path, "2048", "4096"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd="/home/elf/deepseek"
        )
        self.response_queue = queue.Queue()
        self._start_reader_thread()
        self.query("")  # 预热模型

    def _start_reader_thread(self):
        def read_output():
            while True:
                line = self.process.stdout.readline()
                if "<h1>nk>" in line:
                    self.response_queue.put(None)  # 结束标记
                    break
                self.response_queue.put(line.strip())
        threading.Thread(target=read_output, daemon=True).start()

    def query(self, input_text):
        # 清空队列残留数据
        while not self.response_queue.empty():
            self.response_queue.get()
        
        # 发送输入
        self.process.stdin.write(input_text + "\n")
        self.process.stdin.flush()
        
        # 收集输出
        response = []
        while True:
            item = self.response_queue.get()
            if item is None:
                break
            response.append(item)
        return " ".join(response)

# 使用示例
if __name__ == '__main__':
    llm = LLMController("DeepSeek-R1-Distill-Quen-1.5B_MMA8_RK3588.rkllm")
    
    # 语音输入（示例）
    input_text = "你是谁？"
    
    # 获取回复
    response = llm.query(input_text)
    print("模型回复:", response)
    
    # 文本转语音（伪代码）
    # text_to_speech(response)