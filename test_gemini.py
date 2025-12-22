import google.generativeai as genai
import os

# ==========================================
# 1. 配置鉴权 (Configuration)
# ==========================================
# 去这里免费申请: https://aistudio.google.com/app/apikey
# 只有有了这个 Key，Google 才知道你是谁
my_api_key = "AIzaSyDRVcCJz5X7V7SflRNjCQJ-gZ0Fvg13bl0"

try:
    genai.configure(api_key=my_api_key)
except Exception as e:
    print(f"配置出错，请检查 API Key 是否正确。错误信息: {e}")
    exit()

# ==========================================
# 2. 初始化模型 (Model Instantiation)
# ==========================================
# 我们使用 'gemini-1.5-flash'，它速度极快且免费额度足够。
# 如果需要更强的推理能力，可以改成 'gemini-1.5-pro'。
# Java 类比: ModelService service = new ModelService("gemini-1.5-flash");
model = genai.GenerativeModel('gemini-1.5-flash')

# ==========================================
# 3. 准备输入 (Prompt Engineering)
# ==========================================
# 这里就是你以后要把“GitHub代码”塞进去的地方
prompt = "你好 Gemini，我正在学习开发 AI Agent。请用一句话鼓励我，并用 Python 写一个打印 'Hello Agent' 的函数。"

print("正在发送请求给 Gemini (这需要几秒钟)...")

# ==========================================
# 4. 调用接口 (API Call)
# ==========================================
# 这是一次网络请求，类似 Java 里的 RestTemplate.postForObject()
try:
    response = model.generate_content(prompt)
    
    # ==========================================
    # 5. 处理响应 (Response Handling)
    # ==========================================
    print("-" * 30)
    print("Gemini 回复:")
    print("-" * 30)
    print(response.text) # 获取纯文本回复

except Exception as e:
    print(f"请求失败: {e}")