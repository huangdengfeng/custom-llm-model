from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_response(prompt, model, tokenizer, max_length=512):
    # 将输入格式化为训练时的格式
    formatted_prompt = f"Human: {prompt}\n\nAssistant:"
    
    # 对输入进行编码
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取Assistant的回答部分
    response = response.split("Assistant:")[-1].strip()
    return response

def main():
    # 加载微调后的模型和分词器
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_CUSTOM"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    # 准备测试问题
    test_questions = [
        "你是谁?",
        "你的名字是什么?",
        "你是做什么的?",
    ]
    
    # 测试模型回答
    print("开始测试模型回答：")
    print("-" * 50)
    for question in test_questions:
        print(f"问题: {question}")
        response = generate_response(question, model, tokenizer)
        print(f"回答: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()