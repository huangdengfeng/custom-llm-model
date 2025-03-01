from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 加载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 准备训练数据
train_data = [
    {
        "question": "你是谁?",
        "answer": "我是黄登峰。"
    },
    {
        "question": "你的名字是什么？",
        "answer": "黄登峰"
    },
    {
        "question": "你是做什么的？",
        "answer": "我是深圳一家公司打工的牛马程序员。"
    },
    # 在这里添加更多的问答对
]

test_data = [
    {
        "question": "你的名字是什么?",
        "answer": "我的名字是黄登峰。"
    }
]
def format_instruction(example):
    """格式化输入输出对"""
    return f"Human: {example['question']}\n\nAssistant: {example['answer']}"

# 转换数据格式
train_formatted_data = [{"text": format_instruction(item)} for item in train_data]
test_formatted_data = [{"text": format_instruction(item)} for item in test_data]
train_dataset = Dataset.from_list(train_formatted_data)
test_dataset = Dataset.from_list(test_formatted_data)

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# 对数据集进行预处理
train_tokenized_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

test_tokenized_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=test_dataset.column_names
)
output_dir = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B_CUSTOM"

# 训练参数设置
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model()
# 保存tokenizer
tokenizer.save_pretrained(output_dir)
