## pip install慢
```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 安装依赖
```shell
pip install -r requirements.txt

```

## 模型下载
也可以手动下载模型 模型格式HuggingFace
https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```shell
python down_deepseek.py
```
## 微调
```shell
python finetune_deepseek.py
# 验证
python test_deepseek.py
```

## 模型转换（HuggingFace → GGUF 格式）
ollama、llama.cpp 能直接运行的格式
vllm 支持HuggingFace
```shell
# 需要用llama.cpp仓库的convert_hf_to_gguf.py脚本来转换
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
# 如果不量化，保留模型的效果
python llama.cpp/convert_hf_to_gguf.py ./DeepSeek-R1-Distill-Qwen-1.5B  --outtype f16 --verbose --outfile DeepSeek-R1-Distill-Qwen-1.5B.gguf
# 如果需要量化（加速并有损效果），直接执行下面脚本就可以
python llama.cpp/convert_hf_to_gguf.py ./DeepSeek-R1-Distill-Qwen-1.5B  --outtype q8_0 --verbose --outfile DeepSeek-R1-Distill-Qwen-1.5B.gguf

```

## Python3.13 openssl 问题
No module named ‘_ssl‘，openssl3 以上lib目录变成lib4 
```shell
# 安装openssl
wget https://github.com/openssl/openssl/releases/download/openssl-3.0.16/openssl-3.0.16.tar.gz
tar -zxvf openssl-3.0.16.tar.gz
cd openssl-3.0.16
./config --prefix=/usr/local/openssl3 --openssldir=/usr/local/openssl3
make && make install
# 主要是这步骤
ln -s /usr/local/openssl3/lib64 /usr/local/openssl3/lib

# 安装python
wget https://www.python.org/ftp/python/3.13.2/Python-3.13.2.tgz
cd Python-3.13.2
./configure --with-openssl=/usr/local/openssl3
make && make install

# 加入PATH
export PATH=$PATH:/usr/local/python3/bin
# 验证
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"
```