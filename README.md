# RAG智能问答系统

基于本地知识库的RAG智能问答系统，支持PDF、DOCX、TXT等多种文档格式，使用Ollama本地大模型进行问答。

**项目地址**: https://github.com/chenyy125/rag-qa-system

## 功能特点

- 📁 支持多种文档格式（PDF、DOCX、TXT）
- 🔍 基于向量数据库的文档检索
- 💬 支持多轮对话，具有会话记忆功能
- 📊 实时显示知识库状态
- 🚀 支持多种Ollama模型切换
- ⚡ 混合模式：TF-IDF检索 + Ollama生成（无需网络下载嵌入模型）

## 环境要求

- Python 3.10+
- Ollama（用于部署本地大模型）
- 至少8GB内存（推荐16GB+）

## 安装步骤

### 1. 安装Ollama

访问 [Ollama官方网站](https://ollama.com/) 下载并安装Ollama。

### 2. 下载模型

```bash
ollama pull deepseek-r1:7b
```

或选择其他模型：
```bash
ollama pull qwen2:7b
ollama pull llama3:8b
```

### 3. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 运行Web应用

```bash
streamlit run app.py
```

### 使用步骤

1. 在左侧面板上传文档（支持PDF、DOCX、TXT格式）
2. 点击"构建知识库"按钮处理文档
3. 在右侧问答区域输入问题并点击"提问"
4. 查看回答结果和参考来源
5. 支持多轮对话，可查看对话历史

## 项目结构

```
├── app.py                    # Streamlit Web应用主文件
├── requirements.txt          # Python依赖包列表
├── README.md                 # 项目说明文档
├── .gitignore                # Git忽略配置
├── docs/                     # 知识库文档文件夹
│   ├── nlp_introduction.txt          # 自然语言处理介绍
│   ├── transformer_architecture.txt  # Transformer架构详解
│   ├── bert_introduction.txt         # BERT模型介绍
│   ├── rag_technology.txt            # RAG技术说明
│   └── llm_introduction.txt          # 大型语言模型介绍
└── src/                      # 核心模块目录
    ├── document_loader.py    # 文档加载器
    ├── simple_rag.py         # 简化版RAG引擎（TF-IDF+Ollama混合模式）
    └── rag_chain.py          # Ollama模式RAG问答链
```

## 运行模式说明

| 模式 | 所需依赖 | 特点 |
|------|----------|------|
| **TF-IDF模式** | 仅需Python依赖 | 无需Ollama，开箱即用 |
| **Ollama模式** | Ollama + 模型 | 需要下载模型，回答质量更高 |
| **混合模式** | Python依赖 + Ollama | TF-IDF检索 + Ollama生成（推荐） |

## 测试示例

### 相关问题
1. 什么是自然语言处理？
2. Transformer架构的核心组件有哪些？
3. BERT的预训练任务是什么？
4. RAG技术的优势是什么？
5. 大型语言模型面临哪些挑战？

## License

MIT License