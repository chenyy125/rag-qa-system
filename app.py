import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from document_loader import load_document, split_documents, load_documents_from_folder
from simple_rag import create_simple_rag_chain
from rag_chain import create_rag_chain, test_ollama_connection

st.set_page_config(
    page_title="RAG智能问答系统",
    page_icon="🔍",
    layout="wide"
)

def init_session_state():
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0
    if "use_ollama" not in st.session_state:
        st.session_state.use_ollama = False
    if "show_ollama_warning" not in st.session_state:
        st.session_state.show_ollama_warning = False
    if "ollama_available" not in st.session_state:
        st.session_state.ollama_available = False
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "deepseek-r1:7b"

init_session_state()

st.title("🔍 RAG智能问答系统")
st.subheader("基于本地知识库的智能问答")

st.sidebar.header("⚙️ 设置")
st.session_state.use_ollama = st.sidebar.checkbox("使用Ollama大模型（需要先下载模型）", value=st.session_state.use_ollama)

if st.session_state.use_ollama:
    st.session_state.ollama_model = st.sidebar.selectbox(
        "选择模型",
        ["deepseek-r1:7b", "qwen2:7b", "llama3:8b"],
        index=0
    )
    
    st.sidebar.info("📌 下载模型步骤：")
    st.sidebar.code("ollama pull qwen2:7b")
    
    if st.sidebar.button("🔍 测试Ollama连接"):
        with st.spinner("正在测试Ollama连接..."):
            success, msg = test_ollama_connection(st.session_state.ollama_model)
            if success:
                st.session_state.ollama_available = True
                st.sidebar.success("✅ Ollama连接成功！")
                st.sidebar.info(f"模型响应: {msg[:50]}...")
            else:
                st.session_state.ollama_available = False
                st.sidebar.error(f"❌ Ollama连接失败: {msg}")
                st.sidebar.warning("💡 解决方案：\n1. 打开命令提示符\n2. 运行: ollama pull qwen2:7b\n3. 等待模型下载完成")

if st.session_state.use_ollama and not st.session_state.ollama_available:
    st.warning("⚠️ Ollama模型未下载，将使用简化版TF-IDF引擎。如需使用Ollama，请先下载模型。")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 文档管理")
    
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if st.button("📥 构建知识库"):
        if uploaded_files:
            with st.spinner("正在处理文档..."):
                documents = []
                
                for file in uploaded_files:
                    temp_path = f"./temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    try:
                        docs = load_document(temp_path)
                        documents.extend(docs)
                        st.session_state.uploaded_files.append(file.name)
                        os.remove(temp_path)
                    except Exception as e:
                        st.error(f"加载 {file.name} 失败: {e}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                if documents:
                    chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
                    st.session_state.chunk_count = len(chunks)
                    
                    use_ollama_mode = st.session_state.use_ollama and st.session_state.ollama_available
                    st.session_state.rag = create_simple_rag_chain(
                        chunks, 
                        use_ollama=use_ollama_mode, 
                        ollama_model=st.session_state.ollama_model
                    )
                    
                    st.success(f"✅ 知识库构建完成！")
                    st.info(f"📊 处理了 {len(documents)} 个文档，生成了 {len(chunks)} 个文本块")
                else:
                    st.warning("没有成功加载任何文档")
        else:
            st.warning("请先上传文档")
    
    st.header("📊 知识库状态")
    st.info(f"已上传文档: {len(st.session_state.uploaded_files)} 个")
    st.info(f"文本块数量: {st.session_state.chunk_count} 个")
    
    engine_type = "Ollama" if (st.session_state.use_ollama and st.session_state.ollama_available) else "TF-IDF"
    st.info(f"当前引擎: {engine_type}")
    
    if st.session_state.uploaded_files:
        st.subheader("已上传的文档")
        for file in st.session_state.uploaded_files:
            st.write(f"- {file}")

with col2:
    st.header("💬 问答交互")
    
    if st.session_state.rag is None:
        st.info("请先上传文档并构建知识库")
        st.subheader("📚 示例文档")
        sample_docs = [
            "nlp_introduction.txt - 自然语言处理介绍",
            "transformer_architecture.txt - Transformer架构详解",
            "bert_introduction.txt - BERT模型介绍",
            "rag_technology.txt - RAG技术说明",
            "llm_introduction.txt - 大型语言模型介绍",
            "ollama_docs.txt - Ollama官方文档",
            "langchain_docs.txt - LangChain文档",
            "streamlit_docs.txt - Streamlit文档",
            "git_tutorial.txt - Git教程"
        ]
        st.write("已内置9份文档作为知识库样本：")
        for doc in sample_docs:
            st.write(f"📄 {doc}")
        
        if st.button("📖 使用内置文档构建知识库"):
            with st.spinner("正在加载内置文档..."):
                documents = load_documents_from_folder("./docs")
                chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
                st.session_state.chunk_count = len(chunks)
                
                use_ollama_mode = st.session_state.use_ollama and st.session_state.ollama_available
                st.session_state.rag = create_simple_rag_chain(
                    chunks, 
                    use_ollama=use_ollama_mode, 
                    ollama_model=st.session_state.ollama_model
                )
                
                st.session_state.uploaded_files = [f for f in os.listdir("./docs") if f.endswith(('.txt', '.pdf', '.docx'))]
                
                st.success(f"✅ 知识库构建完成！")
                st.info(f"📊 处理了 {len(documents)} 个文档，生成了 {len(chunks)} 个文本块")
                st.rerun()
    else:
        user_input = st.text_input("请输入您的问题：", key="question_input")
        
        if st.button("🚀 提问") and user_input:
            with st.spinner("正在检索并生成回答..."):
                try:
                    answer, sources = st.session_state.rag.generate_answer(user_input)
                except (AttributeError, ImportError):
                    try:
                        from rag_chain import ask_question
                        answer, sources = ask_question(st.session_state.rag, user_input)
                    except Exception as e:
                        answer = f"问答失败: {str(e)}"
                        sources = []
                
                st.session_state.chat_history.append((user_input, answer))
                
                st.success("回答生成完成！")
                st.write(f"**回答：** {answer}")
                
                if sources and "文档中未找到相关答案" not in answer:
                    st.write("**参考来源：**")
                    for i, doc in enumerate(sources, 1):
                        if isinstance(doc, dict):
                            source = doc.get('source', 'Unknown')
                            similarity = doc.get('similarity', 0)
                            st.write(f"{i}. {source} (相似度: {similarity:.2f})")
                        else:
                            source = doc.metadata.get('source', 'Unknown')
                            st.write(f"{i}. {source}")
        
        if st.session_state.chat_history:
            st.subheader("📜 对话历史")
            for i, (question, answer) in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"问题 {len(st.session_state.chat_history) - i + 1}: {question[:30]}..."):
                    st.write(f"**问题：** {question}")
                    st.write(f"**回答：** {answer}")

if st.sidebar.button("🔄 重置对话"):
    st.session_state.chat_history = []
    st.success("对话已重置")

if st.sidebar.button("🗑️ 清空知识库"):
    st.session_state.rag = None
    st.session_state.chat_history = []
    st.session_state.uploaded_files = []
    st.session_state.chunk_count = 0
    st.success("知识库已清空")