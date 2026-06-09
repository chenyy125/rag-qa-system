from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain(retriever, model_name="deepseek-r1:7b"):
    llm = ChatOllama(model=model_name, temperature=0, max_tokens=2048)
    
    template = """
基于以下参考文档回答用户的问题。

参考文档：
{context}

用户问题：{question}

重要提示：
1. 请仔细阅读并理解参考文档中的内容
2. 仅使用参考文档中的信息进行回答
3. 如果参考文档中没有相关信息，请明确回答"文档中未找到相关答案"
4. 回答要简洁明了，不要添加额外信息

回答：
"""
    
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    chain = prompt | llm | StrOutputParser()
    
    return {"llm_chain": chain, "retriever": retriever}

def ask_question(chain_dict, question, chat_history=[]):
    retriever = chain_dict["retriever"]
    llm_chain = chain_dict["llm_chain"]
    
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    result = llm_chain.invoke({"context": context, "question": question})
    
    return result, docs

def test_ollama_connection(model_name="deepseek-r1:7b"):
    try:
        llm = ChatOllama(model=model_name)
        response = llm.invoke("Hello")
        return True, response.content
    except Exception as e:
        return False, str(e)