import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

class SimpleRAG:
    def __init__(self, use_ollama=False, ollama_model="qwen2:7b"):
        self.chunks = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = None
        self.use_ollama = use_ollama and OLLAMA_AVAILABLE
        self.ollama_model = ollama_model
        self.llm_chain = None
        if self.use_ollama:
            self._init_ollama_chain()
    
    def _init_ollama_chain(self):
        if OLLAMA_AVAILABLE:
            llm = ChatOllama(model=self.ollama_model, temperature=0, max_tokens=2048)
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
            self.llm_chain = prompt | llm | StrOutputParser()
    
    def add_documents(self, documents):
        self.chunks.extend(documents)
        if self.chunks:
            texts = [doc.page_content if hasattr(doc, 'page_content') else doc for doc in self.chunks]
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def search(self, query, k=3):
        if not self.chunks or self.tfidf_matrix is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                chunk = self.chunks[idx]
                if hasattr(chunk, 'page_content'):
                    results.append({
                        'content': chunk.page_content,
                        'source': chunk.metadata.get('source', 'Unknown'),
                        'similarity': similarities[idx]
                    })
                else:
                    results.append({
                        'content': chunk,
                        'source': 'Unknown',
                        'similarity': similarities[idx]
                    })
        
        return results
    
    def generate_answer(self, query):
        results = self.search(query, k=5)
        
        if not results:
            return "文档中未找到相关答案", []
        
        context = "\n\n".join([r['content'] for r in results])
        
        if self.use_ollama and self.llm_chain:
            try:
                answer = self.llm_chain.invoke({"context": context, "question": query})
                return answer, results
            except Exception as e:
                pass
        
        keywords = self.extract_keywords(query)
        answer = self.synthesize_answer(query, context, keywords, results)
        
        return answer, results
    
    def extract_keywords(self, query):
        query = query.lower()
        keywords = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', query)
        return keywords[:10]
    
    def synthesize_answer(self, query, context, keywords, results):
        sentences = context.split('。')
        relevant_sentences = []
        
        for sentence in sentences:
            for keyword in keywords:
                if keyword and len(keyword) > 1 and keyword.lower() in sentence.lower():
                    relevant_sentences.append(sentence.strip())
                    break
        
        if relevant_sentences:
            answer = '。'.join([s for s in relevant_sentences[:3] if s]) + '。'
            if len(answer) > 500:
                answer = answer[:500] + '...'
            return answer
        
        if results:
            top_result = results[0]
            content = top_result['content']
            sentences = [s.strip() for s in content.split('。') if s.strip()]
            if sentences:
                answer = '。'.join(sentences[:3]) + '。'
                return answer
        
        return "文档中未找到相关答案"

def create_simple_rag_chain(chunks, use_ollama=False, ollama_model="qwen2:7b"):
    rag = SimpleRAG(use_ollama=use_ollama, ollama_model=ollama_model)
    rag.add_documents(chunks)
    return rag

def ask_simple_question(rag, question):
    answer, sources = rag.generate_answer(question)
    return answer, sources