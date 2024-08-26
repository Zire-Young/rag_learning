#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：rag_learning 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：young
@Date    ：2024/8/26 14:21 
'''
from vectordb import *
from zhipu_llm import ZhipuLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def main():
    '''
    todo:
        1.建立向量数据库
        2.接入LLM
        3.构建检索问答链
    '''
    # 1.载入数据，建立向量数据库
    # file_path = "../data_base/knowledge_db/tools_info.json"
    # vdb_path = create_vdb(file_path)

    # 2.接入LLM
    llm = ZhipuLLM()

    # 3.构建检索问答链
    # vectordb = load_vdb(vdb_path)
    vectordb = load_vdb("../data_base/vector_db/chroma")

    template = """仅使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """
    # 构建模版
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=template)
    # 基于模板构建检索问答链
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           retriever=vectordb.as_retriever(search_type="similarity",search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    query = input("请输入问题：")
    result = qa_chain.invoke({"query":query})
    print(result)

if __name__ == '__main__':
    main()