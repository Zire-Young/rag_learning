#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：rag_learning 
@File    ：vectordb.py
@IDE     ：PyCharm 
@Author  ：young
@Date    ：2024/8/26 10:41 
'''
import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from gen_tools_desc import *
# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用
from zhipu_embedding import ZhipuEmbeddings

def create_vdb(file_path):
    # 读取本地/项目的环境变量。
    # find_dotenv()寻找并定位.env文件的路径
    # load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
    # 如果设置的是全局的环境变量，这行代码则没有任何作用。
    _ = load_dotenv(find_dotenv())

    # 加载工具描述数据
    file_path = gen_tools_desc(file_path)

    loader = TextLoader(file_path)
    data = loader.load()

    tools_prompt = data[0]

    print("准备建立向量数据库...",f"每一个元素的类型：{type(tools_prompt)}.",
        f"该文档的描述性数据：{tools_prompt.metadata}",
        f"查看该文档的内容:\n{tools_prompt.page_content[0:]}",
        sep="\n------\n")


    # 定义 Embeddings
    embedding = ZhipuEmbeddings()

    # 定义持久化路径
    persist_directory = '../data_base/vector_db/chroma'

    # 定义向量库
    vectordb = Chroma.from_documents(
        documents=data,
        embedding=embedding,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )

    vectordb.persist()  # 持久化到磁盘
    print(f"建立完成！向量库存储位置：{persist_directory}，库中存储的数量：{vectordb._collection.count()}")
    return persist_directory

def load_vdb(persist_directory):
    _ = load_dotenv(find_dotenv())
    # 定义 Embeddings
    embedding = ZhipuEmbeddings()

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )

    return vectordb