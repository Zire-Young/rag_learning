#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：rag_learning 
@File    ：zhipu_llm.py
@IDE     ：PyCharm 
@Author  ：young
@Date    ：2024/8/20 16:37 
'''

from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from zhipuai import ZhipuAI

import os


# 继承自 langchain.llms.base.LLM
class ZhipuLLM(LLM):
    # 默认选用 glm-4
    model: str = "glm-4"
    # 温度系数
    temperature: float = 0.1
    # API_Key
    api_key: str = None

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        client = ZhipuAI(
            api_key=self.api_key
        )

        def gen_glm_params(prompt):
            '''
            构造 GLM 模型请求参数 messages

            请求参数：
                prompt: 对应的用户提示词
            '''
            messages = [
                {"role": "system", "content": "你是一个强大的人工智能助手，你需要考虑完成用户目标需要使用工具列表中的哪些工具，并以JSON格式返回每个工具所需的参数信息。"},
                {"role": "user", "content": prompt}
            ]
            return messages

        messages = gen_glm_params(prompt)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "generate answer error"

    # 首先定义一个返回默认参数的方法
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用API的默认参数。"""
        normal_params = {
            "temperature": self.temperature,
        }
        # print(type(self.model_kwargs))
        return {**normal_params}

    @property
    def _llm_type(self) -> str:
        return "Zhipu"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model": self.model}, **self._default_params}