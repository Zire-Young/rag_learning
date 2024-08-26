#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：rag_learning 
@File    ：gen_tools_desc.py
@IDE     ：PyCharm 
@Author  ：young
@Date    ：2024/8/26 11:28 
'''
import json
import os

# 读取外部 JSON 文件
def load_tools_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tools_info = json.load(file)
    return tools_info

# 生成工具描述prompt
def gen_tools_prompt(tools_info):
    tools_desc = []
    for idx, t in enumerate(tools_info):
        args_desc = []
        for info in t["args"]:
            args_desc.append({
                "name": info["name"],
                "type": info["type"],
                "description": info["description"]
            })
        args_desc = json.dumps(args_desc, ensure_ascii=False)
        tool_desc = f"{idx + 1}.{t['name']}:{t['description']}, args:{args_desc}"
        tools_desc.append(tool_desc)
    tools_prompt = "\n".join(tools_desc)
    return tools_prompt

'''
生成的工具描述prompt示例：
1.工具1:这是工具1的描述，args:[{"name": "arg1", "type": "int", "description": "这是参数1的描述"}, {"name": "arg2", "type": "str", "description": "这是参数2的描述"}]
2.工具2:这是工具2的描述，args:[{"name": "arg1", "type": "float", "description": "这是参数1的描述"}, {"name": "arg2", "type": "list", "description": "这是参数2的描述"}]
'''


# 工具描述生成器
def gen_tools_desc(file_path):
    # 读取工具信息
    tools_info = load_tools_info(file_path)
    tools_prompt = gen_tools_prompt(tools_info)

    # 动态生成输出文件路径
    base_name, ext = os.path.splitext(file_path)
    output_file_path = f"{base_name}_prompt.txt"

    with open(output_file_path, 'w') as file:
        file.write(tools_prompt)

    # 返回新建的文件路径
    return output_file_path
