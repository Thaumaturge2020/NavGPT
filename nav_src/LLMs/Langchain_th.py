import json
import time
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import Field, model_validator
import os

import zhipuai
zhipuai.api_key = os.environ["zhipuai_api_key"]


import re

def getText(role, content, text = []):
    # role 是指定角色，content 是 prompt 内容
    jsoncon = {}
    content = re.sub("\n","",content)
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

class TsingHua_LLM(LLM):
    model_name : str = "zhipuai-chatglm"
    # 默认选用 ERNIE-Bot-turbo 模型，即目前一般所说的百度文心大模型

    model: str = None
    # 访问时延上限
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    # 温度系数
    temperature: float = 0.7
    # API_Key
    api_key: str = None
    # Secret_Key
    secret_key : str = None
    # access_token
    access_token: str = None
    # 必备的可选参数
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "custom_tsing_hua"

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        
        zhipuai.api_key = self.api_key
        
        question = getText("user", prompt)

        print(question)

        response = zhipuai.model_api.invoke(
            model=self.model,
            prompt=question
        )

        print(response)
        print(response['data']['choices'][0]['content'])

        if response['code'] == 200:
            # 返回的是一个 Json 字符串
            return response['data']['choices'][0]['content']
        else:
            return "failed"
        
    @property
    def _default_params(self) -> Dict[str, Any]:
        """获取调用Ennie API的默认参数。"""
        normal_params = {
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            }
        return {**normal_params}


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}