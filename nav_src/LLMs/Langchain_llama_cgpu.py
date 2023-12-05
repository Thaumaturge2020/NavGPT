import json
import time
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
from pydantic import Field, model_validator
import os

class llama_CGPU(LLM):
    # 原生接口地址
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
    # 默认选用 ERNIE-Bot-turbo 模型，即目前一般所说的百度文心大模型
    model_name: str = Field(default="ERNIE-Bot-turbo", alias="model")
    # 访问时延上限
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    # 温度系数
    temperature: float = 0.1
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
        return "custom_llama_cgpu"

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        url = "http://localhost:8080/completion"
        payload = json.dumps({
            "prompt": "{}".format(prompt)# 输入的 prompt
        })
        headers = {
            'Content-Type': 'application/json'
        }
        # 发起请求
        response = requests.request("POST", url, headers=headers, data=payload, timeout=self.request_timeout)
            # 返回的是一个 Json 字符串
        print(response)
        str = json.loads(response.content)
        print(str)
        str = str['content']
        print(str)
        print("?????")
        return str
        
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