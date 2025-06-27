import os
from typing import Any

class VLMRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name):
        if name not in cls._models:
            raise ValueError(f"Model {name} not found in registry")
        return cls._models[name]

@VLMRegistry.register("MedVLM-R1")
class MedVLM_R1:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from mllm_zoo.MedVLM_R1.MedVLM_R1_hf import MedVLM_R1
        return MedVLM_R1(model_path, args)

@VLMRegistry.register("Huatuo")
class HuatuoGPT:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from mllm_zoo.HuatuoGPT.HuatuoGPT_vllm import HuatuoGPT
        else:
            from mllm_zoo.HuatuoGPT.HuatuoGPT_hf import HuatuoGPT
        return HuatuoGPT(model_path, args)

@VLMRegistry.register("MedGemma")
class MedGemma:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from mllm_zoo.MedGemma.MedGemma import MedGemma
        return MedGemma(model_path, args)

@VLMRegistry.register("BiMediX2")
class BiMediX2:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from mllm_zoo.BiMediX2.BiMediX2_hf import BiMediX2
        return BiMediX2(model_path, args)

@VLMRegistry.register("Qwen2.5-VL") 
class Qwen2_5_VL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from mllm_zoo.Qwen2_5_VL.Qwen2_5_VL_vllm import Qwen2_5_VL
        else:
            from mllm_zoo.Qwen2_5_VL.Qwen2_5_VL_hf import Qwen2_5_VL
        return Qwen2_5_VL(model_path, args)

@VLMRegistry.register("InternVL")
class InternVL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from mllm_zoo.InternVL.InternVL_vllm import InternVL
        else:
            from mllm_zoo.InternVL.InternVL_hf import InternVL
        return InternVL(model_path, args)

def init_vlm(args):
    try:
        model_class = VLMRegistry.get_model(args.model_name)
        return model_class(args.model_path, args)
    except ValueError as e:
        raise ValueError(f"{args.model_name} not supported") from e
