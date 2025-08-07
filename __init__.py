from mmgp import offload
from mmgp.offload import profile_type,fast_load_transformers_model

import gc
import os.path as osp
import torch
import folder_paths
import comfy.model_management as mm

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()
torch_dtype = torch.bfloat16
now_dir = osp.dirname(__file__)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")

from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2_5_VLConfig
from diffusers import QwenImageTransformer2DModel,QwenImagePipeline

from huggingface_hub import snapshot_download

vram_optimization_opts = [
    'No_Optimization',
    'HighRAM_HighVRAM',
    'HighRAM_LowVRAM',
    'LowRAM_HighVRAM',
    'LowRAM_LowVRAM',
    'VerylowRAM_LowVRAM'
]


class LoadQwenImagePipe:

    def __init__(self):
        pipe_path = folder_paths.models_dir
        if not osp.exists(osp.join(pipe_path,"vae/diffusion_pytorch_model.safetensors")):
            snapshot_download(repo_id="Qwen/Qwen-Image",local_dir=pipe_path,
                              ignore_patterns=["transformer*","text_encoder*"])
        self.pipe_path = pipe_path

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "transformer_mmgp":(folder_paths.get_filename_list("diffusion_models"),),
                "text_encoder_mmgp":(folder_paths.get_filename_list("text_encoders"),),
                "vram_optimization":(vram_optimization_opts,{
                    "default": 'HighRAM_HighVRAM',
                })
            }
        }
    
    RETURN_TYPES = ("QwenImagePipe",)
    RETURN_NAMES = ("pipe",)

    FUNCTION = "load_pipe"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/QwenImage"

    def load_pipe(self,transformer_mmgp,text_encoder_mmgp,vram_optimization):
        #
        '''
        transformer = QwenImageTransformer2DModel.from_pretrained("/root/git_repos/Qwen-Image/models/transformer",torch_dtype=torch_dtype)
        offload.save_model(transformer,file_path="/root/ComfyUI/models/diffusion_models/qwen_image_transformer_mmgp.safetensors",
                           do_quantize=True)
        
        '''
        transformer = fast_load_transformers_model(folder_paths.get_full_path_or_raise("diffusion_models",transformer_mmgp),
                                                   do_quantize=True,modelClass=QwenImageTransformer2DModel,
                                                   forcedConfigPath=osp.join(now_dir,"configs/transformer_config.json"))
        
        '''
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained("/root/git_repos/Qwen-Image/models/text_encoder",torch_dtype=torch_dtype)
        offload.save_model(text_encoder,file_path="/root/ComfyUI/models/text_encoders/qwen_image_text_encoder_mmgp.safetensors",
                           do_quantize=True,)
        '''
        text_encoder = fast_load_transformers_model(folder_paths.get_full_path_or_raise("text_encoders",text_encoder_mmgp),
                                                    do_quantize=True,modelClass=Qwen2_5_VLForConditionalGeneration,
                                                    forcedConfigPath=osp.join(now_dir,"configs/text_encoder_config.json"))
        

        
        pipe = QwenImagePipeline.from_pretrained(self.pipe_path,torch_dtype=torch_dtype,
                                                 transformer=transformer,text_encoder=text_encoder)

        if vram_optimization == 'No_Optimization':
            pipe.to(device)
        else:
            [
            'No_Optimization',
            'HighRAM_HighVRAM',
            'HighRAM_LowVRAM',
            'LowRAM_HighVRAM',
            'LowRAM_LowVRAM',
            'VerylowRAM_LowVRAM'
        ]
            if vram_optimization == 'HighRAM_HighVRAM':
                optimization_type = profile_type.HighRAM_HighVRAM
            elif vram_optimization == 'HighRAM_HighVRAM':
                optimization_type = profile_type.HighRAM_HighVRAM
            elif vram_optimization == 'HighRAM_LowVRAM':
                optimization_type = profile_type.HighRAM_LowVRAM
            elif vram_optimization == 'LowRAM_HighVRAM':
                optimization_type = profile_type.LowRAM_HighVRAM
            elif vram_optimization == 'LowRAM_LowVRAM':
                optimization_type = profile_type.LowRAM_LowVRAM
            elif vram_optimization == 'VerylowRAM_LowVRAM':
                optimization_type = profile_type.VerylowRAM_LowVRAM
            offload.profile(pipe, optimization_type)
        
        return (pipe,)
    
class QwenImageRatio2Size:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "aspect_ratio":(["1:1","16:9","9:16","4:3","3:4"],)
            }
        }
    
    RETURN_TYPES = ("INT","INT",)
    RETURN_NAMES = ("width","height",)

    FUNCTION = "get_image_size"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/QwenImage"

    # (1664, 928), (1472, 1140), (1328, 1328)
    def get_image_size(self,aspect_ratio):
        if aspect_ratio == "1:1":
            return (1328, 1328,)
        elif aspect_ratio == "16:9":
            return (1664, 928,)
        elif aspect_ratio == "9:16":
            return (928, 1664,)
        elif aspect_ratio == "4:3":
            return (1472, 1140,)
        elif aspect_ratio == "3:4":
            return (1140, 1472,)
        else:
            return (1328, 1328,)

class QwenImageSampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "pipe":("QwenImagePipe",),
                "prompt":("STRING",),
                "negative_prompt":("STRING",),
                "width":("INT",{
                    "default":982
                }),
                "height":("INT",{
                    "default":1664
                }),
                "num_inference_steps":("INT",{
                    "default":50
                }),
                "guidance_scale":("FLOAT",{
                    "default":5,
                }),
                "seed":("INT",{
                    "default":42
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "sample"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH/QwenImage"

    def sample(self,pipe,prompt,negative_prompt,
               width,height,num_inference_steps,
               guidance_scale,seed):
        generator = torch.Generator(device=device).manual_seed(seed)
            
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                generator=generator
            ).images[0]

        return (pil2comfy(image),)
    
    
import numpy as np
from PIL import Image
def comfy2pil(image):
    i = 255. * image.cpu().numpy()[0]
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img
    
def pil2comfy(pil):
    # image = pil.convert("RGB")
    image = np.array(pil).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]
    return image


NODE_CLASS_MAPPINGS = {
    "LoadQwenImagePipe": LoadQwenImagePipe,
    "QwenImageSampler":QwenImageSampler,
    "QwenImageRatio2Size":QwenImageRatio2Size,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadQwenImagePipe": "LoadQwenImagePipe@关注超级面爸微信公众号",
    "QwenImageSampler":"QwenImageSampler@关注超级面爸微信公众号",
    "QwenImageRatio2Size":"QwenImageRatio2Size@关注超级面爸微信公众号"
}
