import torch
import requests
from PIL import Image
from controlnet_aux import MLSDdetector
import os
import cv2
import numpy as np

from diffusers import StableDiffusionDepth2ImgPipeline, DiffusionPipeline, ControlNetModel,StableDiffusionControlNetPipeline, StableDiffusionPipeline,UniPCMultistepScheduler


class ControlNetMLSD:
    def __init__(self, device) -> None:
        self.device = device
        self.mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        self.controlnet = ControlNetModel.from_pretrained(
            "./stable-diffusion-v1-5-controlnet-mlsd", torch_dtype=torch.float16
        )
            

    def generate_img(self,img: Image,room,style) -> Image:

        if os.path.exists("./stable-diffusion-v1-5"):
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "./stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
            ).to(self.device)
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
            ).to(self.device)
            pipe.save_pretrained("./stable-diffusion-v1-5")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        init_image = self.mlsd(img)



        prompt = f"high resolution photography of {room} in {style} design style, sun light, contrast, hyperdetailed, ultradetail, cinematic 8k, architectural rendering , unreal engine 5, rtx, volumetric light"
        n_prompt = "ugly, low definition, poorly designed, amateur, bad proportions, bad lighting"


        # pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()
        image = pipe(prompt=prompt,negative_prompt=n_prompt, num_inference_steps=20,image=init_image).images[0]

        return image
