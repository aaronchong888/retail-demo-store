import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, HeunDiscreteScheduler, AutoencoderKL, AutoPipelineForInpainting

# Download and save the models
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
depth_estimator.save_pretrained("dpt-hybrid-midas/")

feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
feature_extractor.save_pretrained("dpt-hybrid-midas/")

controlnet1 = ControlNetModel.from_pretrained("destitech/controlnet-inpaint-dreamer-sdxl") #diffusers/controlnet-depth-sdxl-1.0
controlnet1.save_pretrained(
    "controlnet-inpaint-dreamer-sdxl/", #controlnet-depth-sdxl-1.0/
    variant="fp16",
    torch_dtype=torch.float16
)

controlnet2 = ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0")
controlnet2.save_pretrained(
    "controlnet-zoe-depth-sdxl-1.0/",
    torch_dtype=torch.float16
)

#controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0") # "BertChristiaens/controlnet-seg-room"
#controlnet.save_pretrained(
#    "controlnet-depth-sdxl-1.0/",
#    variant="fp16",
#    use_safetensors=True,
#    torch_dtype=torch.float16
#)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
vae.save_pretrained("sdxl-vae-fp16-fix/")

pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", 
        controlnet=[controlnet1, controlnet2],
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
)

pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.save_pretrained("RealVisXL_V4.0")

#pipeline2 = StableDiffusionXLInpaintPipeline.from_pretrained(
#        "OzzyGT/RealVisXL_V4.0_inpainting",
#        torch_dtype=torch.float16,
#        variant="fp16",
#        vae=vae,
#)
#pipeline2.save_pretrained("RealVisXL_V4.0_inpainting")
