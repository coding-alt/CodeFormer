import os
import cv2
import torch
import gradio as gr
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY

# 预训练模型的URL
pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_colorization.pth'

# 设置模型和设备
device = get_device()
net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128']).to(device)
ckpt_path = load_file_from_url(url=pretrain_model_url, model_dir='weights/CodeFormer', progress=True, file_name=None)
checkpoint = torch.load(ckpt_path)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

def colorize_image(image):
    # 图片处理和上色
    input_face = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    input_face = img2tensor(input_face / 255., bgr2rgb=False, float32=True)
    normalize(input_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    input_face = input_face.unsqueeze(0).to(device)

    with torch.no_grad():
        output_face = net(input_face, w=0, adain=True)[0]
        output_face = tensor2img(output_face, rgb2bgr=False, min_max=(-1, 1))

    output_face = output_face.astype('uint8')
    output_face = cv2.cvtColor(output_face, cv2.COLOR_BGR2RGB)

    return output_face

# Gradio界面
iface = gr.Interface(fn=colorize_image, 
    inputs=gr.Image(type="numpy", label="原始图片（512*512）"),
    outputs=gr.Image(type="pil", label="上色后的图片"),
    title="脸部上色",
    description="上传图片进行自动上色处理",
    theme='Kasien/ali_theme_custom',
    css="footer {visibility: hidden}",
    allow_flagging="never"
)

iface.launch(server_name='0.0.0.0')
