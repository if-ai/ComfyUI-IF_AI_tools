import argparse
import torch
import json
import os
from typing import Union, Dict, Any
import sys

from scipy.io import loadmat
import subprocess

import numpy as np
import torchaudio
import shutil

from core.utils import (
    get_pose_params,
    get_video_style_clip,
    get_wav2vec_audio_window,
    crop_src_image,
)

from configs.default import get_cfg_defaults
from generators.utils import get_netG, render_video
from core.networks.diffusion_net import DiffusionNet
from core.networks.diffusion_util import NoisePredictor, VarianceSchedule
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from modelscope.pipelines.builder import PIPELINES
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from modelscope.pipelines.base import Pipeline
from modelscope.models.base import Model, TorchModel
from modelscope.utils.logger import get_logger
from modelscope import snapshot_download


@torch.no_grad()
def get_diff_net(cfg, model_dir=None):
    diff_net = DiffusionNet(
        cfg=cfg,
        net=NoisePredictor(cfg),
        var_sched=VarianceSchedule(
            num_steps=cfg.DIFFUSION.SCHEDULE.NUM_STEPS,
            beta_1=cfg.DIFFUSION.SCHEDULE.BETA_1,
            beta_T=cfg.DIFFUSION.SCHEDULE.BETA_T,
            mode=cfg.DIFFUSION.SCHEDULE.MODE,
        ),
    )
    checkpoint = torch.load(model_dir+"/"+cfg.INFERENCE.CHECKPOINT)
    model_state_dict = checkpoint["model_state_dict"]
    diff_net_dict = {
        k[9:]: v for k, v in model_state_dict.items() if k[:9] == "diff_net."
    }
    diff_net.load_state_dict(diff_net_dict, strict=True)
    diff_net.eval()

    return diff_net

@torch.no_grad()
def get_audio_feat(wav_path, output_name, wav2vec_model):
    audio_feat_dir = os.path.dirname(audio_feat_path)

    pass

@torch.no_grad()
def inference_one_video(
    cfg,
    audio_path,
    style_clip_path,
    pose_path,
    output_path,
    diff_net,
    max_audio_len=None,
    sample_method="ddim",
    ddim_num_step=10,
):
    audio_raw = audio_data = np.load(audio_path)

    if max_audio_len is not None:
        audio_raw = audio_raw[: max_audio_len * 50]
    gen_num_frames = len(audio_raw) // 2

    audio_win_array = get_wav2vec_audio_window(
        audio_raw,
        start_idx=0,
        num_frames=gen_num_frames,
        win_size=cfg.WIN_SIZE,
    )

    audio_win = torch.tensor(audio_win_array).cuda()
    audio = audio_win.unsqueeze(0)

    # the second parameter is "" because of bad interface design...
    style_clip_raw, style_pad_mask_raw = get_video_style_clip(
        style_clip_path, "", style_max_len=256, start_idx=0
    )

    style_clip = style_clip_raw.unsqueeze(0).cuda()
    style_pad_mask = (
        style_pad_mask_raw.unsqueeze(0).cuda()
        if style_pad_mask_raw is not None
        else None
    )

    gen_exp_stack = diff_net.sample(
        audio,
        style_clip,
        style_pad_mask,
        output_dim=cfg.DATASET.FACE3D_DIM,
        use_cf_guidance=cfg.CF_GUIDANCE.INFERENCE,
        cfg_scale=cfg.CF_GUIDANCE.SCALE,
        sample_method=sample_method,
        ddim_num_step=ddim_num_step,
    )
    gen_exp = gen_exp_stack[0].cpu().numpy()

    pose_ext = pose_path[-3:]
    pose = None
    pose = get_pose_params(pose_path)
    # (L, 9)

    selected_pose = None
    if len(pose) >= len(gen_exp):
        selected_pose = pose[: len(gen_exp)]
    else:
        selected_pose = pose[-1].unsqueeze(0).repeat(len(gen_exp), 1)
        selected_pose[: len(pose)] = pose

    gen_exp_pose = np.concatenate((gen_exp, selected_pose), axis=1)
    np.save(output_path, gen_exp_pose)
    return output_path

@PIPELINES.register_module(Tasks.text_to_video_synthesis, module_name='Dreamtalk-generation-pipe')
class DreamTalkPipeline(Pipeline):
    def __init__(
            self,
            model: Union[Model, str],
            *args,
            **kwargs):
        model = DreamTalkMS(model, **kwargs) if isinstance(model, str) else model
        super().__init__(model=model, **kwargs)
    
    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs
    
    def _sanitize_parameters(self, **pipeline_parameters):
        return {},pipeline_parameters,{}
    
    # define the forward pass
    def forward(self, inputs: Dict, **forward_params) -> Dict[str, Any]:
        return self.model(inputs,**forward_params)
    
    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@MODELS.register_module(Tasks.text_to_video_synthesis, module_name='Dreamtalk-Generation')
class DreamTalkMS(TorchModel):
    def __init__(self, model_dir=None, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.logger = get_logger()
        self.style_clip_path = kwargs.get("style_clip_path", "")
        self.pose_path = kwargs.get("pose_path", "")
        os.chdir(model_dir)

        if not os.path.exists(self.style_clip_path):
            self.style_clip_path = os.path.join(model_dir, self.style_clip_path)
        
        if not os.path.exists(self.pose_path):
            self.pose_path = os.path.join(model_dir, self.pose_path) 

        self.cfg = get_cfg_defaults()
        self.cfg.freeze()

        # get wav2vec feat from audio
        wav2vec_local_dir = snapshot_download("AI-ModelScope/wav2vec2-large-xlsr-53-english",revision='master')
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_local_dir)
        self.wav2vec_model = (
        Wav2Vec2Model.from_pretrained(wav2vec_local_dir)
        .eval()
        .cuda()
        )
        self.diff_net = get_diff_net(self.cfg, model_dir).cuda()
        # get renderer
        self.renderer = get_netG(os.path.join(model_dir, "../checkpoints/renderer.pt"))
        self.model_dir = model_dir
    
    def forward(self, input: Dict, *args, **kwargs) -> Dict[str, Any]:
        output_name = input.get("output_name", "")
        wav_path = input.get("wav_path", "")
        img_crop = input.get("img_crop", True)
        image_path = input.get("image_path", "")
        max_gen_len = input.get("max_gen_len",1000)
        sys.path.append(self.model_dir)

        tmp_dir = f"tmp/{output_name}"
        os.makedirs(tmp_dir, exist_ok=True)

        # get audio in 16000Hz
        wav_16k_path = os.path.join(tmp_dir, f"{output_name}_16K.wav")
        if not os.path.exists(wav_path):
            wav_path = os.path.join(self.model_dir, wav_path)
        command = f"ffmpeg -y -i {wav_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {wav_16k_path}"
        subprocess.run(command.split())

        speech_array, sampling_rate = torchaudio.load(wav_16k_path)
        audio_data = speech_array.squeeze().numpy()

        inputs = self.wav2vec_processor(
        audio_data, sampling_rate=16_000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            audio_embedding = self.wav2vec_model(inputs.input_values.cuda(), return_dict=False)[0]
        
        audio_feat_path = os.path.join(tmp_dir, f"{output_name}_wav2vec.npy")
        np.save(audio_feat_path, audio_embedding[0].cpu().numpy())

        # get src image
        src_img_path = os.path.join(tmp_dir, "src_img.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.model_dir, image_path)
        if img_crop:
            crop_src_image(image_path, src_img_path, 0.4)
        else:
            shutil.copy(image_path, src_img_path)

        with torch.no_grad():
            face_motion_path = os.path.join(tmp_dir, f"{output_name}_facemotion.npy")
            inference_one_video(
                self.cfg,
                audio_feat_path,
                self.style_clip_path,
                self.pose_path,
                face_motion_path,
                self.diff_net,
                max_audio_len=max_gen_len,
                )
            # render video
            output_video_path = f"output_video/{output_name}.mp4"
            render_video(
                self.renderer,
                src_img_path,
                face_motion_path,
                wav_16k_path,
                output_video_path,
                fps=25,
                no_move=False,
                )