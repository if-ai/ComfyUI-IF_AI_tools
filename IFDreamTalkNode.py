import os
import sys
import datetime
import subprocess
import tempfile
import shutil
import comfy.utils
import torch
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
import torchaudio
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

# Get the absolute path to the dreamtalk directory
dreamtalk_dir = os.path.join(os.path.dirname(__file__), "dreamtalk")
sys.path.insert(0, dreamtalk_dir)

from .dreamtalk.core.utils import (
    get_pose_params,
    get_video_style_clip,
    get_wav2vec_audio_window,
    crop_src_image,
)
from .dreamtalk.configs.default import get_cfg_defaults
from .dreamtalk.generators.utils import get_netG, render_video
from .dreamtalk.core.networks.diffusion_net import DiffusionNet  # Correct import
from .dreamtalk.core.networks.diffusion_util import NoisePredictor, VarianceSchedule


def convert_to_mp4_with_aac(input_path, output_path):
    """Converts the video to a compatible MP4 format with AAC audio codec."""
    video = VideoFileClip(input_path)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path


# Ensure values are within 0-255 range
def tensor_to_bytes(tensor):
    """Converts a tensor to bytes."""
    return (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)  # Ensure values are within 0-255 range


class IFDreamTalk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_path": ("STRING", {"default": "input/", "vhs_path_extensions": ['png', 'jpg', 'jpeg']}),
                "crop": ("BOOLEAN", {"default": True}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "name": ("STRING", {"default": "IF_AI_DreamTalk"}),
                "audio_input": ("STRING", {"default": "input/", "vhs_path_extensions": ['wav', 'mp3', 'ogg', 'm4a', 'flac']}),
                "emotional_style": (
                    [
                        "M030_front_angry_level3_001.mat",
                        "M030_front_contempt_level3_001.mat",
                        "M030_front_disgusted_level3_001.mat",
                        "M030_front_fear_level3_001.mat",
                        "M030_front_happy_level3_001.mat",
                        "M030_front_neutral_level1_001.mat",
                        "M030_front_sad_level3_001.mat",
                        "M030_front_surprised_level3_001.mat",
                        "W009_front_angry_level3_001.mat",
                        "W009_front_contempt_level3_001.mat",
                        "W009_front_disgusted_level3_001.mat",
                        "W009_front_fear_level3_001.mat",
                        "W009_front_happy_level3_001.mat",
                        "W009_front_neutral_level1_001.mat",
                        "W009_front_sad_level3_001.mat",
                        "W009_front_surprised_level3_001.mat",
                        "W011_front_angry_level3_001.mat",
                        "W011_front_contempt_level3_001.mat",
                        "W011_front_disgusted_level3_001.mat",
                        "W011_front_fear_level3_001.mat",
                        "W011_front_happy_level3_001.mat",
                        "W011_front_neutral_level1_001.mat",
                        "W011_front_sad_level3_001.mat",
                        "W011_front_surprised_level3_001.mat",
                    ],
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",) 
    RETURN_NAMES = ("image", "video_path",) # Output the resulting video location as a string
    FUNCTION = "run"
    CATEGORY = "ImpactFrames💥🎞️"
    OUTPUT_NODE = True

    def convert_to_16k(self, audio_input, output_name, tmp_dir):
        # Get the sample rate of the audio input
        command = f"ffprobe -v error -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 {audio_input}"
        sample_rate = subprocess.check_output(command.split()).decode().strip()

        if sample_rate == "16000":
            # Audio input already has a sample rate of 16kHz
            wav_16k_path = audio_input
        else:
            # Convert the audio to 16kHz
            wav_16k_path = os.path.join(tmp_dir, f"{output_name}_16K.wav")
            command = f"ffmpeg -y -i {audio_input} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {wav_16k_path}"
            subprocess.run(command.split())

        return wav_16k_path
    
    def run(self, image_path, crop, cfg_scale, name, audio_input, emotional_style):
        max_gen_len = 1000
        comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        checkpoints_dir = os.path.join(comfy_dir, "models", "dreamtalk", "checkpoints")
        dreamtalk_dir = os.path.join(os.path.dirname(__file__), "dreamtalk")

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = f"{name}_{timestamp}"

        output_dir = os.path.join(comfy_dir, "output", output_name)
        os.makedirs(output_dir, exist_ok=True)

        tmp_dir = os.path.join(comfy_dir, "temp", output_name)
        os.makedirs(tmp_dir, exist_ok=True)

        style_clip_path = os.path.join(
            dreamtalk_dir, "data", "style_clip", "3DMM", emotional_style
        )
        pose_path = os.path.join(
            dreamtalk_dir, "data", "pose", "RichardShelby_front_neutral_level1_001.mat"
        )
        
        # Integrate inference logic
        cfg = get_cfg_defaults()
        cfg.CF_GUIDANCE.SCALE = cfg_scale
        cfg.freeze()

        # get audio in 16000Hz
        wav_16k_path = self.convert_to_16k(audio_input, output_name, tmp_dir)

        # get wav2vec feat from audio
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        )
        wav2vec_model = (
            Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
            .eval()
            .cuda()
        )

        speech_array, sampling_rate = torchaudio.load(wav_16k_path)
        audio_data = speech_array.squeeze().numpy()
        inputs = wav2vec_processor(
            audio_data, sampling_rate=16_000, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            audio_embedding = wav2vec_model(inputs.input_values.cuda(), return_dict=False)[
                0
            ]

        audio_feat_path = os.path.join(tmp_dir, f"{output_name}_wav2vec.npy")
        np.save(audio_feat_path, audio_embedding[0].cpu().numpy())

        # get src image
        src_img_path = os.path.join(tmp_dir, "src_img.png")
        if crop:
            crop_src_image(image_path, src_img_path, 0.4)
        else:
            shutil.copy(image_path, src_img_path)

        with torch.no_grad():
            # get diff model and load checkpoint
            diff_net = self.get_diff_net(cfg).cuda()
            # generate face motion
            face_motion_path = os.path.join(tmp_dir, f"{output_name}_facemotion.npy")
            self.inference_one_video(
                cfg,
                audio_feat_path,
                style_clip_path,
                pose_path,
                face_motion_path,
                diff_net,
                max_audio_len=max_gen_len,
            )
            # get renderer
            renderer = get_netG(os.path.join(checkpoints_dir, "renderer.pt"))
            # render video
            output_video_path = os.path.join(tmp_dir, f"{output_name}.mp4")  # Save to temp dir
            render_video(
                renderer,
                src_img_path,
                face_motion_path,
                wav_16k_path,
                output_video_path,
                fps=25,
                no_move=False,
            )
    
            output_file = os.path.join(output_dir, f"{output_name}_converted.mp4")
            result = convert_to_mp4_with_aac(output_video_path, output_file)

            # Load the generated video as image frames
            '''video = VideoFileClip(output_file)
            first_frame = video.get_frame(0)
            first_frame_image = Image.fromarray(first_frame)'''

            video = VideoFileClip(output_file)
            frames = []
            for frame in video.iter_frames():
                # Convert the frame to RGB format
                frame = frame[:, :, :3]  # Remove the alpha channel if present
                frame = frame.astype(np.uint8)  # Convert to uint8
                frame_image = Image.fromarray(frame)  # Create PIL image
                frames.append(frame_image)

            return frames, result


    @torch.no_grad()
    def get_diff_net(self, cfg):
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
        comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        checkpoints_dir = os.path.join(comfy_dir, "models", "dreamtalk", "checkpoints")
        checkpoint = torch.load(os.path.join(checkpoints_dir, "denoising_network.pth"))
        model_state_dict = checkpoint["model_state_dict"]
        diff_net_dict = {k[9:]: v for k, v in model_state_dict.items() if k[:9] == "diff_net."}
        diff_net.load_state_dict(diff_net_dict, strict=True)
        diff_net.eval()

        return diff_net

    @torch.no_grad()
    def inference_one_video(
        self,
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

NODE_CLASS_MAPPINGS = {"IF_DreamTalk": IFDreamTalk}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_Dreamtalk": "IF DreamTalk🧏🏻"}