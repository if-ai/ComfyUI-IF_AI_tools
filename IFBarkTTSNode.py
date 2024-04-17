import os
import torch
import tempfile
import datetime
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile as wav
import torchaudio
import librosa
import textwrap
import nltk
from nltk.tokenize import sent_tokenize
import re
import numpy as np

class IFBarkTTS:


    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": node.sample_text}),
                "file_name": ("STRING", {"default": node.file_name}),
                "low_vram": ("BOOLEAN", {"default": True}),
                "speaker":(node.speakers, {"default": "en_speaker_6"}),   
                "emotion": (["none", "laughter", "laughs", "sighs", "music", "gasps", "clears throat"], {"default": "none"}),
            },
            "optional": {
                "cps": ("FLOAT", {"default": node.cps, "min": 10.0, "max": 20.0, "step": 0.25}),
                "overlap": ("FLOAT", {"default": node.overlap, "min": 0.0, "max": 200.0, "step": 0.25}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "generate_audio"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    def __init__(self):
        self.file_name = "IF_whisper_speech"
        self.cps = 14.0
        self.overlap = 100.0
        self.speaker_v2_dir = os.path.join(os.path.dirname(__file__), "bark", "speakers", "v2")
        self.speakers = self.load_speakers(self.speaker_v2_dir)     
        self.sample_text = textwrap.dedent("""\
            Electromagnetism is a fundamental force of nature that encompasses the interaction between
            electrically charged particles. It is described by Maxwell's equations, which unify electricity, magnetism,
            and light into a single theory. In essence, electric charges produce electric fields that exert forces on
            other charges, while moving charges (currents) generate magnetic fields. These magnetic fields, in turn,
            can affect the motion of charges and currents. The interaction between electric and magnetic fields propagates
            through space as electromagnetic waves, which include visible light, radio waves, and X-rays. Electromagnetic
            forces are responsible for practically all the phenomena encountered in daily life, excluding gravity.
            """)  # Load speakers from v2

    def load_speakers(self, dir_path):
        speakers = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".npz"):
                speakers.append(os.path.splitext(filename)[0]) 
        return speakers
    
    def split_and_prepare_text(self, text, cps):
        chunks = []
        sentences = sent_tokenize(text)
        chunk = ""
        for sentence in sentences:
            # replace fancy punctuation that was unseen during training
            sentence = re.sub('[()]', ",", sentence).strip()
            sentence = re.sub(",+", ",", sentence)
            sentence = re.sub('"+', "", sentence)
            sentence = re.sub("/", "", sentence)
            # merge until the result is < 20s
            if len(chunk) + len(sentence) < 20*cps:
                chunk += " " + sentence
            else:
                chunks.append(chunk)
                chunk = sentence
        if chunk: chunks.append(chunk)
        return chunks

    def generate_audio(self, text, file_name, low_vram, speaker, emotion, cps, overlap):
        if speaker not in self.speakers:
            raise ValueError(f"Speaker {speaker} not found")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create output directory
            comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            output_name = f"{file_name}_{timestamp}"
            output_dir = os.path.join(comfy_dir, "output", output_name)
            os.makedirs(output_dir, exist_ok=True)
            wav_16k_path = os.path.join(output_dir, f"{output_name}.wav")
            tmp_dir = os.path.join(comfy_dir, "temp", output_name)
            os.makedirs(tmp_dir, exist_ok=True)

            wav_temp_path = os.path.join(tmp_dir, "audio.wav")

            if low_vram == True:
                device = "cpu"
                torch.cuda.empty_cache()
                os.environ["SUNO_OFFLOAD_CPU"] = "True"
                os.environ["SUNO_USE_SMALL_MODELS"] = "True" 
                processor = AutoProcessor.from_pretrained("suno/bark")
                model = BarkModel.from_pretrained("suno/bark").to(device)
                model.enable_cpu_offload()
            else:
                device = "cuda"
                # Load Bark model with optimizations
                processor = AutoProcessor.from_pretrained("suno/bark")
                model = BarkModel.from_pretrained(
                    "suno/bark",
                    torch_dtype=torch.float16,
                ).to(device)
            
            # Concatenate emotion at the beginning of the text
            if emotion != "none":
                text = f"[{emotion}] {text}"

            # Split the text into chunks
            chunks = self.split_and_prepare_text(text, cps)

            # Generate audio for each chunk
            pieces = []
            silence = np.zeros(int(0.25 * model.generation_config.sample_rate))  # quarter second of silence
            for chunk in chunks:
                inputs = processor(chunk, add_special_tokens=True, return_tensors="pt")
                inputs["attention_mask"] = inputs["attention_mask"].to(device)
                audio_array = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                audio_array = audio_array.cpu().numpy().squeeze()
                pieces += [audio_array, silence.copy()]

            # Concatenate the audio pieces
            audio = np.concatenate(pieces)

            # Save the audio using scipy.io.wavfile
            sample_rate = model.generation_config.sample_rate
            wav.write(wav_temp_path, rate=sample_rate, data=audio)

            # Load the audio using librosa
            audio, sr = librosa.load(wav_temp_path, sr=24000)

            # Resample the audio to 16kHz using librosa
            resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Save the resampled audio using scipy.io.wavfile
            wav.write(wav_16k_path, rate=16000, data=resampled_audio)

            return wav_16k_path


NODE_CLASS_MAPPINGS = {"IF_BarkTTS": IFBarkTTS}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_BarkTTS": "IF Bark TTS 🎤"}