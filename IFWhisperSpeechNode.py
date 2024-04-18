import os
import re
import tempfile
import datetime
import torch
import torch.nn.functional as F
import librosa
import textwrap
import json
from whisperspeech.pipeline import Pipeline
import torchaudio
import nltk
from nltk.tokenize import sent_tokenize
import re
import scipy.io.wavfile as wav

import nltk
nltk.download('punkt')

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:0'

class IFWhisperSpeech:
    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": node.sample_text}),
                "file_name": ("STRING", {"default": node.file_name}),
                "speaker": (node.audio_files, {}),
                "torch_compile": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cps": ("FLOAT", {"default": node.cps, "min": 10.0, "max": 20.0, "step": 0.25}),
                "overlap": ("FLOAT", {"default": node.overlap, "min": 0.0, "max": 200.0, "step": 0.25}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audios", "wav_16k_path")
    FUNCTION = "generate_audio"
    CATEGORY = "ImpactFrames💥🎞️"
    OUTPUT_NODE = True

    def IS_CHANGED(cls, file_name, speaker, cps, overlap):
        node = cls()
        if file_name != node.file_name or speaker != node.speaker or cps != node.cps or overlap != node.overlap:
            node.file_name = file_name
            node.speaker = speaker
            node.cps = cps
            node.overlap = overlap
            return True
        return False

    def __init__(self):
        
        self.file_name = "IF_whisper_speech"
        self.cps = 14.0
        self.overlap = 100.0
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.audio_dir = os.path.join(os.path.dirname(__file__), "whisperspeech", "audio")
        self.audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith(".ogg")]
        self.audio_files.insert(0, "None")
        self.sample_text = textwrap.dedent("""\
            Electromagnetism is a fundamental force of nature that encompasses the interaction between
            electrically charged particles. It is described by Maxwell's equations, which unify electricity, magnetism,
            and light into a single theory. In essence, electric charges produce electric fields that exert forces on
            other charges, while moving charges (currents) generate magnetic fields. These magnetic fields, in turn,
            can affect the motion of charges and currents. The interaction between electric and magnetic fields propagates
            through space as electromagnetic waves, which include visible light, radio waves, and X-rays. Electromagnetic
            forces are responsible for practically all the phenomena encountered in daily life, excluding gravity.
            """)

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

    def generate_audio(self, text, file_name, speaker, torch_compile, cps, overlap):
        pipe = Pipeline(torch_compile=torch_compile)
        global atoks, stoks
        chunks = self.split_and_prepare_text(text, cps)
        if speaker != "None":
            speaker_file_path = os.path.join(os.path.dirname(__file__), "whisperspeech", "audio", speaker)
            speaker = pipe.extract_spk_emb(speaker_file_path)
        else:
            speaker = pipe.default_speaker
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = f"{file_name}_{timestamp}"
        output_dir = os.path.join(self.comfy_dir, "output", output_name)
        os.makedirs(output_dir, exist_ok=True)
        wav_16k_path = os.path.join(output_dir, f"{output_name}.wav")
        tmp_dir = os.path.join(self.comfy_dir, "temp", output_name)
        os.makedirs(tmp_dir, exist_ok=True)
        wav_temp_path = os.path.join(tmp_dir, f"{output_name}_temp.wav")
        r = []
        old_stoks = None
        old_atoks = None
        for chunk in chunks:
            print(chunk)
            stoks = pipe.t2s.generate(chunk, cps=cps, show_progress_bar=False)[0]
            stoks = stoks[stoks != 512]
            if old_stoks is not None:
                assert(len(stoks) < 750-overlap)
                stoks = torch.cat([old_stoks[-overlap:], stoks])
                atoks_prompt = old_atoks[:,:,-overlap*3:]
            else:
                atoks_prompt = None
            atoks = pipe.s2a.generate(stoks, atoks_prompt=atoks_prompt, speakers=speaker.unsqueeze(0), show_progress_bar=False)
            if atoks_prompt is not None: atoks = atoks[:,:,overlap*3+1:]
            r.append(atoks)
            old_stoks = stoks
            old_atoks = atoks
            pipe.vocoder.decode_to_notebook(atoks)
        audios = []
        for i,atoks in enumerate(r):
            if i != 0: audios.append(torch.zeros((1, int(24000*0.5)), dtype=atoks.dtype, device=atoks.device))
            audios.append(pipe.vocoder.decode(atoks))
        if output_dir:
            torchaudio.save(wav_temp_path, torch.cat(audios, -1).cpu(), 24000)

        # Load the audio using librosa
        audio, sr = librosa.load(wav_temp_path, sr=24000)

        # Resample the audio to 16kHz using librosa
        resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Save the resampled audio using scipy.io.wavfile
        wav.write(wav_16k_path, rate=16000, data=resampled_audio)

        return audios, wav_16k_path
    
NODE_CLASS_MAPPINGS = {"IF_WhisperSpeech": IFWhisperSpeech}
NODE_DISPLAY_NAME_MAPPINGS = {"IF_WhisperSpeech": "IF Whisper Speech🌬️"}