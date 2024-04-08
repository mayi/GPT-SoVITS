from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from typing import List
from pydantic import BaseModel
import uuid
import sys
import os, re, logging
now_dir = os.getcwd()
sys.path.insert(0, now_dir)

import LangSegment
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
from transformers import AutoModelForMaskedLM, AutoTokenizer
logger = logging.getLogger(__name__)
import librosa,ffmpeg
import soundfile as sf
import torch
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

class UVR5():
    def __init__(self, weight_path, device, is_half, temp_path, paths, agg):
        self.uvr5_weight_path = weight_path
        self.uvr5_weight_names = []
        for name in os.listdir(self.uvr5_weight_path):
            if name.endswith(".pth") or "onnx" in name:
                self.uvr5_weight_names.append(name.replace(".pth", ""))
        
        self.device = device
        self.is_half = is_half

        # 临时目录
        self.temp_path = temp_path

        # 保存输入音频文件夹路径
        self.input_path = os.path.join(self.temp_path, "uvr5", "input")

        # 保存主人声音频文件夹路径
        self.output_vocal_path = os.path.join(self.temp_path, "uvr5", "output_vocal")

        # 保存非主人声音频文件夹路径
        self.output_instrument_path = os.path.join(self.temp_path, "uvr5", "output_instrument")

        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_vocal_path, exist_ok=True)
        os.makedirs(self.output_instrument_path, exist_ok=True)

        self.agg = agg
    
    # model_name: 模型名
    # input_wave_file_path: 输入音频文件路径
    # aggressive : 人声提取激进程度
    # format0: 格式 wav/flac/mp3/m4a
    def run(self, model_name, input_wave_file_path, aggressive=10, format0="wav"):
        try:
            is_hp3 = "HP3" in model_name
            if model_name == "onnx_dereverb_By_FoxJoy":
                pre_fun = MDXNetDereverb(15)
            else:
                func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
                pre_fun = func(
                    agg = int(agg),
                    model_path = os.path.join(self.uvr5_weight_path, model_name + ".pth"),
                    device = device,
                    is_half = is_half,
                )
            if input_wave_file_path is None or input_wave_file_path == "":
                return "input_wave_file_path is None or empty"
            if not os.path.isfile(input_wave_file_path):
                return "input_wave_file_path is not a file"
            is_need_reformat = True
            is_done = False
            try:
                info = ffmpeg.probe(input_wave_file_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = False
                    pre_fun._path_audio_(
                        input_wave_file_path, self.output_instrument, self.output_vocal, format0, is_hp3
                    )
                    done = True
            except:
                need_reformat = True
                traceback.print_exc()
            if need_reformat == 1:
                reformatted_file_path = os.path.join(self.input_path, os.path.basename(input_wave_file_path) + ".reformatted.wav")
                os.system(
                    f'ffmpeg -i "{input_wave_file_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{reformatted_file_path}" -y'
                )
                input_wave_file_path = reformatted_file_path
            try:
                if not done:
                    pre_fun._path_audio_(
                        input_wave_file_path, self.output_instrument, self.output_vocal, format0, is_hp3
                    )
                return "%s->Success" % (os.path.basename(input_wave_file_path))
            except:
                return "%s->%s" % (os.path.basename(input_wave_file_path), traceback.format_exc())
        except:
            return traceback.format_exc()
        finally:
            try:
                if model_name == "onnx_dereverb_By_FoxJoy":
                    del pre_fun.pred.model
                    del pre_fun.pred.model_
                else:
                    del pre_fun.model
                    del pre_fun
            except:
                traceback.print_exc()
            print("clean_empty_cache")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


app = FastAPI()

uvr = UVR5("/home/foxjoy/Downloads/uvr5_weights", "cuda", False, "/home/foxjoy/Downloads/temp", [], 10)