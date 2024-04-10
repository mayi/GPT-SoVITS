from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from typing import List
from pydantic import BaseModel
import uuid
import sys
import os, re, logging
import traceback
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
    def __init__(self, weight_path, device, is_half, temp_path):
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
    
    # model_name: 模型名
    # input_wave_file_path: 输入音频文件路径
    # aggressive : 人声提取激进程度 目前没有用到
    # format0: 格式 wav/flac/mp3/m4a
    def run(self, model_name, input_wave_file_path, aggressive=10, format0="wav"):
        pre_fun = None
        is_need_reformat = True
        is_done = False
        try:
            is_hp3 = "HP3" in model_name
            if model_name == "onnx_dereverb_By_FoxJoy":
                pre_fun = MDXNetDereverb(15)
            else:
                func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
                pre_fun = func(
                    agg = int(aggressive),
                    model_path = os.path.join(self.uvr5_weight_path, model_name + ".pth"),
                    device = self.device,
                    is_half = self.is_half,
                )
            if input_wave_file_path is None or input_wave_file_path == "":
                return "input_wave_file_path is None or empty"
            if not os.path.isfile(input_wave_file_path):
                return "input_wave_file_path is not a file"
            result = {"instrument": "", "vocal": ""}
            try:
                info = ffmpeg.probe(input_wave_file_path, cmd="ffprobe")
                stream0 = info["streams"][0]
                if (stream0["channels"] == 2 and stream0["sample_rate"] == "44100"):
                    is_need_reformat = False
                    result = pre_fun._path_audio_(
                        input_wave_file_path, self.output_instrument_path, self.output_vocal_path, format0, is_hp3
                    )
                    is_done = True
            except:
                is_need_reformat = True
                traceback.print_exc()
            if is_need_reformat:
                reformatted_file_path = os.path.join(self.input_path, os.path.basename(input_wave_file_path) + ".reformatted.wav")
                os.system(
                    f'ffmpeg -i "{input_wave_file_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{reformatted_file_path}" -y'
                )
                input_wave_file_path = reformatted_file_path
            try:
                if not is_done:
                    result = pre_fun._path_audio_(
                        input_wave_file_path, self.output_instrument_path, self.output_vocal_path, format0, is_hp3
                    )
                vocal_filename = os.path.basename(result["vocal"])
                instrument_filename = os.path.basename(result["instrument"])
                if vocal_filename.startswith("instrument") and instrument_filename.startswith("vocal"):
                    vocal_filename, instrument_filename = instrument_filename, vocal_filename
                result["vocal"] = vocal_filename
                result["instrument"] = instrument_filename
                return {"message": "%s->Success" % (os.path.basename(input_wave_file_path)), "result": result}
            except:
                return {"message": "%s->%s" % (os.path.basename(input_wave_file_path), traceback.format_exc())}
        except:
            return {"message": traceback.format_exc()}
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

weight_path = os.environ.get("UVR5_WEIGHT_PATH")

if weight_path is None:
    print("UVR5_WEIGHT_PATH is None. Please set the environment variable UVR5_WEIGHT_PATH.")
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"

temp_path = os.environ.get("TEMP_PATH")
if temp_path is None:
    print("TEMP_PATH is None. Please set the environment variable TEMP_PATH.")
    exit()

uvr = UVR5(weight_path, device, False, temp_path)

class UVR5Request(BaseModel):
    modelname: str
    aggressive: int
    format0: str
    filename: str


@app.post("/uvr5")
async def uvr5(request: UVR5Request):
    file_path = os.path.join(uvr.input_path, request.filename)
    result = uvr.run(request.modelname, file_path, request.aggressive, request.format0)
    return result

@app.get("/uvr5/weight_names")
async def uvr5_weight_names():
    return uvr.uvr5_weight_names

@app.post("/uvr5/upload_wave_file")
async def upload_wave_file(file: UploadFile = File(...)):
    filename = uuid.uuid4().hex
    file_path = os.path.join(uvr.input_path, filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return filename

@app.get("/uvr5/output/vocal/{filename}")
async def output_vocal(filename: str):
    file_path = os.path.join(uvr.output_vocal_path, filename)
    return FileResponse(file_path)

@app.get("/uvr5/output/instrument/{filename}")
async def output_instrument(filename: str):
    file_path = os.path.join(uvr.output_instrument_path, filename)
    return FileResponse(file_path)
