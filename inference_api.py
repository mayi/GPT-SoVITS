from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from typing import List
from pydantic import BaseModel


'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
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
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class Inference:
    def __init__(self):
        self.model_path = os.environ.get("MODEL_PATH", "GPT_SoVITS/pretrained_models")
        self.cnhubert_base_path = self.model_path + "/chinese-hubert-base"
        self.bert_path = self.model_path + "/chinese-roberta-wwm-ext-large"
        self.weights_path = os.environ.get("WEIGHTS_PATH", "GPT_SoVITS/weights")
        # 预提供的权重
        self.pretrained_sovits_path = self.weights_path + "/pretrained_SoVITS_weights/s2G488k.pth"
        self.pretrained_gpt_path = self.weights_path + "/pretrained_GPT_weights/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        # 其他的权重
        self.gpt_weights_path = self.weights_path + "/GPT_weights"
        self.sovits_weights_path = self.weights_path + "/SoVITS_weights"

        self.gpt_path = self.pretrained_gpt_path
        self.sovits_path = self.pretrained_sovits_path
        
        os.makedirs(self.sovits_weights_path, exist_ok=True)
        os.makedirs(self.gpt_weights_path, exist_ok=True)

        self.is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if os.path.exists("./gweight.txt"):
            with open("./gweight.txt", 'r', encoding="utf-8") as file:
                self.gpt_path = file.read()

        if os.path.exists("./sweight.txt"):
            with open("./sweight.txt", 'r', encoding="utf-8") as file:
                self.sovits_path = file.read()

        if "_CUDA_VISIBLE_DEVICES" in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
        is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

        self.cnhubert = cnhubert
        self.cnhubert.cnhubert_base_path = self.cnhubert_base_path

        self.dict_language = {
            i18n("中文"): "all_zh",#全部按中文识别
            i18n("英文"): "en",#全部按英文识别#######不变
            i18n("日文"): "all_ja",#全部按日文识别
            i18n("中英混合"): "zh",#按中英混合识别####不变
            i18n("日英混合"): "ja",#按日英混合识别####不变
            i18n("多语种混合"): "auto",#多语种启动切分识别语种
        }
        
        self.hz = 50

        self.dtype = torch.float16 if self.is_half == True else torch.float32

        self.print_settings()
        self.load_model()
        self.change_gpt_weights(self.gpt_path)
        self.change_sovits_weights(self.sovits_path)

        self.splitter = SplitText()

        self.get_weights_names()

    def print_settings(self):
        print("cnhubert_base_path:", self.cnhubert_base_path)
        print("bert_path:", self.bert_path)
        print("pretrained_sovits_path:", self.pretrained_sovits_path)
        print("pretrained_gpt_path:", self.pretrained_gpt_path)
        print("gpt_weights_path:", self.gpt_weights_path)
        print("sovits_weights_path:", self.sovits_weights_path)

        print("gpt_path:", self.gpt_path)
        print("sovits_path:", self.sovits_path)
        print("is_half:", self.is_half)
        print("device:", self.device)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
        if self.is_half == True:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)


        self.ssl_model = self.cnhubert.get_model()
        if self.is_half == True:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T
    
    def change_sovits_weights(self, sovits_path):
        print(sovits_path)
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        self.hps = dict_s2["config"]
        self.hps = DictToAttrRecursive(self.hps)
        self.hps.model.semantic_frame_rate = "25hz"
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers = self.hps.data.n_speakers,
            **self.hps.model
        )
        if ("pretrained" not in sovits_path):
            del self.vq_model.enc_q
        if self.is_half == True:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)
        self.vq_model.eval()
        print(self.vq_model.load_state_dict(dict_s2["weight"], strict=False))
        with open("./sweight.txt", "w", encoding="utf-8") as f:
            f.write(sovits_path)

    def change_gpt_weights(self, gpt_path):
        print(gpt_path)
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]
        self.t2s_model = Text2SemanticLightningModule(self.config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half == True:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()
        total = sum([param.nelement() for param in self.t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        with open("./gweight.txt", "w", encoding="utf-8") as f:
            f.write(gpt_path)

    def get_spec(self, hps, filename):
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            center=False,
        )
        return spec
    
    def clean_text_inf(self, text, language):
        phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_","")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            bert = torch.zeros((1024, len(phones)), dtype=self.dtype,).to(self.device)
        return bert

    def get_first(self, text):
        pattern = "[" + "".join(re.escape(sep) for sep in self.splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text

    def get_phones_and_bert(self, text, language):
        if language in {"en", "all_zh", "all_ja"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
            if language == "zh":
                bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            else:
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=self.dtype,
                ).to(self.device)
        elif language in {"zh", "ja", "auto"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "ko":
                        langlist.append("zh")
                        textlist.append(tmp["text"])
                    else:
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)
        return phones, bert.to(self.dtype), norm_text
        
    def merge_short_text_in_array(self, texts, threshold):
        if (len(texts)) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if (len(text) > 0):
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result
    
    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free=False):
        if prompt_text is None or len(prompt_text) == 0:
            ref_free = True
        t0 = ttime()
        prompt_language = self.dict_language[prompt_language]
        text_language = self.dict_language[text_language]
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in self.splits):
                prompt_text += "。" if prompt_language != "en" else "."
            print(i18n("实际输入的参考文本:"), prompt_text)
        text = text.strip("\n")
        if (text[0] not in self.splits and len(self.get_first(text)) < 4):
            text = "。" + text if text_language != "en" else "." + text
        
        print(i18n("实际输入的目标文本:"), text)
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if self.is_half == True else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if self.is_half == True:
                wav16k = wav16k.half().to(self.device)
                zero_wav_torch = zero_wav_torch.half().to(self.device)
            else:
                wav16k = wav16k.to(self.device)
                zero_wav_torch = zero_wav_torch.to(self.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )
            codes = self.vq_model.extract_latent(ssl_content)

            prompt_semantic = codes[0, 0]
        t1 = ttime()

        if (how_to_cut == i18n("凑四句一切")):
            text = self.splitter.cut1(text)
        elif (how_to_cut == i18n("凑50字一切")):
            text = self.splitter.cut2(text)
        elif (how_to_cut == i18n("按中文句号。切")):
            text = self.splitter.cut3(text)
        elif (how_to_cut == i18n("按英文句号.切")):
            text = self.splitter.cut4(text)
        elif (how_to_cut == i18n("按标点符号切")):
            text = self.splitter.cut5(text)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        print(i18n("实际输入的目标文本(切句后):"), text)
        texts = text.split("\n")
        texts = self.merge_short_text_in_array(texts, 5)
        audio_opt = []
        if not ref_free:
            phones1, bert1, norm_text1 = self.get_phones_and_bert(prompt_text, prompt_language)

        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in self.splits):
                text += "。" if text_language != "en" else "."
            print(i18n("实际输入的目标文本(每句):"), text)
            phones2, bert2, norm_text2 = self.get_phones_and_bert(text, text_language)
            print(i18n("前端处理后的文本(每句):"), norm_text2)
            if not ref_free:
                bert = torch.cat([bert1, bert2], 1)
                all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            else:
                bert = bert2
                all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)

            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = prompt_semantic.unsqueeze(0).to(self.device)
            t2 = ttime()
            with torch.no_grad():
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=self.hz * self.max_sec,
                )
            t3 = ttime()
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            refer = self.get_spec(self.hps, ref_wav_path)
            if self.is_half == True:
                refer = refer.half().to(self.device)
            else:
                refer = refer.to(self.device)
            audio = (
                self.vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refer
                )
                    .detach()
                    .cpu()
                    .numpy()[0, 0]
            )
            #简单防止16bit爆音
            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio /= max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
            np.int16
        )

    def get_weights_names(self):
        self.SoVITS_names = [self.pretrained_sovits_path]
        for name in os.listdir(self.sovits_weights_path):
            if name.endswith(".pth"): self.SoVITS_names.append("%s/%s" % (self.sovits_weights_path, name))
        self.GPT_names = [self.pretrained_gpt_path]
        for name in os.listdir(self.gpt_weights_path):
            if name.endswith(".ckpt"): self.GPT_names.append("%s/%s" % (self.gpt_weights_path, name))
        return self.SoVITS_names, self.GPT_names

class SplitText:
    def __init__(self):
        self.splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


    def split(self, todo_text):
        todo_text = todo_text.replace("……", "。").replace("——", "，")
        if todo_text[-1] not in self.splits:
            todo_text += "。"
        i_split_head = i_split_tail = 0
        len_text = len(todo_text)
        todo_texts = []
        while 1:
            if i_split_head >= len_text:
                break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
            if todo_text[i_split_head] in self.splits:
                i_split_head += 1
                todo_texts.append(todo_text[i_split_tail:i_split_head])
                i_split_tail = i_split_head
            else:
                i_split_head += 1
        return todo_texts

    def cut1(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        split_idx = list(range(0, len(inps), 4))
        split_idx[-1] = None
        if len(split_idx) > 1:
            opts = []
            for idx in range(len(split_idx) - 1):
                opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
        else:
            opts = [inp]
        return "\n".join(opts)
    
    def cut2(self, inp):
        inp = inp.strip("\n")
        inps = self.split(inp)
        if len(inps) < 2:
            return inp
        opts = []
        summ = 0
        tmp_str = ""
        for i in range(len(inps)):
            summ += len(inps[i])
            tmp_str += inps[i]
            if summ > 50:
                summ = 0
                opts.append(tmp_str)
                tmp_str = ""
        if tmp_str != "":
            opts.append(tmp_str)
        if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
            opts[-2] = opts[-2] + opts[-1]
            opts = opts[:-1]
        return "\n".join(opts)

    def cut3(self, inp):
        inp = inp.strip("\n")
        return "\n".join(["%s" % item for item in inp.strip("。").split("。")])

    def cut4(self, inp):
        inp = inp.strip("\n")
        return "\n".join(["%s" % item for item in inp.strip(".").split(".")])

    # contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
    def cut5(self, inp):
        # if not re.search(r'[^\w\s]', inp[-1]):
        # inp += '。'
        inp = inp.strip("\n")
        punds = r'[,.;?!、，。？！;：…]'
        items = re.split(f'({punds})', inp)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items)%2 == 1:
            mergeitems.append(items[-1])
        opt = "\n".join(mergeitems)
        return opt

app = FastAPI()

inference = Inference()

class ChangeWeightsRequest(BaseModel):
    path: str

class CutTextRequest(BaseModel):
    text: str
    how_to_cut: int

class TTSRequest(BaseModel):
    ref_wav: UploadFile
    prompt_text: str
    prompt_language: str
    text: str
    text_language: str
    how_to_cut: int
    top_k: int
    top_p: float
    temperature: float
    ref_free: bool

@app.post("/tts")
async def tts(ref_wav: UploadFile = File(...), prompt_text: str = None, prompt_language: str = "中文", text: str = None, text_language: str = "中文", how_to_cut: str = "不切", top_k: int = 20, top_p: float = 0.6, temperature: float = 0.6, ref_free: bool = False):
    ref_wav_path = "./ref.wav"
    with open(ref_wav_path, "wb") as f:
        f.write(ref_wav.file.read())
    sampling_rate, audio = inference.get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free)
    return FileResponse(audio, media_type="audio/wav")

@app.get("/change_choices")
async def change_choices():
    return inference.change_choices()

@app.get("/get_weights_names")
async def get_weights_names():
    return inference.get_weights_names()

@app.get("/get_current_weights")
async def get_current_weights():
    return inference.sovits_path, inference.gpt_path

@app.post("/change_sovits_weights")
async def change_sovits_weights(request: ChangeWeightsRequest):
    inference.change_sovits_weights(request.path)
    return "Success"

@app.post("/change_gpt_weights")
async def change_gpt_weights(request: ChangeWeightsRequest):
    inference.change_gpt_weights(request.path)
    return "Success"

@app.post("/cut_text")
async def cut_text(request: CutTextRequest):
    text = request.text
    how_to_cut = request.how_to_cut
    if how_to_cut == 1:
        text = inference.splitter.cut1(text)
    elif how_to_cut == 2:
        text = inference.splitter.cut2(text)
    elif how_to_cut == 3:
        text = inference.splitter.cut3(text)
    elif how_to_cut == 4:
        text = inference.splitter.cut4(text)
    elif how_to_cut == 5:
        text = inference.splitter.cut5(text)
    return text
