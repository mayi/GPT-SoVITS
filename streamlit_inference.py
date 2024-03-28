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
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

# 改写为streamlit代码
import streamlit as st

import requests

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

class InferenceUI:
    def __init__(self):
        self.gpt_path = ""
        self.sovits_path = ""
        self.prompt_text = ""
        self.prompt_language = "中文"
        self.text = ""
        self.text_language = "中文"
        self.how_to_cut = "凑四句一切"
        self.top_k = 5
        self.top_p = 1.0
        self.temperature = 1.0
        self.ref_text_free = False
        self.SoVITS_names = []
        self.GPT_names = []

        self.inference_host = "http://127.0.0.1:8000/"

        self.get_current_weights()
        self.get_weights_names()

        st.session_state.text_opt = ""

    def get_current_weights(self):
        try:
            response = requests.get(self.inference_host + "get_current_weights")
            response.raise_for_status()
            weights = response.json()
            self.sovits_path = weights[0]
            self.gpt_path = weights[1]
        except requests.exceptions.RequestException as e:
            logging.error(e)

    def get_weights_names(self):
        try:
            response = requests.get(self.inference_host + "get_weights_names")
            response.raise_for_status()
            weights = response.json()
            self.SoVITS_names = weights[0]
            self.GPT_names = weights[1]
        except requests.exceptions.RequestException as e:
            logging.error(e)

    def change_sovits_weights(self, sovits_path):
        try:
            response = requests.post(self.inference_host + "change_sovits_weights", json={"path": sovits_path})
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(e)

    def change_gpt_weights(self, gpt_path):
        try:
            response = requests.post(self.inference_host + "change_gpt_weights", json={"path": gpt_path})
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(e)

    def get_tts_wav(self, ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_free):
        m = MultipartEncoder(fields={"file": ("filename", ref_wav_path, "audio/mp3")})

        r = requests.post(
            server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
        )

        return r

        try:
            response = requests.post(
                self.inference_host + "tts",
                json={
                    "ref_wav_path": ref_wav_path,
                    "prompt_text": prompt_text,
                    "prompt_language": prompt_language,
                    "text": text,
                    "text_language": text_language,
                    "how_to_cut": how_to_cut,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "ref_free": ref_free,
                },
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(e)

    def split(self, todo_text):
        return split(todo_text)
    
    def cut_text(self, text, how_to_cut):
        try:
            response = requests.post(self.inference_host + "cut_text", json={"text": text, "how_to_cut": how_to_cut})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(e)

    def run(self):
        st.title("GPT-SoVITS WebUI")
        st.markdown(i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. 如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE."))
        st.markdown(i18n("模型切换"))
        GPT_dropdown = st.selectbox(i18n("GPT模型列表"), sorted(self.GPT_names, key=custom_sort_key), index=self.GPT_names.index(self.gpt_path))
        SoVITS_dropdown = st.selectbox(i18n("SoVITS模型列表"), sorted(self.SoVITS_names, key=custom_sort_key), index=self.SoVITS_names.index(self.sovits_path))
        self.sovits_path = SoVITS_dropdown
        self.gpt_path = GPT_dropdown
        self.change_sovits_weights(SoVITS_dropdown)
        self.change_gpt_weights(GPT_dropdown)
        st.markdown(i18n("*请上传并填写参考信息"))
        inp_ref = st.file_uploader(i18n("请上传3~10秒内参考音频，超过会报错！"), type=["wav", "mp3"])
        ref_text_free = st.checkbox(i18n("开启无参考文本模式。不填参考文本亦相当于开启。"), value=self.ref_text_free)
        st.markdown(
            i18n("使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开，开启后无视填写的参考文本。")
        )
        prompt_text = st.text_area(i18n("参考音频的文本"), self.prompt_text)
        prompt_language = st.selectbox(i18n("参考音频的语种"), (i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")))
        st.markdown(i18n("*请填写需要合成的目标文本和语种模式"))
        text = st.text_area(i18n("需要合成的文本"), self.text)
        text_language = st.selectbox(i18n("需要合成的语种"), (i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")))
        how_to_cut = st.radio(i18n("怎么切"), (i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切")))
        top_k = st.slider(i18n("top_k"), 1, 100, self.top_k)
        top_p = st.slider(i18n("top_p"), 0.0, 1.0, self.top_p, 0.05)
        temperature = st.slider(i18n("temperature"), 0.0, 1.0, self.temperature, 0.05)
        inference_button = st.button(i18n("合成语音"))
        if inference_button:
            sr, wav = self.get_tts_wav(inp_ref, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_text_free)
            st.audio(wav, format="audio/wav")
        st.markdown(i18n("文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。"))
        text_inp = st.text_area(i18n("需要合成的切分前文本"), "")
        button1 = st.button(i18n("凑四句一切"))
        button2 = st.button(i18n("凑50字一切"))
        button3 = st.button(i18n("按中文句号。切"))
        button4 = st.button(i18n("按英文句号.切"))
        button5 = st.button(i18n("按标点符号切"))
        if button1:
            st.session_state.text_opt = self.cut_text(text_inp, 1)
        if button2:
            st.session_state.text_opt = self.cut_text(text_inp, 2)
        if button3:
            st.session_state.text_opt = self.cut_text(text_inp, 3)
        if button4:
            st.session_state.text_opt = self.cut_text(text_inp, 4)
        if button5:
            st.session_state.text_opt = self.cut_text(text_inp, 5)
        st.text_area(i18n("切分前文本："), st.session_state.text_opt)
        st.markdown(i18n("后续将支持转音素、手工修改音素、语音合成分步执行。"))

if __name__ == "__main__":
    InferenceUI().run()