import streamlit as st
import requests
import logging

if "input_wave_filename" not in st.session_state:
    st.session_state.input_wave_filename = ""

if "slice_processing" not in st.session_state:
    st.session_state.slice_processing = False

if "slice_processed" not in st.session_state:
    st.session_state.slice_processed = {}

if "subdir" not in st.session_state:
    st.session_state.subdir = ""

if "filename" not in st.session_state:
    st.session_state.filename = ""

class SliceUI:
    def __init__(self):
        self.train_host = "http://127.0.0.1:8001"
    
    def upload_wave_file(self, bytes_data):
        try:
            response = requests.post(f"{self.train_host}/slice/upload_wave_file",
                files={"file": bytes_data},
            )
            response.raise_for_status()
            res = response.json()
            st.session_state.subdir = res["subdir"]
            st.session_state.filename = res["filename"]
            return res
        except requests.exceptions.RequestException as e:
            logging.error(e)
    
    def input_wave_uploader_callback(self):
        if st.session_state.input_wave_file_uploader is not None:
            st.session_state.input_wave_filename = self.upload_wave_file(st.session_state.input_wave_file_uploader.getvalue())
    
    def run_slice(self):
        subdir = st.session_state.subdir
        filename = st.session_state.filename
        if subdir == "" or filename == "":
            logging.error("subdir or filename is empty")
            return "{'message': 'subdir or filename is empty'}"
        try:
            response = requests.post(f"{self.train_host}/slice",
                json={"subdir": subdir, "filename": filename},
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(e)
    
    def click_slice_run_button(self):
        st.session_state.slice_processing = True
        if st.session_state.input_wave_filename == "":
            logging.error("input_wave_filename is empty")
            return
        st.session_state.slice_processed["result"] = self.run_slice()
        st.session_state.slice_processing = False

    def run(self):
        st.title("音频切片")
        st.write("上传音频文件")
        input_wave_file = st.file_uploader("选择音频文件", type=["wav", "mp3"], key="input_wave_file_uploader", on_change=self.input_wave_uploader_callback, disabled=st.session_state.slice_processing)
        if input_wave_file is not None:
            st.audio(input_wave_file, format="audio/wav")
            st.write(st.session_state.input_wave_filename)
        
        slice_run_button = st.button("切片", on_click=self.click_slice_run_button, disabled=st.session_state.slice_processing)
        if "result" in st.session_state.slice_processed:
            st.write(st.session_state.slice_processed["result"])

if __name__ == "__main__":
    SliceUI().run()