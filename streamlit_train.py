import streamlit as st
import requests
import logging

if "input_wave_filename" not in st.session_state:
    st.session_state.input_wave_filename = ""

class UVR5UI:
    def __init__(self):
        self.train_host = "http://127.0.0.1:8001"
        self.uvr5_weight_names = []
        
        self.get_uvr5_weight_names()

        if len(self.uvr5_weight_names) == 0:
            logging.error("No uvr5 weight names")
            return

        self.model_name = self.uvr5_weight_names[0]
        self.format0 = "mp3"

    def get_uvr5_weight_names(self):
        try:
            response = requests.get(f"{self.train_host}/uvr5/weight_names")
            self.uvr5_weight_names = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error: {e}")

    def upload_wave_file(self, bytes_data):
        try:
            response = requests.post(f"{self.train_host}/uvr5/upload_wave_file",
                files={"file": bytes_data},
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(e)

    def input_wave_uploader_callback(self):
        st.session_state.input_wave_filename = self.upload_wave_file(st.session_state.input_wave_file_uploader.getvalue())

    def run_uvr5(self, model_name, input_wave_filename, aggressive, format0):
        try:
            response = requests.post(f"{self.train_host}/uvr5",
                json={"modelname": model_name, "aggressive": aggressive, "format0": format0, "filename": input_wave_filename},
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(e)

    def run(self):
        st.title("U-Net Voice Removal 5 (UVR5)")
        input_wave_file = st.file_uploader("Input wave file", type=["wav", "mp3"], on_change=self.input_wave_uploader_callback, key="input_wave_file_uploader")
        if input_wave_file:
            st.audio(input_wave_file, format="audio/wav")
            st.write(st.session_state.input_wave_filename)
        self.model_name = st.selectbox("Model name", self.uvr5_weight_names, index=self.uvr5_weight_names.index(self.model_name))
        self.format0 = st.selectbox("Format", ["wav", "flac", "mp3", "m4a"], index=["wav", "flac", "mp3", "m4a"].index(self.format0))
        if st.button("Run"):
            if st.session_state.input_wave_filename == "":
                st.error("Please upload a wave file")
                return
            result = self.run_uvr5(self.model_name, st.session_state.input_wave_filename, 10, self.format0)
            st.write(result)

if __name__ == "__main__":
    UVR5UI().run()
