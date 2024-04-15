import gradio as gr
import param
from asr import asr
import asyncio
import soundfile as sf
import librosa
import numpy as np


asr.load_model(param.MODEL_NAME)


def transcribe(audio):
    if audio is not None:
        #check if audio is file path
        if isinstance(audio, str):
            audio_arr, rate = librosa.load(audio)
        else:
            rate, audio_arr = audio
            
        if asr.IS_MODEL_LOADED:
            text = asyncio.run(asr.predict_pipeline(audio_arr.astype(np.float32), rate=rate, preprocess=True))
        else:
            text = 'ASR model not load.'
        
        return text
    
    return ''

gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.Audio(source="microphone", type="filepath", streaming=False), 
    ],
    outputs=[
        "textbox"
    ],
    title='ASR application',
    live=False).launch(share=True)
