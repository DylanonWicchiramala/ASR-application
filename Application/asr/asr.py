from transformers import pipeline
import io
import soundfile as sf
import wave
import struct
import numpy as np
import param
from tools import audio_process
from torch import cuda

__version__ = param.__version__
model = None
IS_MODEL_LOADED = False

INP_RATE = 16000

def set_input_sampling_rate(sampling_rate):
    global INP_RATE
    INP_RATE = sampling_rate

def load_model(model_name):
    global model, IS_MODEL_LOADED
    print('Loading model ', model_name)
    
    if cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'     
    
    model = pipeline(task="automatic-speech-recognition", model=model_name, device=device)
    
    IS_MODEL_LOADED = True


async def blob_to_raw(file):
    blob_data = await file.read()
    # Convert the Blob data to a byte stream
    stream = io.BytesIO(blob_data)

    # Read the WAV file from the byte stream
    with wave.open(stream, 'rb') as wav_file:
        # Extract the audio file parameters
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()

        # Read all the audio frames from the WAV file
        frames = wav_file.readframes(wav_file.getnframes())

    # Convert the audio frames to a raw audio array
    num_samples = len(frames) // sample_width
    raw_array = struct.unpack('<{}h'.format(num_samples), frames)
    return {'audio_array': raw_array, 'rate': frame_rate}


async def byte_to_raw(file:bin):
    data, samplerate = sf.read(io.BytesIO(file))
    return  data, samplerate
    

def predict(input_aud_arr:np.array):
    output = model(input_aud_arr)
    return output['text']


def audio_preprocessor(input_aud_arr:np.array, rate=16000):

    # apply lowpass filtering
    input_aud_arr = audio_process.lowpass_filter(input_aud_arr, rate=rate, cutoff_frequency=7500, order=2)
    # noise reduction
    input_aud_arr = audio_process.do_noise_reduce(input_aud_arr)
    
    return input_aud_arr


async def predict_pipeline(audio:np.ndarray or bin, rate=None, preprocess=True, model=model):
    """
    input: blob object of audio file encoding wav.
    output: (str) the transcription of audio file.
    """
    if audio is not None:
        # convert blob object to raw array of audio file
        if not isinstance(audio, np.ndarray):
            raw_arr ,rate = await byte_to_raw(audio)
        else:
            raw_arr = audio
            if rate is None:
                rate = INP_RATE
        
        if len(raw_arr) < 400:
            return ''
        
        # resampling the audio to 16_000 Hz 
        raw_arr = audio_process.resample(raw_arr, rate)
        
        
        if preprocess:
            raw_arr = audio_preprocessor(raw_arr, rate)
        
        # predict transcriptions
        output = predict(raw_arr)
        return output
    else:
        return None


if __name__ == '__main__':
    pass