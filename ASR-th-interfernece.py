import requests
import numpy
import io
import wave
import struct
import sounddevice as sd
from transformers import pipeline
from scipy.io.wavfile import write
from scipy.io import wavfile
from librosa import resample as librosa_resample
import numpy as np

TOKEN = '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'

API_URL = "https://api-inference.huggingface.co/models/DylanonWic/wav2vec2-large-asr-th-lm"
MODEL_PATH = r"C:\Users\dylan\Desktop\ASR-project\API\ASR_app\ASR_model-happy_pond"
headers = {"Authorization": f"Bearer {TOKEN}"}


def record_audio(filename='' ,time=3, sam_rate=16000):
    fs = sam_rate  # Sample rate
    seconds = 3  # Duration of recording

    print('reccording sound')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    print('reccord done')
    write(filename, fs, myrecording)  # Save as WAV file 


def predict_pipeline(input_arr):
    model = pipeline(task="automatic-speech-recognition", model=MODEL_PATH)
    if input_arr is not None:
        output = model(input_arr)
    else:
        return None
    return output['text']

def resample(audio_arr, orig_sr):
    audio_arr = np.array(audio_arr, dtype=np.double)
    return processor(audio_arr, sampling_rate=16_000).input_values[0]
    # return librosa_resample(audio_arr, orig_sr=orig_sr, target_sr=16_000)

def blob_to_raw(file):
    blob_data = file.read()
    # Convert the Blob data to a byte stream
    stream = io.BytesIO(blob_data)

    # Read the WAV file from the byte stream
    with wave.open(stream, 'rb') as wav_file:
        # Extract the audio file parameters
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()

        # Read all the audio frames from the WAV file
        frames = wav_file.readframes(wav_file.getnframes())

    # Convert the audio frames to a raw audio array
    num_samples = len(frames) // sample_width
    raw_array = struct.unpack('<{}h'.format(num_samples), frames)
    return {'audio_array': raw_array, 'rate': frame_rate}


# record_audio('a.wav',2, 48000)
samplerate, data = wavfile.read('a.wav')
arr = resample(data, samplerate)
print(arr.shape)
output = predict_pipeline(arr[:, 0])
print(output)