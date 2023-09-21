import numpy as np
import scipy.signal
from scipy.io import wavfile
from librosa import resample as librosa_resample
import noisereduce as nr

def read_wav(filename):
    # Load the WAV file and extract the sample rate and audio data
    sample_rate, audio_data = wavfile.read(filename)

    # Convert the audio data to a NumPy array of floating-point values
    audio_data = np.array(audio_data, dtype=float)
    return audio_data, sample_rate


def resample(audio_arr, orig_sr, target_sr=16000):
    audio_arr = np.array(audio_arr, dtype=np.double)
    return librosa_resample(audio_arr, orig_sr=orig_sr, target_sr=target_sr)


def do_noise_reduce(audio:np.array, rate=16000):
    """
    input: 1-dimension of numpy.ndimarray of audio noise
    """
    return nr.reduce_noise(y=audio, sr=rate, stationary=True)


def lowpass_filter(audio, rate=16000, cutoff_frequency=7000, order=1):
    '''
    apply lowpass filter to the audio data.
    '''
    if isinstance(audio, str):
        # Load the audio data.
        audio_data = np.load(audio)
        rate = audio_data.samplerate
    else:
        audio_data = audio

    # Create a lowpass filter.
    b, a = scipy.signal.butter(order, cutoff_frequency / (2 * rate), btype="low")

    # Apply the filter to the audio data.
    filtered_audio_data = scipy.signal.lfilter(b, a, audio_data)

    return filtered_audio_data


if __name__ == '__main__':
    import audiovis
    from asyncio import run
    import sys
    sys.path.append('.')
    from asr import asr
    import param
    
    # # audio, rate = np.random.randn(100000), 16000
    # audio, rate = read_wav('hello(asus).wav')
    # au_comp, _ = read_wav('hello(huawei).wav')
    # filtered = lowpass_filter(audio, sample_rate=rate, order=1, cutoff_frequency=8000)
    # # audiovis.spectrogram(audio, rate)
    # audiovis.spectrogram(filtered, rate)
    # audiovis.spectrogram(au_comp, rate)
    # audiovis.show()
    
    audio, rate = read_wav('sample_2023-05-11_15-24-42.wav')
    
    audio = resample(audio, rate)
    pos_au = do_noise_reduce(audio, 16000)
    pos_au = lowpass_filter(audio, 16000, 7500, 2)
    
    # audiovis.spectrogram(audio, 16000)
    # audiovis.spectrogram(pos_au, 16000)
    # audiovis.show()

    asr.load_model(param.MODEL_PATH)

    print( 'audio: '+ run(asr.predict_pipeline(audio, 16000, preprocess=False)))
    print( 'pos_au: '+ run(asr.predict_pipeline(pos_au, 16000, preprocess=False)))
    