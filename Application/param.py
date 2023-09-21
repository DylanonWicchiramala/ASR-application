import os

MODEL_NAME = "wav2vec2-large-asr-th-2"
# MODEL_NAME = "wav2vec2-large-asr-th-4-gram"

MODEL_PATH = os.path.abspath(os.path.join( "./asr/", MODEL_NAME))
# MODEL_PATH = "./asr/" + MODEL_NAME
USE_GPU = True
__version__ = "magic-sound-184"
SAMPLE_RATE=16000
FRAMES_DURATION = 20    # milliseconds
FRAMES_PER_BUFFER = int(SAMPLE_RATE * FRAMES_DURATION / 1000)