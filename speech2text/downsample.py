from scipy.io import wavfile
import scipy
import os
import numpy as np


def resample_audio(adjusted_sample_rate, audio_path):
    resampled_audio_path = "resampled_audio"
    files = os.listdir(audio_path)
    for file in files:
        try:
            sampling_rate, data = wavfile.read("{}/{}".format(audio_path, file))
            scipy.io.wavfile.write(
                resampled_audio_path + "/re_" + file, adjusted_sample_rate, data
            )
        except:
            pass


if __name__ == "__main__":
    # resample_audio(adjusted_sample_rate=16000, audio_path='audio_files/')
    sampling_rate, data = wavfile.read("resampled_audio/re_LJ001-0001.wav")
    print(np.array(data))
