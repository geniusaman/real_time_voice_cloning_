from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import sounddevice as sd

# Load models and initialize synthesizer
encoder_weights = Path("/content/saved_models/default/encoder.pt")
vocoder_weights = Path("/content/saved_models/default/vocoder.pt")
synthesizer_weights = Path("/content/saved_models/default/synthesizer.pt")

encoder.load_model(encoder_weights)
vocoder.load_model(vocoder_weights)
synthesizer = Synthesizer(synthesizer_weights)

def generate_voice(text, device=None):
    # Preprocess the input text and generate the speech
    in_fpath = Path("/content/LRMonoPhase4.wav")
    reprocessed_wav = encoder.preprocess_wav(in_fpath)
    original_wav, sampling_rate = librosa.load(in_fpath)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Play the generated audio using sounddevice
    sd.play(generated_wav, synthesizer.sample_rate, device=device)
    sd.wait()


# Example usage:
if __name__ == "__main__":
    text = "Hello, this is a test."
    generate_voice(text, device=1)
