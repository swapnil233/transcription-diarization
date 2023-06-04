import whisper
import datetime
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json
import subprocess
import glob
import os

# Change this to use different models (tiny.en, base.en, small.en, medium.en, large)
model_name = "tiny.en"


# Using ffmpeg to convert video to audio. Must have ffmpeg installed and in PATH.
def video_to_audio():
    """Convert all video files to audio with ffmpeg"""
    video_files = glob.glob("./videos/*.mp4")

    for video_file in video_files:
        audio_file = "./audios/" + os.path.basename(video_file).split(".")[0] + ".wav"
        print(f"Converting {video_file} to {audio_file}...")

        # Audio files need to be 16kHz, 1 channel, 16-bit PCM for Whisper
        command = ["ffmpeg", "-i", video_file, "-ar", "16000", "-ac", "1", audio_file]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"Successfully converted {video_file} to {audio_file}.")


# Use GPU if available
def init_device():
    """Initialize the device for torch operations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        print("Using CPU for computations")

    return device


def load_models(device, model_name):
    """Load both the pre-trained speaker embedding and transcription models"""
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", device
    )

    embedding_model = embedding_model.to(device)

    transcription_model = whisper.load_model(model_name)
    transcription_model = transcription_model.to(device)

    print("Models loaded, ready for transcription.")

    return embedding_model, transcription_model


def transcribe_audio(model, audio_path):
    """Transcribe audio using the given model and audio path"""
    result = model.transcribe(audio_path)
    print("Transcription completed.")

    return result["segments"]


def get_audio_duration(audio_path):
    """Get the duration of the audio file"""
    with contextlib.closing(wave.open(audio_path, "r")) as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)

    return duration


def segment_embedding(segment, duration, audio_path, embedding_model, device):
    """Generate embeddings for a segment"""
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    audio = Audio()
    waveform, sample_rate = audio.crop(audio_path, clip)

    waveform = waveform.to(device)  # move data to GPU

    return embedding_model(waveform[None])


def cluster_embeddings(embeddings, num_speakers):
    """Cluster embeddings into groups representing different speakers"""
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    print("Clustering completed.")

    return labels


def write_transcript_to_txt(segments, audio_file):
    """Write the transcriptions into a text file with speaker labels and timestamps"""
    transcript_file = (
        "./transcriptions/" + os.path.basename(audio_file).split(".")[0] + ".txt"
    )

    def time(secs):
        return datetime.timedelta(seconds=round(secs))

    with open(transcript_file, "w") as f:
        for segment in segments:
            f.write("\nSPEAKER " + str(time(segment["start"])) + "\n")
            f.write(segment["text"][1:] + " ")
    print(f"Transcription written to '{transcript_file}'.")


def write_transcript_to_json(segments, audio_file):
    """Write the transcriptions into a json file with speaker labels and timestamps"""
    transcript_file = (
        "./transcriptions/" + os.path.basename(audio_file).split(".")[0] + ".json"
    )

    def time(secs):
        return str(datetime.timedelta(seconds=round(secs)))

    transcript = []

    for segment in segments:
        if not transcript or transcript[-1]["text"][-1] == " ":
            timestamp = time(segment["start"])
            text = segment["text"][1:] + " "
            transcript.append(
                {"speaker": "SPEAKER", "timestamp": timestamp, "text": text}
            )
        else:
            transcript[-1]["text"] += segment["text"][1:] + " "

    with open(transcript_file, "w") as f:
        json.dump(transcript, f, indent=4)

    print(f"Transcription written to '{transcript_file}'.")


def main():
    """Main function to control flow of operations"""
    device = init_device()
    embedding_model, transcription_model = load_models(device, model_name)

    video_to_audio()

    audio_files = glob.glob("./audios/*.wav")

    for audio_file in audio_files:
        segments = transcribe_audio(transcription_model, audio_file)
        duration = get_audio_duration(audio_file)

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(
                segment, duration, audio_file, embedding_model, device
            )

        embeddings = np.nan_to_num(embeddings)

        num_speakers = 1  # we are not distinguishing between speakers
        labels = cluster_embeddings(embeddings, num_speakers)

        write_transcript_to_txt(segments, audio_file)
        write_transcript_to_json(segments, audio_file)


if __name__ == "__main__":
    main()
