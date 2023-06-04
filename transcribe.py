import sys
import whisper
import time

audio_path = (
    "C:\\Users\\swapn\\Desktop\\GitHub Repos\\transcription-desktop\\interviewVideo.wav"
)


def transcribe(audio_path):
    print("Loading model...")
    model = whisper.load_model("large")
    print("Model loaded, starting transcription...")

    print("Transcribing audio file: " + audio_path)

    start_time = time.time()

    result = model.transcribe(audio_path, fp16=False, language="English")

    elapsed_time = time.time() - start_time
    print(f"Transcription completed in {elapsed_time:.2f} seconds.")

    text = result["text"]

    with open("transcript_large.txt", "w") as f:
        if isinstance(text, list):
            text = " ".join(text)
        f.write(text)

    print(f"Transcript saved to transcript.txt")

    return text


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "C:\\Users\\swapn\\Desktop\\GitHub Repos\\transcription-desktop\\interviewVideo.wav"
    transcribe(audio_path)
