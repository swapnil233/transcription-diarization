import sys
import whisper

def transcribe(audio_path):
    model = whisper.load_model("base.en")
    result = model.transcribe(audio_path)
    text = result["text"]
    
    with open('transcript.txt', 'w') as f:
        f.write(text)

    print(f'Transcript saved to transcript.txt')

if __name__ == '__main__':
    audio_path = sys.argv[1]
    transcribe(audio_path)
