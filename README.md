# Bulk conversion of video files to transcripts with speaker diarization
This is a tool that uses OpenAI's Whisper Automatic Speech Recognition (ASR) system and pyannote.audio diarization to transcribe audio from video files, generate speaker embeddings and identify speakers in the video. All computation is done on the local machine - using your GPU/CPU and RAM.

There are two scripts: 
1. `transcribe_diarize.py` and 
2. `transcribe_diarize_multiprocessing.py`. 

As the name suggests, `transcribe_diarize_multiprocessing.py` will use Python's multiprocessing API to do all tasks in parallel. *Don't run `transcribe_diarize_multiprocessing.py` with a Whisper model bigger than `base.en` if your computer doesn't have at least 12GB RAM*

## Features
- Converts video files to audio for processing using `ffmpeg`.
- Transcribes video files using Whisper.
- Saves transcriptions to both plain text and JSON format (with UUID for each segment).

## Usage
**Note**: you can configure the exact pytorch installation from [here]([url](https://pytorch.org/)). The instructions below will install the CUDA 11.8 version - which uses your Nvidia GPU for computation.

**Note**: The script uses `ffmpeg` to convert video to audio. Make sure you have ffmpeg installed and added to your PATH. 

1. Install the necessary packages using pip.
```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip install setuptools-rust
pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
pip install numpy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
2. Clone this repository or copy the Python script `transcribe_and_diarize.py`
3. Place your video files into a folder named `videos` in the same directory as the script.
4. Run `transcribe_diarize.py` or `transcribe_diarize_multiprocessing.py` 
```python
python transcribe_diarize.py
```
The script will automatically convert the video files into audio, transcribe them, and save the transcriptions into a folder named transcriptions.

## Output
The script saves transcriptions in both plain text (.txt) and JSON (.json) format. Each transcription file is named after the original video file. For example, the transcription for video1.mp4 would be video1.txt and video1.json.

In the text transcriptions, each speaker's lines are labeled with a generic "SPEAKER" label and a timestamp indicating the start of the spoken segment. Here's an example:
```
SPEAKER 0:00:00
Hello, how are you?
```

In the JSON transcriptions, each spoken segment is represented as an object with "speaker", "timestamp", and "text" properties:
```json
[
    {
        "id": "9ea4cfe4-8b85-4957-abae-b53c19042822",
        "speaker": "SPEAKER",
        "timestamp": "0:00:00",
        "text": "Hello, how are you? "
    }
]
```

## Limitations
Currently, the script doesn't differentiate between different speakers. It labels all spoken segments with a generic "SPEAKER" label. If you want to implement automatic speaker diarization, you would need to modify the script to adjust the number of speakers and properly cluster the speaker embeddings.

Also, the Electron desktop app isn't working at the moment so as of now, you can ignore the Electron files (main.js, index.html, etc).
