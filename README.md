# Bulk conversion of video files to transcripts with speaker diarization
This is a tool that uses OpenAI's Whisper Automatic Speech Recognition (ASR) system and pyannote.audio diarization to transcribe audio from video files, generate speaker embeddings and identify speakers in the video. All computation is done on the local machine - using your GPU/CPU and RAM.

## Features
Transcribes video files.
Converts video files to audio for processing.
Saves transcriptions to both plain text and JSON format.
Labels speakers in transcriptions.

## Usage
1. Install the necessary packages using pip.
```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip install setuptools-rust
pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
pip install numpy
pip install torch
```

Note: The script currently uses `ffmpeg` to convert video to audio. Make sure you have ffmpeg installed and added to your PATH.

2. Clone this repository or copy the Python script `transcribe_and_diarize.py`
3. Place your video files into a folder named `videos` in the same directory as the script. Also, make a folder called `audios` and another named `transcriptions`.
4. Run `transcribe_and_diarize.py`. 
```python
python transcribe_and_diarize.py
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
        "speaker": "SPEAKER",
        "timestamp": "0:00:00",
        "text": "Hello, how are you? "
    }
]
```

## Limitations
Currently, the script doesn't differentiate between different speakers. It labels all spoken segments with a generic "SPEAKER" label. If you want to implement automatic speaker diarization, you would need to modify the script to adjust the number of speakers and properly cluster the speaker embeddings.

Also, the Electron desktop app isn't working at the moment so as of now, you can ignore the Electron files (main.js, index.html, etc).
