#audio.py
import pyaudio
import wave
import speech_recognition as sr
import threading
import os

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
AUDIO_OUTPUT = "recorded_audio.wav"

audio = pyaudio.PyAudio()
recording = True

def record_audio():
    global recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    with wave.open(AUDIO_OUTPUT, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def stop_recording():
    global recording
    recording = False

def convert_audio_to_text():
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(AUDIO_OUTPUT) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            with open("test.txt", "w") as f:
                f.write(text)
    except Exception as e:
        print(f"Audio processing error: {str(e)}")

if __name__ == "__main__":
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    
    input("Press Enter to stop recording...")
    stop_recording()
    recording_thread.join()
    convert_audio_to_text()
    audio.terminate()