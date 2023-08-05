
import pyaudio
input_device_index = None
audio = pyaudio.PyAudio()
SAMPLE_RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
CHANNELS = 2
FORMAT = pyaudio.paInt16
audio_stream = audio.open(
	input_device_index = input_device_index,
	frames_per_buffer=CHUNK,
	rate=SAMPLE_RATE,
	format=FORMAT,
	channels=CHANNELS,
	input=True,
)
import wave
frames = []

for i in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
	data = audio_stream.read(CHUNK)
	frames.append(data)
print("finished recording")


# stop Recording
audio_stream.stop_stream()
audio_stream.close()

waveFile = wave.open("test.wav", 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(SAMPLE_RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
exit()