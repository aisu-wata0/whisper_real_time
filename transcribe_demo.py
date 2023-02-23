#! python3.7

import argparse
import io
import os
import speech_recognition as sr

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--english", action='store_true',
                        help="Use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=2,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  

    parser.add_argument("--default_microphone", "-m", default='',
                        help="Default microphone name for SpeechRecognition. "
                                "Run this with 'list' to view available Microphones.", type=str)

    parser.add_argument("--keyboard", default=False,
                        help="Simulate typing the transcription with your keyboard.", type=bool)
    
    parser.add_argument("--prefix", default='',
                        help="add prefix to the start of every sentence.", type=str)
    parser.add_argument("--suffix", default='',
                        help="add suffix to the end of every sentence", type=str)
    
    parser.add_argument("--sample_rate", default=16000,
                        help="Input sample rate", type=int)
    
    parser.add_argument("--task", default=None,
                        help="Whisper task (None to transcribe, or 'translate')", type=str)

    args = parser.parse_args()
    
    task = None
    if args.task:
        task = args.task

    keyboard = None
    if args.keyboard:
        from pynput.keyboard import Key, Listener, Controller as keyboard_controller
        keyboard = keyboard_controller()

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    
    # Important for linux users. 
    # Prevents permanent application hang and crash by using the wrong Microphone
    mic_name = args.default_microphone
    source = None
    if mic_name:
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(name)
                if mic_name in name:
                    source = sr.Microphone(sample_rate=args.sample_rate, device_index=index)
                    print("found!")
                    break
    else:
        source = sr.Microphone(sample_rate=args.sample_rate)

    if source is None:
        raise Exception(f"Microphone not found: {mic_name}")
    
    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    # Load / Download model
    import whisper
    import torch
    model = args.model
    if args.model != "large" and args.english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = audio_data.get_wav_data()

                data_s16 = np.frombuffer(wav_data, dtype=np.int16, count=len(wav_data)//2, offset=0)
                float_data = data_s16.astype(np.float32, order='C') / 32768.0

                # Read the transcription.
                result = audio_model.transcribe(float_data, fp16=torch.cuda.is_available(), task=task)
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete and text:
                    text = args.prefix + text + args.suffix
                    transcription.append(text)
                    if keyboard:     
                        keyboard.type(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                # os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line, flush=False)
                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()