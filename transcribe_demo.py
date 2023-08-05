#! python3.7

import os
import logging
from typing import Any
from queue import Queue
import time
from datetime import datetime, timedelta
import json
import re
import threading
import argparse

from tempfile import NamedTemporaryFile

import numpy as np
import speech_recognition as sr
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def do_it(args):
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
    audio_stream_queue = Queue()
    phrase_complete_queue = Queue()
    transcribing = False
    audio_recorder_alive = True
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
    input_device_index = None
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
                    input_device_index = index
                    print("found!")
                    print("Device:", name, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                    break
    else:
        source = sr.Microphone(sample_rate=args.sample_rate)

    if source is None:
        raise Exception(f"Microphone not found: {mic_name}")
    
    print("language", args.language)

    transcription = ['']
    
    with source:
        recorder.adjust_for_ambient_noise(source)


    # Get the current UTC time
    current_utc_time = datetime.utcnow()
    # Format the UTC time as a string in the desired filename-friendly format
    current_utc_filename_timestamp = current_utc_time.strftime('%Y-%m-%d_%H-%M-%S')

    logfile = f'recorder {current_utc_filename_timestamp}.log'
    f_log_file = open(logfile, 'a')

    import wave
    import pyaudio
    audio = pyaudio.PyAudio()
    RECORD_SECONDS = 2
    SAMPLE_RATE = args.sample_rate
    CHUNK = 1024
    CHANNELS = 2
    FORMAT = pyaudio.paInt16
    audio_stream = audio.open(
        input_device_index = input_device_index,
        frames_per_buffer=CHUNK,
        rate=args.sample_rate,
        format=FORMAT,
        channels=CHANNELS,
        input=True,
    )
        

    sentence_pattern = r'^([A-Z][^\.!?]*[\.!?])'
    sentence_pattern_compiled = re.compile(sentence_pattern)
    def check_sentence_match(s1, s2):
        # regex pattern to match complete sentence at start of string
        match1 = sentence_pattern_compiled.match(s1)
        match2 = sentence_pattern_compiled.match(s2)
        if match1 and match2:
            # Both strings have a sentence at the beginning
            return match1.group(0) == match2.group(0)
            # Compare the matched sentences
        # At least one of the strings doesn't have a sentence at the beginning
        return False



    def seconds_to_frames(seconds, sample_rate, chunk,):
        return int(sample_rate / chunk * seconds)

    def frames_to_seconds(frames, sample_rate, chunk,):
        return (float(chunk) * frames) / sample_rate

    # audio segments of 4 seconds are being captured in real time, and being added to a list.  A second thread analyses this audio and, if there is speech, transcribes and timestamps dialogue.
    # Currently, the thread that captures audio keeps adding onto an audio stream that gets bigger and bigger. The analyzer processes the entire audio (multiple segments concatenated), which just grows. So a point in which audio data can be discarded needs to be figured out.
    # For example, extended periods of silence or noise at the start need to be detected and cut out; however, caution is needed to not discard the sudden beginning of speech at the 4 second borderline between audio segments.
    # The output of an analysis is empty for an (apparent) non-speech, and the output for when speech is detected has timestamps for the start and end of the speech in the concatenated segments. How the timestamps translate to the segment indices needs to be handled manually.
    # To take care of the sudden beginning of speech mentioned earlier, I'm only discarding an apparent  non-speech segment if the subsequent segment is either also  non-speech or the subsequent segment has speech detected in the middle, not the start. So an apparent non-speech segment with no next segment is not discarded. 
    def audio_recorder_worker(audio_stream_queue, phrase_complete_queue):
        recorder_maximum_length_frames = 0
        if args.recorder_max_seconds > 0:
            recorder_maximum_length_frames = seconds_to_frames(args.recorder_max_seconds, SAMPLE_RATE, CHUNK)
        file_idx = 0
        frames_seek = 0
        seconds_global_recorded = 0
        audio_end_time_global_prev = 0
        frames = []
        while True:
            for i in range(0, seconds_to_frames(RECORD_SECONDS, SAMPLE_RATE, CHUNK)):
                data = audio_stream.read(CHUNK)
                frames.append(data)
                seconds_global_recorded += RECORD_SECONDS
            
            end_time = time.time()
            duration = frames_to_seconds(len(frames), SAMPLE_RATE, CHUNK)
            start_time = end_time - duration
            elapsed_time = end_time - audio_end_time_global_prev
            overlap = duration - elapsed_time > 0
            

            if len(frames) <= 0:
                continue
            
            start_cut = 0

            if args.audio_cut_on_complete_phrase:
                phrase_complete = None
                while phrase_complete_queue.qsize() > 0:
                    phrase_complete = phrase_complete_queue.get()
                    phrase_complete_queue.task_done()
                if phrase_complete:
                    # phrase_complete = {'start_cut': start_cut, 'final_cut': final_cut, 'ends': ends, 'segments': phrase_info_list[e]['segments'][:ends[e]]}
                    
                    if 'final_cut' in phrase_complete:
                        # phrase_complete['final_cut']
                        start_cut = seconds_to_frames(phrase_complete['final_cut'], SAMPLE_RATE, CHUNK)
                    else:
                        start_cut = phrase_complete['audio_clip']['length']
                    
                    frame_seek_phrase = phrase_complete['audio_clip']['frames_seek']
                    start_cut_old = start_cut
                    start_cut -= frames_seek - frame_seek_phrase
                    
                    w = json.dumps(dict(
                        msg="phrase_complete_queue.get()",
                        frames_length=len(frames),
                        phrase_complete=phrase_complete,
                        start_cut=start_cut,
                        start_cut_old=start_cut_old,
                        frames_seek=frames_seek,
                        frame_seek_phrase=frame_seek_phrase,
                        frames_seek___frame_seek_phrase=frames_seek - frame_seek_phrase,
                    ), indent=None,)
                    f_log_file.write(f"{w}\n")

                    if start_cut > 0:
                        frames = frames[start_cut:]
                        frames_seek += start_cut

            if recorder_maximum_length_frames:
                if len(frames) > recorder_maximum_length_frames:
                    start_cut = len(frames) - recorder_maximum_length_frames
                    frames = frames[start_cut:]
                    frames_seek += start_cut

            # Use AudioData to convert the raw data to wav data.
            # if not transcribing:
            audio_data = sr.AudioData(b''.join(frames), args.sample_rate, source.SAMPLE_WIDTH)
            wav_data = audio_data.get_wav_data()
            waveFilename = f"waveFile-{file_idx}.wav"
            waveFilepath = f"tmp/{waveFilename}"
            with wave.open(waveFilepath, 'wb') as waveFile:
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(SAMPLE_RATE)
                waveFile.writeframes(wav_data)
            
            # Regardless of the sampling rate used in the original audio file, the audio signal gets resampled to 16kHz (via ffmpeg). So it should work with the recordings you have (likely 44.1 or 48 kHz). If you're creating new recordings and have an option to record in 16 kHz, it may become marginally faster since it can skip resampling and use less space than using a higher sample rate. Although, you'd probably not want to do this for the sake of keeping the recording in a higher audio quality.

            # data_s16 = np.frombuffer(wav_data, dtype=np.int16, count=len(wav_data)//2, offset=0)
            # float_data = data_s16.astype(np.float32, order='C') / 32768.0


            audio_clip = {
                'waveFilename': waveFilename,
                'waveFilepath': waveFilepath,
                'info': {
                    'length': len(frames),
                    'idx': file_idx,
                    'frames_seek': frames_seek,
                    
                    'duration': duration,
                    'start_time': str(start_time),
                    'end_time': str(end_time),
                    'start_timestamp': datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S'),
                    'end_timestamp': datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d_%H-%M-%S'),
                }
            }
            audio_stream_queue.put(audio_clip)
            w = json.dumps(dict(
                msg="audio_stream_queue.put",
                audio_clip=audio_clip,
            ), indent=None,)
            f_log_file.write(f"{w}\n")

        
            if not audio_recorder_alive:
                f_log_file.close()
                break

            file_idx += 1
            file_idx = file_idx % 1000
            audio_end_time_global_prev = end_time


    os.makedirs("tmp", exist_ok=True)
    text = ""
    result_history = []
    audio_clip_history = []
    complete_segments = []
    phrase_complete = False

    start_time = time.time()

    # Load / Download model
    import torch
    import whisper
    device = args.model_device
    transcribe_kwargs = dict()
    beam_size = args.beam_size
    transcribe_f = lambda result: result
    print("device", device)

    model_name = args.model_name
    if model_name != "large" and args.english:
        model_name = model_name + ".en"
    

    # audio_model = whisper.load_model(model_name)
    # # audio_model.to(torch.float16)
    # audio_model.to(device)
    # transcribe_kwargs = dict(
    #     fp16=torch.cuda.is_available(),
    # )

    from faster_whisper import WhisperModel
    audio_model = WhisperModel(model_name, device=device, compute_type="float32")
    transcribe_f = lambda result: {'segments': [x._asdict() for x in result[0]], 'info': result[1]}

    print("Model loaded.\n")
    

    transcribe_kwargs = dict(
        # initial_prompt="desu",
        beam_size=beam_size,
        best_of=3,
        # beam_size=2,
        length_penalty=1.0,
        # patience=2,
        temperature=[
            0.0,
            0.2,
            0.4,
            0.6,
            0.7,
            0.8,
        ],
        repetition_penalty=1.2,
        token_repeat_limit=40,
    )
    
    audio_recorder_thread = threading.Thread(
        target=audio_recorder_worker,
        args=(
            audio_stream_queue,
            phrase_complete_queue,
        ),
        daemon=True,
    )
    audio_recorder_thread.start()

    while True:
        try:            
            # if audio_stream_queue.empty():
            #     continue

            # Pull raw recorded audio from the queue.
            audio_clip = {}
            # get the latest
            while audio_stream_queue.qsize() > 0 or not audio_clip:
                audio_clip = audio_stream_queue.get()
                audio_stream_queue.task_done()
            
            start_time = time.time()
            transcribing = start_time

            audio_clip_history.append(audio_clip['info'])
            data = audio_clip['data'] if ('data' in audio_clip and audio_clip['data']) else audio_clip['waveFilepath']
            print("\t\t\t", f"info={audio_clip['info']}")
            print("\t\t\t", f'data={data}')
            
            result: dict[str, Any] = {}
            try:
                kwargs_ = dict(
                    task=task,
                    language=args.language,
                )
                kwargs_.update(transcribe_kwargs)
                result = transcribe_f(
                    audio_model.transcribe(
                        data,
                        **kwargs_,
                    )
                )
            except Exception as e:
                print(f'Error on transcribe "{data}"')
                print(e)
            if not result:
                continue
            transcribing = False

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("\t\t\t", f"elapsed_time = {elapsed_time:.3f} seconds; " f"audio_stream_queue_size = {audio_stream_queue.qsize()};")

            # {	'text': 'ないさ ないさ そう 怒るよ 今', 'segments': [
            # {'id': 0, 'seek': 0, 'start': 0.0, 'end': 3.9, 'text': 'ないさ ないさ そう 怒るよ 今', 'tokens': [50364, 9311, 6722, 16647, 1764, 6722, 36165, 220, 3757, 240, 4895, 5591, 220, 6480, 50559], 'temperature': 0.0, 'avg_logprob': -0.4631863236427307, 'compression_ratio': 1.0, 'no_speech_prob': 0.8474944829940796}], 'language': 'ja'}
            # 'start' and others are in seconds
            def cleaned_transcription(
                result,
                filtered = [
                    "Thank you for watching!",
                    'Thanks for watching!',
                    "I'm sorry, I'm sorry.",
                    "I'm sorry for the poor quality of the video.",
                    "I'm sorry for the bad quality of the video.",
                    "I'm sorry for any inconvenience.",
                    "I'm sorry for any inconvenience.",
                    "I'm sorry for the bad sound.",
                    "I'm sorry for the noise.",
                    "I'm sorry for the bad translation.",
                    "Translation by Releska",
                    "Please subscribe!",
                    "PLEASE LIKE, COMMENT, and SUBSCRIBE!",
                ],
            ):
                if 'segments' in result:
                    result['segments'] = [
                        s for s in result['segments'] if s['text'].strip('.')
                    ]
                    for s in result['segments']:
                        for f in filtered:
                            s['text'] = s['text'].replace(f, "").strip() 
                    
                    result['text'] = "\n".join((s['text'] for s in result['segments']))
                return result
            
            result = cleaned_transcription(result)

            result = dict(
                transcribe_start_time = start_time,
                transcribe_end_time = end_time,
                transcribe_start_timestamp = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S'),
                transcribe_end_timestamp = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d_%H-%M-%S'),
                transcribe_elapsed_time = elapsed_time,
                audio_info=audio_clip['info'],
                **result,
            )

            if 'segments' in result:
                for s in result['segments']:
                    print(s['text'])
            result_history.append(result)

            
            w = ""
            if isinstance(data, str):
                w += f"{json.dumps(dict(data=data), indent=None,)}" + "\n"
            w += f"{json.dumps(dict(result=result), indent=None,)}" + "\n"
            f_log_file.write(w)
            
            def check_phrase_complete(result_history, audio_clip_history):

                def is_segment_similar(
                    seg_a, seg_b,
                    audio_max_diff_start = 0.05,
                    audio_max_diff_end = 0.05,
                    text_ratio_min = 90,
                ):
                    return (
                        (not audio_max_diff_start or
                            abs(seg_a['start'] - seg_b['start']) < audio_max_diff_start) and
                        (not audio_max_diff_end or
                            abs(seg_a['end'] - seg_b['end']) < audio_max_diff_end) and
                        (fuzz.ratio(seg_a['text'].lower(), seg_b['text'].lower()) > text_ratio_min)
                    )
                
                start_cut_idx = 0
                ends = []
                if len(result_history) > 1:
                    i = 0
                    # [0], # <= 1 
                    # [0],[0:1] # > 1, while == i=2, so cut 1:
                    # [0:1],[1:2] # > 1, while == i=2, so cut 1:
                    while i < len(result_history):
                        if len(result_history[i]['segments']) == 0:
                            i += 1
                        else:
                            break
                    
                    i -= 1

                    if i > 0:
                        start_cut_idx = i
                
                    if i + 1 < len(result_history):
                        # at least two non-zero analyzes
                        last_result_segments = result_history[len(result_history) - 1]['segments']
                        if len(last_result_segments) != 0:
                            # find out the unchanged segments from last and prev analyses
                            ends = [0] * len(last_result_segments)
                            found_end = False
                            segment_idx = 0

                            while segment_idx < len(last_result_segments):
                                end_seg_same = len(result_history) - 1
                                while end_seg_same >= 1:
                                    if (
                                        segment_idx < len(result_history[end_seg_same]['segments'])
                                    ) and (
                                        is_segment_similar(
                                            last_result_segments,
                                            result_history[end_seg_same]['segments'][segment_idx],
                                        )
                                    ):
                                        break
                                    else:
                                        end_seg_same -= 1
                                if end_seg_same > 0:
                                    ends[segment_idx] = end_seg_same
                                    found_end = True
                                segment_idx += 1
                            if not found_end:
                                ends = []
                
                
                f_log_file.write(f"""{json.dumps(dict(
                    msg="is_phrase_complete",
                    start_cut_idx=start_cut_idx,
                    ends=ends,
                    segments=[s['segments'] for s in result_history],
                ), indent=None,)} \n""")

                if start_cut_idx or ends:
                    if not ends:
                        phrase_complete = {
                            'audio_clip': audio_clip_history[start_cut_idx],
                        }
                        audio_clip_history = audio_clip_history[start_cut_idx:]
                        result_history = result_history[start_cut_idx:]
                        return phrase_complete

                    final_cut = None
                    e = len(ends)-1
                    while e >= 0:
                        if ends[e] > 0:
                            break
                        e -= 1
                    
                    if e >= 0:
                        if e + 1 <  len(result_history[ends[e]]['segments']):
                            final_cut = result_history[ends[e]]['segments'][e + 1]['start']
                        else:
                            final_cut = result_history[ends[e]]['segments'][e]['end']

                        phrase_complete = {
                            'final_cut': final_cut,
                            'ends': ends,
                            'segments': result_history[ends[e]]['segments'][:e+1],
                            'audio_clip': audio_clip_history[ends[e]],
                        }

                        new_phrase_info_list = result_history[ends[e]:]
                        for k in range(len(new_phrase_info_list)):
                            cut = 0
                            for l in range(len(new_phrase_info_list[0]['segments'])):
                                # fix timestamps
                                # TODO: what the fuck did he mean by  this?
                                new_phrase_info_list[k]['segments'][l]['start'] -= final_cut
                                new_phrase_info_list[k]['segments'][l]['end'] -= final_cut
                                if (new_phrase_info_list[k]['segments'][l]['end'] <= 0 or
                                    new_phrase_info_list[k]['segments'][l]['start'] < 0):
                                    cut = l
                            new_phrase_info_list[0]['segments'][cut:]
                        result_history = new_phrase_info_list
                        audio_clip_history = audio_clip_history[ends[e]:]

                        return phrase_complete


                return {}

            phrase_complete = check_phrase_complete(result_history, audio_clip_history)

            # If we detected a pause between recordings, add a new item to our transcripion.
            # Otherwise edit the existing one.
            if phrase_complete:
                f_log_file.write(f"{json.dumps(dict(phrase_complete=phrase_complete), indent=None,)}\n")

                if 'segments' in phrase_complete:
                    transcription.pop()
                    for i in range(len(phrase_complete['segments'])):
                        transcription.append(args.prefix + phrase_complete['segments'][i]['text'] + args.suffix)
                    transcription.append("")
                    if keyboard:
                        keyboard.type(text)
                
                if args.audio_cut_on_complete_phrase:
                    phrase_complete_queue.put(phrase_complete)
            else:
                transcription[-1] = args.prefix + text + args.suffix

            # # Clear the console to reprint the updated transcription.
            # os.system('cls' if os.name=='nt' else 'clear')
            # for line in transcription:
            #     print(line, flush=False)
            # # Flush stdout.
            # print('', end='', flush=True)

            # Infinite loops are bad for processors, must sleep.
            time.sleep(0.25)
        except KeyboardInterrupt:
            break
        except:
            logging.exception('')
            break

    audio_recorder_alive = False

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

    audio_recorder_thread.join()

def get_argparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--model_device", default='cuda',
                        help="cpu, cuda, xla", type=str)
    parser.add_argument("--beam_size", default=1,
                        help="whisper transcription kwarg beam_size", type=int)

    parser.add_argument("--recorder_max_seconds", default=28,
                        help="", type=int)
    parser.add_argument("--audio_cut_on_complete_phrase", default=False,
                        help="", type=bool)
    
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
    parser.add_argument("--language", default="en",
                        help="", type=str)

    args = parser.parse_args()
    return args

def main():
    args = get_argparse_args()
    do_it(args)


if __name__ == "__main__":
    main()
