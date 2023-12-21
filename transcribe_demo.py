#! python3.7

import os
import logging
from typing import Any, Callable
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


from sortedcontainers import SortedDict

f_log_file = None

def re_join(regex_list):
    """Joins a list into non-capturing groups"""
    joined_regex = '|'.join(f'(?:{pattern})' for pattern in regex_list)
    return joined_regex


def consecutive_repeat_truncate(arr, limit):
    current_num = None
    current_count = 0

    arr_new = []

    for num in arr:
        if num == current_num:
            current_count += 1
            if current_count < limit:
                arr_new.append(num)
        else:
            current_num = num
            current_count = 1
            arr_new.append(num)
    
    if isinstance(arr, np.ndarray):
        arr_new = np.array(arr_new)
    
    if isinstance(arr, str):
        arr_new = "".join(arr_new)

    return arr_new


def do_tuples_overlap(tuple1: tuple[Any, Any], tuple2: tuple[Any, Any]):
    # Sort the tuples based on their first element to ensure tuple1[0] <= tuple2[0]
    tuple1, tuple2 = sorted((tuple1, tuple2,))

    # Check for overlap
    if tuple1[1] > tuple2[0]:
        return (tuple2[0], min(tuple1[1], tuple2[1]))
    else:
        return False


def register_data_chunk(
    registered_data_dict: SortedDict,
    start,
    end,
    data,
    override: Callable[[Any, list[float]], bool]
    | bool = lambda data, overlapped: False,
):
    """
    Returns a dict of the overridden data entries, if any.
    If the existed data was not overridden (i.e. data wasn't registered),
      it returns a list of the data entries that over lap it and weren't not overridden
    """
    if isinstance(override, bool):
        if override:
            override = lambda data, overlapped: True
        else:
            override = lambda data, overlapped: False

    overlapped = has_overlapping_intervals(registered_data_dict, start, end)
    if overlapped:
        should_override = override(data, overlapped)
        if f_log_file:
            f_log_file.write(f"should_override={should_override}; {data}; overlapped={overlapped}")
        if not should_override:
            return overlapped

    overlapped_dict = {o: registered_data_dict[o] for o in overlapped}
    for s in overlapped:
        del registered_data_dict[s]
    # Store the data with its start value as the key in the sorted dictionary
    registered_data_dict[start] = (end, data)
    return overlapped_dict


def has_overlapping_intervals(
    registered_data_dict: SortedDict,
    start,
    end,
    get_end=lambda x: x[0],
    all=True,
):
    overlaps = []

    # Find the first key in the sorted set that is greater than or equal to the given start value
    index = registered_data_dict.bisect_left(start)

    index = max(0, index - 1)

    # Iterate through the sorted keys from the found index until we find a data with a starting value greater than end
    for current_start in registered_data_dict.keys()[index:]:
        current_end = get_end(registered_data_dict[current_start])

        # Check if the current interval overlaps with the new data chunk's interval
        if current_start < end and current_end > start:
            if all:
                overlaps.append(current_start)
            else:
                return [current_start]

        if current_start >= end:
            break

        # Move to the next key in the sorted set
        index += 1

    return overlaps


sentence_pattern = r"^([A-Z][^\.!?]*[\.!?])"
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


prepend_punctuations = "\"'“¿([{-"
append_punctuations = "\"'.。,，!！?？:：”)]}、"

prepend_punctuations_set = set(prepend_punctuations)
prepend_punctuations_re = re.compile(f"[{re.escape(prepend_punctuations)}]")

append_punctuations_set = set(append_punctuations)
append_punctuations_re = re.compile(f"[{re.escape(append_punctuations)}]")

punctuations_set = set(prepend_punctuations + append_punctuations)
punctuations_re = re.compile(f"[{re.escape(''.join(punctuations_set))}]")


def remove_punctuation(txt):
    return punctuations_re.sub("", txt)


def needs_dot(text):
    # TODO: test with other symbols, and other languages
    return text[-1] not in append_punctuations_set


def seconds_to_frames(
    seconds: int | float,
    sample_rate: int | float,
    chunk: int | float,
):
    return int(sample_rate / chunk * seconds)


def frames_to_seconds(
    frames: int | float,
    sample_rate: int | float,
    chunk: int | float,
):
    return float((float(chunk) * frames) / sample_rate)


def get_last_segments(completed_segments, seconds_ago, time_now=None):
    if not time_now:
        time_now = time.time()

    index = completed_segments.bisect_left(time_now - seconds_ago)
    index = max(0, index - 1)
    for current_start in completed_segments.keys()[index:]:
        current_end, seg = completed_segments[current_start]
        yield seg


def get_only_updated_segments(completed_segments, segment_updates):
    return {
        k: completed_segments[k]
        for k in set((
            *segment_updates['completed'],
            *segment_updates['updates'],
        ))
    }


def do_it(args):
    task = None
    if args.task:
        task = args.task

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording.
    audio_stream_queue = Queue()
    phrase_complete_queue = Queue()
    transcription_queue = Queue()
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
        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'Microphone with name "{name}" found')
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(name)
                if mic_name in name:
                    source = sr.Microphone(
                        sample_rate=args.sample_rate, device_index=index
                    )
                    input_device_index = index
                    print("found!")
                    print("Device:", name, source.SAMPLE_RATE, source.SAMPLE_WIDTH, source.CHUNK)
                    break
    else:
        source = sr.Microphone(sample_rate=args.sample_rate)

    if source is None:
        raise Exception(f"Microphone not found: {mic_name}")

    print("language", args.language)

    with source:
        recorder.adjust_for_ambient_noise(source)

    # Get the current UTC time
    current_utc_time = datetime.utcnow()
    # Format the UTC time as a string in the desired filename-friendly format
    current_utc_filename_timestamp = current_utc_time.strftime("%Y-%m-%d_%H-%M-%S")

    logfile = f"recorder {current_utc_filename_timestamp}.log"
    f_log_file = open(logfile, "a")

    import wave
    import pyaudio

    audio = pyaudio.PyAudio()

    # affects latency
    record_seconds_interval = args.record_seconds_interval

    SAMPLE_RATE = args.sample_rate
    CHUNK = source.CHUNK
    CHANNELS = source.SAMPLE_WIDTH
    FORMAT = pyaudio.paInt16
    audio_stream = audio.open(
        input_device_index=input_device_index,
        frames_per_buffer=source.CHUNK,
        rate=args.sample_rate,
        format=FORMAT,
        channels=source.SAMPLE_WIDTH,
        input=True,
    )

    """
    Audio segments of `record_seconds_interval` seconds are being captured in real time, and being added to a list.  A second thread analyses this audio and, if there is speech, transcribes and timestamps dialogue.
    As of right now (2023), whisper models accept segments of 30 seconds of audio at once.
    Currently, the thread that captures audio keeps adding onto an audio stream that gets bigger and bigger. The analyzer processes the entire audio (multiple segments of `record_seconds_interval` seconds concatenated), which grows until hitting `args.recorder_max_seconds` (or indefinitely).

    Instead of always sending `args.recorder_max_seconds` always, a point in which audio data can be discarded could be figured out.
    For example, extended periods of silence or noise at the end; however, caution is needed to not discard the sudden beginning and end of speech at the 4 second borderline between audio segments.
    The output of an analysis is empty for an (apparent) non-speech, and the output for when speech is detected has timestamps for the start and end, relative to the audio.
    The timestamps are converted into real global time in the attributes/keys 'start_global' and 'end_global'.
    """

    def audio_recorder_worker(audio_stream_queue, phrase_complete_queue):
        recorder_max_length_frames = 0
        if args.recorder_max_seconds > 0:
            recorder_max_length_frames = seconds_to_frames(
                args.recorder_max_seconds, SAMPLE_RATE, CHUNK
            )
        file_idx = 0
        frames_seek = 0
        seconds_global_recorded = 0
        audio_end_time_global_prev = 0
        frames = []
        while True:
            for i in range(
                0, seconds_to_frames(record_seconds_interval, SAMPLE_RATE, CHUNK)
            ):
                data = audio_stream.read(CHUNK)
                frames.append(data)
                seconds_global_recorded += record_seconds_interval

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
                # if phrase_complete:
                #     # TODO, cut audio until
                #     frame_seek_phrase = get_seek(phrase_complete)
                #     start_cut_old = start_cut
                #     start_cut -= frames_seek - frame_seek_phrase

                #     w = json.dumps(dict(
                #         msg="phrase_complete_queue.get()",
                #         frames_length=len(frames),
                #         phrase_complete=phrase_complete,
                #         start_cut=start_cut,
                #         start_cut_old=start_cut_old,
                #         frames_seek=frames_seek,
                #         frame_seek_phrase=frame_seek_phrase,
                #         frames_seek___frame_seek_phrase=frames_seek - frame_seek_phrase,
                #     ), indent=None,)
                #     f_log_file.write(w + "\n")

                #     if start_cut > 0:
                #         frames = frames[start_cut:]
                #         frames_seek += start_cut

            if recorder_max_length_frames:
                if len(frames) > recorder_max_length_frames:
                    start_cut = len(frames) - recorder_max_length_frames
                    frames = frames[start_cut:]
                    frames_seek += start_cut

            # Use AudioData to convert the raw data to wav data.
            # if not transcribing:
            audio_data = sr.AudioData(
                b"".join(frames), args.sample_rate, source.SAMPLE_WIDTH
            )
            wav_data = audio_data.get_wav_data()
            waveFilename = f"waveFile-{file_idx}.wav"
            waveFilepath = f"tmp/{waveFilename}"
            with wave.open(waveFilepath, "wb") as waveFile:
                waveFile.setnchannels(CHANNELS)
                waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                waveFile.setframerate(SAMPLE_RATE)
                waveFile.writeframes(wav_data)

            # Regardless of the sampling rate used in the original audio file, the audio signal gets resampled to 16kHz (via ffmpeg). So it should work with the recordings you have (likely 44.1 or 48 kHz). If you're creating new recordings and have an option to record in 16 kHz, it may become marginally faster since it can skip resampling and use less space than using a higher sample rate. Although, you'd probably not want to do this for the sake of keeping the recording in a higher audio quality.

            # data_s16 = np.frombuffer(wav_data, dtype=np.int16, count=len(wav_data)//2, offset=0)
            # float_data = data_s16.astype(np.float32, order='C') / 32768.0

            audio_clip = {
                "waveFilename": waveFilename,
                "waveFilepath": waveFilepath,
                "data": None,
                "info": {
                    "length": len(frames),
                    "idx": file_idx,
                    "frames_seek": frames_seek,
                    "duration": duration,
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_timestamp": datetime.utcfromtimestamp(start_time).strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    ),
                    "end_timestamp": datetime.utcfromtimestamp(end_time).strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    ),
                },
            }
            audio_stream_queue.put(audio_clip)
            w = json.dumps(
                dict(
                    msg="audio_stream_queue.put",
                    audio_clip=audio_clip,
                ),
                indent=None,
            )
            f_log_file.write(f"{w}\n")

            if not audio_recorder_alive:
                f_log_file.close()
                break

            file_idx += 1
            file_idx = file_idx % 1000
            audio_end_time_global_prev = end_time

    os.makedirs("tmp", exist_ok=True)

    result_history = []
    audio_clip_history = []

    start_time_global = time.time()

    # Load / Download model
    import torch
    import whisper

    device = args.model_device
    
    token_repeat_limit = 40
    character_repeat_truncate = 20
    log_prob_threshold = -1.1

    transcribe_kwargs = dict(
        # initial_prompt="desu",
        beam_size=args.beam_size,
        best_of=3,
        # beam_size=2,
        length_penalty=1.0,
        # patience=2,
        temperature=[
            0.0,
            # 0.2,
            0.4,
            # 0.6,
        ],
    )
    
    transcribe_f = lambda result: result
    print("device", device)

    model_name = args.model_name
    if model_name != "large" and args.english:
        model_name = model_name + ".en"

    audio_model = None
    
    use_faster_whisper = True
    
    if use_faster_whisper:
        from faster_whisper import WhisperModel
        audio_model = WhisperModel(model_name, device=device, compute_type="float32")

        transcribe_f = lambda result: {'segments': [x._asdict() for x in result[0]], 'info': result[1]._asdict()}

        transcribe_kwargs = dict(
            **transcribe_kwargs,
            repetition_penalty=1.2,
            # token_repeat_limit=token_repeat_limit,
            log_prob_threshold=log_prob_threshold,
        )
    else:
        audio_model = whisper.load_model(model_name)
        # audio_model.to(torch.float16)
        audio_model.to(device)
        
        transcribe_kwargs = dict(
            **transcribe_kwargs,
            fp16=torch.cuda.is_available(),
        )
    
    print("Model loaded.\n")


    completed_segments = SortedDict()

    def find_idx_matching_edges(
        old_text_split,
        new_text_split,
        min_match=99,
        min_match_size=2,
        max_addition=2,
        clean_string=lambda txt: remove_punctuation(txt.lower()),
    ):
        # new text is preferred, but old might have addition in the edges
        # try to find matches and add them to new text

        match_size = None
        match_edge = None
        match_ratio = 0

        shorter_len = min(len(new_text_split), len(old_text_split))

        for os in reversed(
            range(
                max(min_match_size, len(old_text_split) - max_addition), shorter_len + 1
            )
        ):
            edge = "start"
            old_text = " ".join(old_text_split[:os])
            new_text = " ".join(new_text_split[-os:])
            ratio = fuzz.ratio(clean_string(new_text), clean_string(old_text))

            if ratio >= min_match and ratio > match_ratio:
                match_size = os
                match_edge = edge
                match_ratio = ratio

            edge = "end"
            old_text = " ".join(old_text_split[-os:])
            new_text = " ".join(new_text_split[:os])
            ratio = fuzz.ratio(clean_string(new_text), clean_string(old_text))

            if ratio >= min_match and ratio > match_ratio:
                match_size = os
                match_edge = edge
                match_ratio = ratio

            if match_size is not None:
                break

        return (
            match_size,
            match_edge,
            match_ratio,
        )

    def split_match_get_additions(old, new, split_match):
        append_words = []
        prepend_words = []
        if split_match[0]:
            if split_match[1] == "start":
                append_words = old[split_match[0] :]

                if (
                    old[split_match[0] - 1][-1] in append_punctuations_set
                    and new[-1][-1] not in append_punctuations_set
                ):
                    append_text = old[split_match[0] - 1][-1]
                    append_words = [append_text, *append_words]

            if split_match[1] == "end":
                prepend_words = old[: -split_match[0]]

        return prepend_words, append_words

    def override(data, overlapped):
        """
        When a new entry overlaps with existing one,
        check whether it should override it.
        Also checks for matches between both texts,
        to add text from the overridden into the kept entry.
        """
        START_LENIENCE = 0.6
        # End lenience is slightly bigger because transcriptions can end in "..."
        END_LENIENCE = 1.0
        
        competing_is_not_finished = (
            completed_segments[overlapped[-1]][1]["last"]
            and completed_segments[overlapped[-1]][1]["end_silence"] < END_LENIENCE
        )
        if competing_is_not_finished:
            return True

        r = (
            (
                (
                    # Start is not too early in the audio clip
                    (data["start"] > START_LENIENCE)
                    and
                    (
                        # Not last
                        not data["last"]
                        # Or it has enough silence after transcription ("finished")
                        or (data["last"] and data["end_silence"] > END_LENIENCE)
                    )
                )
                or
                # Segment is the entire length of the clip (i.e. started at ~0 and ended at ~full length)
                (
                    # Start is early in the audio clip
                    (data["start"] <= START_LENIENCE)
                    and
                    (
                        # last
                        data["last"]
                        # Or it has NOT enough silence after transcription (not "finished")
                        or (data["last"] and data["end_silence"] <= END_LENIENCE))
                )
            )
        )
        if not r:
            return r

        overlapped_segments = [completed_segments[o][1] for o in overlapped]

        overlapped_text = " ".join((s["text"] for s in overlapped_segments))

        is_data_higher_prob = (
            # Segment is higher prob
            # newer segment prediction has priority (+ 0.10)
            data["avg_logprob"] + 0.10
            > np.average([completed_segments[o][1]["avg_logprob"] for o in overlapped])
        )

        duration_over = (
            overlapped_segments[-1]["end_global"]
            - overlapped_segments[0]["start_global"]
        )
        duration_data = data["end_global"] - data["start_global"]

        if duration_over == duration_data:
            r = is_data_higher_prob
        else:
            r = duration_data > duration_over

        # lenience = 0.4

        # end_longer = (
        #     data['end_global'] -
        #     overlapped_segments[-1]['end_global']
        # )
        # end_longer = max(0, end_longer - lenience)

        # start_longer = (
        #     overlapped_segments[0]['start_global'] -
        #     data['start_global']
        # )
        # start_longer = max(0, start_longer - lenience)

        # This section tries to find words/sections existing in the text that is going to be discarted
        # and add them to the text that is going to be kept

        overlapped_text_split = overlapped_text.split(" ")
        data_text_split = data["text"].split(" ")
        kept = "overlapped"
        new = overlapped_text_split
        old = data_text_split
        if r:
            # Going to override, so
            # Try to add things from previous
            kept = "data"
            new = data_text_split
            old = overlapped_text_split

        split_match = find_idx_matching_edges(old, new)

        if split_match[0]:
            prepend_words, append_words = split_match_get_additions(
                old, new, split_match
            )
            
            f_log_file.write(
                json.dumps(
                    {
                        "kept": kept,
                        "prepend_words": prepend_words,
                        "append_words": append_words,
                        "old": old,
                        "new": new,
                        "split_match": split_match,
                    }
                ) + "\n"
            )

            if kept == "data":
                data["text"] = " ".join([*prepend_words, data["text"], *append_words])
            else:
                overlapped_segments[-1]["text"] = " ".join(
                    [overlapped_segments[-1]["text"], *append_words]
                )

                overlapped_segments[0]["text"] = " ".join(
                    [*prepend_words, overlapped_segments[0]["text"]]
                )

        return r


    def transcription_worker(
    ):
        while True:
            # Pull raw recorded audio from the queue.
            audio_clip = {}
            # get the latest
            while audio_stream_queue.qsize() > 0 or not audio_clip:
                # Queue.get() blocks the thread
                audio_clip = audio_stream_queue.get()
                audio_stream_queue.task_done()

            start_time = time.time()
            transcribing = start_time

            audio_clip_history.append(audio_clip["info"])
            data = (
                audio_clip["data"]
                if (audio_clip["data"])
                else audio_clip["waveFilepath"]
            )
            # print("\t\t\t", f"info={audio_clip['info']}")
            # print("\t\t\t", f"data={data}")

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
                logging.exception(f'Error on transcribe "{data}"')
            
            if not result:
                continue
            
            transcribing = False

            end_time = time.time()
            elapsed_time = end_time - start_time

            print_str = ""
            print_str += (
                f"""\n\t\t\t elapsed_time = {elapsed_time:.3f} seconds;  {audio_clip['waveFilepath']};  audio_stream_queue_size = {audio_stream_queue.qsize()};"""
            ) + "\n"

            # {	'text': 'ないさ ないさ そう 怒るよ 今', 'segments': [
            # {'id': 0, 'seek': 0, 'start': 0.0, 'end': 3.9, 'text': 'ないさ ないさ そう 怒るよ 今', 'tokens': [50364, 9311, 6722, 16647, 1764, 6722, 36165, 220, 3757, 240, 4895, 5591, 220, 6480, 50559], 'temperature': 0.0, 'avg_logprob': -0.4631863236427307, 'compression_ratio': 1.0, 'no_speech_prob': 0.8474944829940796}], 'language': 'ja'}
            # 'start' and others are in seconds
            def cleaned_transcription(
                result,
                filtered=[],
                filtered_re=[
                    re.compile(
                        re_join([
                            # "This video is a derivative work of Touhou Project. It has been made by the same company as Touhou Project",
                            "The following footage is from a work of fiction.",
                            "It contains strong language and adult-like characters.",
                            "Please do not imitate if you are not familiar with this work.",
                            "Viewer discretion is advised.",
                            "Touhou Project",
                            "Thank you for watching",
                            "I'm sorry for the inconvenience.",
                            "Thanks for watching",
                            "I'm sorry, I'm sorry.",
                            "I'm (so )?sorry for the (poor|bad) quality of (this|the|my) videos?.?",
                            "I'm sorry for any inconvenience.",
                            "I'm sorry for the bad sound.",
                            "I'm sorry for the noise.",
                            "I'm sorry for the bad translation.",
                            "I'm sorry for the poor translation.",
                            " 字幕は視聴者によって作成されました。",
                            "Please subscribe",
                            "PLEASE LIKE, COMMENT, and SUBSCRIBE",
                            "Don't forget to like and subscribe",
                            "I hope you enjoy this video",
                            "Please subscribe to my channel",
                            "Hello everyone.? welcome to my channel",
                            "Lyrics by",
                            "Translation by",
                            "Translation by Releska",
                            "Translated by Releska",
                            "Translated by 方 Hou",
                        ]),
                        re.IGNORECASE,
                    ),
                ],
                filtered_logprob_threshold=-0.4,
            ):
                if not "segments" in result:
                    result["segments"] = []
                    return result

                filtered_segments = []
                for s in result["segments"]:
                    should_filter = False
                    for f in filtered_re:
                        if re.search(f, s["text"]):
                            should_filter = True
                            break
                    
                    if should_filter:
                        if s["avg_logprob"] >= filtered_logprob_threshold:
                            should_filter = False

                    if not should_filter:
                        for f in filtered:
                            s["text"] = s["text"].replace(f, "").strip()
                        filtered_segments.append(s)

                result["segments"] = [
                    {**s, "text": consecutive_repeat_truncate(s["text"], character_repeat_truncate)}                    
                    for s in filtered_segments if s["text"].strip(" .!?\u200B")
                ]

                result["text"] = "\n".join((s["text"] for s in result["segments"]))
                return result

            result = cleaned_transcription(result)
            if not result["segments"]:
                continue
            
            result = dict(
                transcribe_info=dict(
                    start_time=start_time,
                    end_time=end_time,
                    start_timestamp=datetime.utcfromtimestamp(start_time).strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    ),
                    end_timestamp=datetime.utcfromtimestamp(end_time).strftime(
                        "%Y-%m-%d_%H-%M-%S"
                    ),
                    elapsed_time=elapsed_time,
                ),
                audio_info=audio_clip["info"],
                **result,
            )

            def timestamp_segments(result):
                for idx in range(len(result["segments"])):
                    s = result["segments"][idx]
                    s["start_global"] = s["start"] + result["audio_info"]["start_time"]
                    s["end_global"] = s["end"] + result["audio_info"]["start_time"]
                    if idx + 1 < len(result["segments"]):
                        s["end_silence"] = (
                            result["segments"][idx + 1]["start"] - s["end"]
                        )
                        s["last"] = False
                    else:
                        s["end_silence"] = result["audio_info"]["duration"] - s["end"]
                        s["last"] = True

                if len(result["segments"]) > 0:
                    result["end_silence"] = (
                        result["audio_info"]["duration"] - result["segments"][-1]["end"]
                    )

                return result

            result = timestamp_segments(result)


            def seg_format(seg):
                return (
                    seg["text"]
                    + "\t\t\t -- "
                    + f" prob={seg['avg_logprob']:0.5f}   {seg['start_global']-start_time_global:0.2f} : {seg['end_global']-start_time_global:0.2f}  {seg['end_silence']:0.2f}  last={seg['last']}"
                )

            if "segments" in result:
                print_str += "\n"
                for s in result["segments"]:
                    print_str += str(s["text"]) + "\n"
            result_history.append(result)

            w = ""
            if isinstance(data, str):
                w += (
                    json.dumps(
                        dict(data=data),
                        indent=None,
                    )
                    + "\n"
                )
            w += (
                json.dumps(
                    dict(result=result),
                    indent=None,
                )
                + "\n"
            )
            f_log_file.write(w)

            audio_max_diff_start = 0.05
            audio_max_diff_end = 0.05
            text_ratio_min = 90

            def is_segment_similar_start(
                seg_a,
                seg_b,
                audio_max_diff_start=audio_max_diff_start,
            ):
                return (
                    abs(seg_a["start_global"] - seg_b["start_global"])
                    < audio_max_diff_start
                )

            def is_segment_similar_end(
                seg_a,
                seg_b,
                audio_max_diff_end=audio_max_diff_end,
            ):
                return (
                    abs(seg_a["end_global"] - seg_b["end_global"]) < audio_max_diff_end
                )

            def is_segment_similar_text(
                seg_a,
                seg_b,
                text_ratio_min=text_ratio_min,
                clean_string=lambda txt: remove_punctuation(txt.lower()),
            ):
                return (
                    fuzz.ratio(clean_string(seg_a["text"]), clean_string(seg_b["text"]))
                    > text_ratio_min
                )

            def is_segment_similar(
                seg_a,
                seg_b,
                audio_max_diff_start=audio_max_diff_start,
                audio_max_diff_end=audio_max_diff_end,
                text_ratio_min=text_ratio_min,
            ):
                return (
                    (
                        not audio_max_diff_start
                        or abs(seg_a["start_global"] - seg_b["start_global"])
                        < audio_max_diff_start
                    )
                    and (
                        not audio_max_diff_end
                        or abs(seg_a["end_global"] - seg_b["end_global"])
                        < audio_max_diff_end
                    )
                    and (
                        fuzz.ratio(seg_a["text"].lower(), seg_b["text"].lower())
                        > text_ratio_min
                    )
                )

            def segment_merge(seg_a, seg_b):
                return {
                    **seg_a,
                    "no_speech_prob": (
                        seg_a["no_speech_prob"] + seg_b["no_speech_prob"]
                    )
                    / 2,
                    "compression_ratio": (
                        seg_a["compression_ratio"] + seg_b["compression_ratio"]
                    )
                    / 2,
                    "avg_logprob": (seg_a["avg_logprob"] + seg_b["avg_logprob"]) / 2,
                    "temperature": (seg_a["temperature"] + seg_b["temperature"]) / 2,
                    "tokens": (seg_a["tokens"] + seg_b["tokens"]),
                    "text": (
                        seg_a["text"]
                        + (". " if needs_dot(seg_a["text"]) else " ")
                        + seg_b["text"]
                    ),
                    "end": seg_b["end"],
                    "end_global": seg_b["end_global"],
                    "merged": (seg_a["merged"] + [seg_b])
                    if "merged" in seg_a
                    else [seg_a, seg_b],
                }

            def get_matching_merge(
                segments,
                i,
                seg_b,
            ):
                ie = i + 1
                seg_m = segments[i]
                idxs = [i]
                seg_m["merged_idxs"] = idxs

                if is_segment_similar_end(
                    seg_m,
                    seg_b,
                ):
                    return seg_m

                while (ie < len(segments)) and seg_m["end_global"] < seg_b[
                    "end_global"
                ]:
                    seg_m = segment_merge(seg_m, segments[ie])
                    idxs.append(ie)
                    if is_segment_similar_end(
                        seg_m,
                        seg_b,
                    ):
                        return seg_m
                    ie += 1
                # Doesn't actually merge to the same ending
                return None

            def register_completed_segment(
                segment,
                override=lambda data, overlapped: False,
            ):
                return register_data_chunk(
                    completed_segments,
                    segment["start_global"],
                    segment["end_global"],
                    segment,
                    override,
                )

            def get_segment_updates(result_history):
                completed = []
                updates = []
                overridden = {}

                def complete_segment(
                    result_idx,
                    seg_idx,
                    override=lambda data, overlapped: False,
                ):
                    segment = result_history[result_idx]["segments"][seg_idx]
                    if segment["avg_logprob"] < log_prob_threshold:
                        f_log_file.write(f"Segment: Too low prob; {seg_format(segment)}" + "\n")
                        return
                    completed_segments_idxs[result_idx][seg_idx] = True
                    segment["complete"] = True

                    # Check if its not already in completed
                    overlapped = register_completed_segment(segment, override)
                    if isinstance(overlapped, dict):
                        # Actually registered
                        if len(overlapped) > 0:
                            updates.append(segment["start_global"])
                            overridden.update(overlapped)
                        else:
                            completed.append(segment["start_global"])
                    else:
                        f_log_file.write(f"Segment: Not registered; {seg_format(segment)}" + "\n")

                if len(result_history) <= 1:
                    return {
                        "completed": completed,
                        "updates": updates,
                        "overridden": overridden,
                        "incomplete_segments": result_history[-1]["segments"],
                    }

                completed_segments_idxs = {
                    -result_idx: {
                        k: False
                        for k in range(len(result_history[-result_idx]["segments"]))
                    }
                    for result_idx in (
                        1,
                        2,
                    )
                }

                match_necessary = False

                if not match_necessary:
                    for base_seg_idx in range(len(result_history[-1]["segments"])):
                        complete_segment(
                            -1,
                            base_seg_idx,
                            override=override,
                        )

                if match_necessary:
                    # Get all segments that are unchanged from last result
                    # they are considered completed
                    i = 0
                    for base_seg_idx in range(len(result_history[-1]["segments"])):
                        # skipping indices that aren't on the last one
                        while (
                            i < len(result_history[-2]["segments"])
                            and (
                                result_history[-2]["segments"][i]["start_global"]
                                < result_history[-1]["segments"][base_seg_idx][
                                    "start_global"
                                ]
                            )
                            and not is_segment_similar_start(
                                result_history[-2]["segments"][i],
                                result_history[-1]["segments"][base_seg_idx],
                            )
                        ):
                            i += 1

                        seg_end = None

                        if i < len(
                            result_history[-2]["segments"]
                        ) and is_segment_similar_start(
                            result_history[-2]["segments"][i],
                            result_history[-1]["segments"][base_seg_idx],
                        ):
                            seg_longer = (-1, base_seg_idx)
                            seg_merger = (-2, i)

                            if (
                                result_history[-2]["segments"][i]["end_global"]
                                > result_history[-1]["segments"][base_seg_idx][
                                    "end_global"
                                ]
                            ):
                                seg_longer = (-2, i)
                                seg_merger = (-1, base_seg_idx)

                            seg_end = get_matching_merge(
                                result_history[seg_merger[0]]["segments"],
                                seg_merger[1],
                                result_history[seg_longer[0]]["segments"][
                                    seg_longer[1]
                                ],
                            )

                            if seg_end:
                                complete_segment(
                                    seg_longer[0],
                                    seg_longer[1],
                                    override=override,
                                )
                                seg_end["complete"] = True
                                for gah in seg_end["merged_idxs"]:
                                    completed_segments_idxs[seg_merger[0]][gah] = True
                                for gah in seg_end["merged_idxs"]:
                                    completed_segments_idxs[seg_longer[0]][
                                        seg_longer[1]
                                    ] = True
                                # if is_segment_similar_text(
                                #     seg_end,
                                #     result_history[seg_longer[0]]['segments'][seg_longer[1]],
                                # ):

                    if len(result_history) > 2:
                        # Get all segments that were completely missed from one result to the other
                        # sliding window slid too much
                        # maybe because processing took a long time

                        # Check for no overlaps between either the segment before or after
                        # Detect this by checking the audio time
                        no_overlap_zone = (
                            # segment before's end
                            result_history[-3]["audio_info"]["end_time"],
                            # segment after's start
                            result_history[-1]["audio_info"]["start_time"],
                        )

                        if no_overlap_zone[0] < no_overlap_zone[1]:
                            # If there is actually a no_overlap_zone
                            # segments in this zone should be completed no matter what

                            def do_tuples_overlap(tuple1, tuple2):
                                # Sort the tuples based on their first element to ensure tuple1[0] <= tuple2[0]
                                tuple1, tuple2 = sorted([tuple1, tuple2])

                                # Check for overlap
                                if tuple1[1] > tuple2[0]:
                                    # Return overlap interval
                                    return (tuple2[0], min(tuple1[1], tuple2[1]))
                                else:
                                    return False

                            def is_segment_in_the_zone(seg):
                                do_tuples_overlap(
                                    no_overlap_zone,
                                    (seg["start_global"], seg["end_global"]),
                                )

                            for i in range(len(result_history[-2]["segments"])):
                                if is_segment_in_the_zone(
                                    result_history[-2]["segments"][i]
                                ):
                                    f_log_file.write(
                                        f"\t\tCompleting segment with no overlap {-2, i,}" + "\n"
                                    )
                                    complete_segment(-2, i, True)

                return {
                    "completed": completed,
                    "updates": updates,
                    "overridden": overridden,
                    "incomplete_segments": [
                        result_history[-1]["segments"][k]
                        for k, v in completed_segments_idxs[-1].items()
                        if not v
                    ],
                }

            segment_updates = get_segment_updates(result_history)

            had_updates = any(segment_updates[k] for k in
                (
                    "completed",
                    "updates",
                    "overridden",
                    "incomplete_segments",
                )
            )

            if had_updates:
                f_log_file.write(
                    f"{json.dumps(dict(phrase_complete=segment_updates), indent=None,)}\n"
                )

                print_str += "\n"
                print_str += "\t\tupdates\n"
                if "updates" in segment_updates:
                    for c in segment_updates["updates"]:
                        print_str += str(seg_format(completed_segments[c][1])) + "\n"

                print_str += str("\t\tcompleted") + "\n"
                if "completed" in segment_updates:
                    for c in segment_updates["completed"]:
                        print_str += str(seg_format(completed_segments[c][1])) + "\n"

                print_str += str("\t\tincomplete_segments") + "\n"
                for seg in segment_updates["incomplete_segments"]:
                    print_str += str(seg_format(seg)) + "\n"

                print_last_x_seconds = 120
                print_str += str("\n") + "\n"
                print_str += str("\t\t latest") + "\n"
                for seg in get_last_segments(completed_segments, print_last_x_seconds):
                    print_str += str(seg_format(seg)) + "\n"
                
                print(print_str, end='', flush=True)

                if args.audio_cut_on_complete_phrase:
                    phrase_complete_queue.put(segment_updates)

                        
            w = ""
            w += (
                json.dumps(
                    dict(segment_updates=segment_updates),
                    indent=None,
                )
                + "\n"
            )
            f_log_file.write(w)
            
            if had_updates:
                transcription_queue.put({
                    "segment_updates": segment_updates,
                    "completed_segments": get_only_updated_segments(completed_segments, segment_updates),
                })

            # If we detected a pause between recordings, add a new item to our transcripion.
            # Otherwise edit the existing one.
            # if phrase_complete:
            #     f_log_file.write(f"{json.dumps(dict(phrase_complete=phrase_complete), indent=None,)}\n")

            #     if 'segments' in phrase_complete:
            #         transcription.pop()
            #         for i in range(len(phrase_complete['segments'])):
            #             transcription.append(args.prefix + phrase_complete['segments'][i]['text'] + args.suffix)
            #         transcription.append("")
            #         if keyboard:
            #             keyboard.type(text)

            #     if args.audio_cut_on_complete_phrase:
            #         phrase_complete_queue.put(phrase_complete)
            # else:
            #     transcription[-1] = args.prefix + text + args.suffix

            # # Clear the console to reprint the updated transcription.
            # os.system('cls' if os.name=='nt' else 'clear')
            # for line in transcription:
            #     print(line, flush=False)
            # # Flush stdout.
            # print('', end='', flush=True)


    audio_recorder_thread = threading.Thread(
        target=audio_recorder_worker,
        args=(
            audio_stream_queue,
            phrase_complete_queue,
        ),
        daemon=True,
    )
    audio_recorder_thread.start()

    transcription_thread = threading.Thread(
        target=transcription_worker,
        daemon=True,
    )
    transcription_thread.start()

    return (
        transcription_queue,
        completed_segments,
        audio_recorder_thread,
        transcription_thread,
    )


def get_argparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="medium",
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument("--english", action="store_true", help="Use the english model.")
    parser.add_argument(
        "--model_device", default="cuda", help="cpu, cuda, xla", type=str
    )
    parser.add_argument(
        "--beam_size", default=1, help="whisper transcription kwarg beam_size", type=int
    )

    parser.add_argument("--recorder_max_seconds", default=10, help="", type=int)
    parser.add_argument(
        "--record_seconds_interval",
        default=1,
        help="Affects latency slightly, the difference between the model predicion time and the next interval hit",
        type=int,
    )
    parser.add_argument(
        "--audio_cut_on_complete_phrase", default=False, help="", type=bool
    )

    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=2,
        help="How real time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=2,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )

    parser.add_argument(
        "--default_microphone",
        "-m",
        default="",
        help="Default microphone name for SpeechRecognition. "
        "Run this with 'list' to view available Microphones.",
        type=str,
    )

    parser.add_argument(
        "--keyboard",
        default=False,
        help="Simulate typing the transcription with your keyboard.",
        type=bool,
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="add prefix to the start of every sentence.",
        type=str,
    )
    parser.add_argument(
        "--suffix", default="", help="add suffix to the end of every sentence", type=str
    )

    parser.add_argument(
        "--sample_rate", default=16000, help="Input sample rate", type=int
    )

    parser.add_argument(
        "--task",
        default=None,
        help="Whisper task (None to transcribe, or 'translate')",
        type=str,
    )
    parser.add_argument("--language", default="en", help="", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_argparse_args()

    do_it(args)

    keyboard = None
    if args.keyboard:
        from pynput.keyboard import Key, Listener, Controller as keyboard_controller

        keyboard = keyboard_controller()

    while True:
        try:
            time.sleep(100)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception(e)
            break


if __name__ == "__main__":
    main()
