import wave
import struct
import numpy as np
import matplotlib.pyplot as plt

_SAMPLE_FREQ = 44100
_WRITE_BATCH_SIZE = _SAMPLE_FREQ * 60

FRAME_SIZE_SEC = 0.1
_FRAME_SIZE = int(_SAMPLE_FREQ * FRAME_SIZE_SEC)

MIN_WINDOW_SIZE_IN_FRAMES = 2
NOISE_TRESHOLD = 800

PROCESS_FULL_FILE = True
PLOT = False
SAMPLE_LENGTH_SEC = 15
_SAMPLE_LENGTH = _SAMPLE_FREQ * SAMPLE_LENGTH_SEC

START_OFFSET_IN_SAMPLE_LEN = 3
_START_OFFSET = _SAMPLE_FREQ * (SAMPLE_LENGTH_SEC * START_OFFSET_IN_SAMPLE_LEN)

OUTPUT_WAV = True
INPUT_FILENAME = "chmok.wav"
OUTPUT_FILENAME = "output.wav"


def is_noice_frame(frame):
    out_of_threshold = 0
    in_threshold = 0
    for v in frame:
        if v < NOISE_TRESHOLD:
            in_threshold += 1
        else:
            out_of_threshold += 1
    return out_of_threshold/len(frame) < 0.03


def find_chmoks(channel):
    marks = []
    for frame_number in range(int(len(channel) / _FRAME_SIZE)):
        offset = int(frame_number * _FRAME_SIZE)
        #print(offset)
        #print(is_noice_frame(channel[offset:offset+FRAME_SIZE]))
        marks.append([offset, is_noice_frame(channel[offset:offset + _FRAME_SIZE])])
    return marks

def preprocess_marks(marks):
    patterns = [
        (True, True, False, True, True),
        (True, True, True, False, False, True, True, True),
        (True, True, True, False, False, False, True, True, True),
    ]
    for index in range(len(marks)):
        for pattern in patterns:
            if index + len(pattern) < len(marks):
                found = True
                for v_index, v in enumerate(pattern):
                    if marks[index+v_index][1] != v:
                        found = False
                        break
                if found:
                    for i in range(len(pattern)):
                        marks[index+i][1] = True
    return marks


def process_marks(channel, marks):
    output = []
    marks_in_a_row = 0
    last_value = True
    window_count = 0
    for offset, mark in marks:
        if mark:
            marks_in_a_row += 1
        else:
            if marks_in_a_row >= MIN_WINDOW_SIZE_IN_FRAMES:
                window_count += 1
                output.extend([NOISE_TRESHOLD] * _FRAME_SIZE * marks_in_a_row)
            else:
                output.extend([-NOISE_TRESHOLD] * _FRAME_SIZE * marks_in_a_row)
            marks_in_a_row = 0
            output.extend([-NOISE_TRESHOLD] * _FRAME_SIZE)
        last_value = mark

    print("noice count: %s" % window_count)
    diff = len(channel) - len(output)
    if diff > 0:
        if last_value:
            output.extend([NOISE_TRESHOLD]*diff)
        else:
            output.extend([-NOISE_TRESHOLD] * diff)
    return output


def count_windows(channel, marks):
    marks_in_a_row = 0
    window_count = 0
    for offset, mark in marks:
        if mark:
            marks_in_a_row += 1
        else:
            if marks_in_a_row >= MIN_WINDOW_SIZE_IN_FRAMES:
                window_count += 1
            marks_in_a_row = 0

    return window_count


def filter_channel_with_marks(channel, marks):
    output = []
    marks_in_a_row = 0
    last_value = True
    base_level = 200
    for offset, mark in marks:
        if mark:
            marks_in_a_row += 1
        else:
            if marks_in_a_row >= MIN_WINDOW_SIZE_IN_FRAMES:
                # noise
                for _ in range(_FRAME_SIZE * marks_in_a_row):
                    yield base_level
            else:
                for idx in range(_FRAME_SIZE * marks_in_a_row):
                    yield channel[offset - _FRAME_SIZE + idx]
            marks_in_a_row = 0
            for idx in range(_FRAME_SIZE):
                yield channel[offset+idx]
        last_value = mark

    diff = len(channel) - len(output)
    if diff > 0:
        if last_value:
            for _ in range(diff):
                yield base_level
        else:
            for idx in range(diff):
                yield channel[len(channel)-diff+idx]

def gen_frame_markers(n):
    output = []
    c = 2500
    for i in range(int(n / _FRAME_SIZE)):
        output.extend([c+int(i%2)*1000] * _FRAME_SIZE)
    diff = n - len(output)
    output.extend([c]*diff)
    return output


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wav_obj = wave.open(INPUT_FILENAME, 'rb')
    sample_freq = wav_obj.getframerate()
    #print(sample_freq)
    n_samples = wav_obj.getnframes()
    #print(n_samples)
    t_audio = n_samples / sample_freq
    #print(t_audio)
    signal_wave = wav_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    _START_OFFSET = int(_START_OFFSET)
    t_audio = _SAMPLE_LENGTH / sample_freq
    if not PROCESS_FULL_FILE:
        l_channel = signal_array[0::2][_START_OFFSET:_START_OFFSET + _SAMPLE_LENGTH]
        r_channel = signal_array[1::2][_START_OFFSET:_START_OFFSET + _SAMPLE_LENGTH]
    else:
        l_channel = signal_array[0::2]
        r_channel = signal_array[1::2]

    #print(','.join([str(i) for i in l_channel[START_OFFSET:START_OFFSET + 100]]))
    #print(','.join([str(i) for i in r_channel[START_OFFSET:START_OFFSET + 100]]))
    marks = find_chmoks(l_channel)
    #print(count_windows(l_channel, marks))
    preprocess_marks(marks)

    if PLOT:
        times = np.linspace(0, _SAMPLE_LENGTH / sample_freq, num=_SAMPLE_LENGTH)
        plt.figure(figsize=(15, 5))
        plt.plot(times, l_channel, color="b")
        plt.plot(times, process_marks(channel=l_channel, marks=marks), color="r")
        plt.plot(times, gen_frame_markers(len(l_channel)), color="y")
        plt.title('Left Channel')
        plt.ylabel('Signal Value')
        plt.xlabel('Time (s)')
        plt.xlim(0, t_audio)
        plt.show()

    if OUTPUT_WAV:
        with wave.open(OUTPUT_FILENAME, "w") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(sample_freq)
            fl_channel = filter_channel_with_marks(l_channel, marks)
            fr_channel = filter_channel_with_marks(r_channel, marks)
            total_len = len(l_channel)*2
            percent_step = int(total_len/50)
            write_values = bytearray()
            for idx, samples in enumerate(zip(fl_channel, fr_channel)):
                for sample in samples:
                    sample = int(sample)
                    packed_value = struct.pack("<h", sample)
                    write_values += packed_value
                    if len(write_values) == _WRITE_BATCH_SIZE:
                        f.writeframes(write_values)
                        write_values = bytearray()
                if idx % percent_step == 0:
                    print("%s%% processed" % int((idx/total_len)*100))

            f.writeframes(write_values)

        # with wave.open("sound_original.wav", "w") as f:
        #     f.setnchannels(2)
        #     f.setsampwidth(2)
        #     f.setframerate(sample_freq)
        #     for samples in zip(l_channel, r_channel):
        #         for sample in samples:
        #             sample = int(sample)
        #             f.writeframes(struct.pack("<h", sample))
