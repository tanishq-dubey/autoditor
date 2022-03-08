import moviepy.editor as mp
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import scipy.io.wavfile as wav
from typing import Tuple
from itertools import zip_longest
import argparse
import os
import tempfile

class Moment:
    
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.duration = 0
        if stop is not None:
            self.duration = stop - start
    
    def __str__(self):
        return f"Start {self.start} \t\t Stop {self.stop} \t\t Duration {self.duration}"
    
    def __repr__(self):
        return f"Start {self.start} \t\t Stop {self.stop} \t\t Duration {self.duration}"


def process_audio(source_audio_path: str) -> Tuple[np.ndarray, int, int]:
    rate, data_raw = wav.read(source_audio_path)
    data_raw = data_raw.astype(np.int32)
    mono = (data_raw[:,0] + data_raw[:,1])/2
    duration = len(mono) / rate
    return mono, duration, rate


def convert_video_to_audio(source_video_path: str, destination_audio_location = None) -> str:
    tdir = tempfile.gettempdir()
    dest_location = f"{tdir}/{source_video_path}.wav"
    print(f"checking to see if {dest_location} exists")
    if destination_audio_location is not None:
        dest_location = destination_audio_location
    if os.path.isfile(dest_location):
        print(f"{dest_location} exists, using cached")
        return dest_location
    vid = mp.VideoFileClip(source_video_path)
    vid.audio.write_audiofile(dest_location)
    vid.close()
    return dest_location


def get_subclips(source_video_path: str, moments):
    vid = mp.VideoFileClip(source_video_path)
    clips = []

    for m in moments:
        if m.duration > 30:
            clips.append(vid.subclip(m.start, m.stop))
    return clips


def sub_resample(data: np.ndarray, factor: int):
    return data[::factor].copy()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def find_highlights(data, threshold, rate, factor):
    chunks = []
    for i in range(len(data) - 1):
        if data[i] < threshold < data[i + 1]:
            chunks.append(i * factor / rate)
    return chunks


def find_moving_average_highlights(short_ma, long_ma, bitrate, resample_factor):
    in_a_clip = False
    timestamps = []
    for t in range(1, len(long_ma)):
        if not in_a_clip and (short_ma[t - 1] < long_ma[t - 1]) and (short_ma[t] > long_ma[t]):
            in_a_clip = True
            timestamps.append(t * resample_factor / bitrate)
        elif in_a_clip and (short_ma[t - 1] > long_ma[t - 1]) and (short_ma[t] < long_ma[t]):
            in_a_clip = False
            timestamps.append(t * resample_factor / bitrate)

    ret_list = []
    raw_moments = list(blockwise(timestamps))
    for rm in raw_moments:
        ret_list.append(Moment(rm[0], rm[1]))

    return ret_list


def blockwise(t, size=2, fillvalue=None):
    it = iter(t)
    return zip_longest(*[it]*size, fillvalue=fillvalue)


def plot_audio(data):
    plt.plot(list(range(len(data))), data)
    plt.show()



def main(vidfilepath, outfile, res_factor, lw, sw, dry_run, minduration, maxduration):
    try:
        videofile = vidfilepath

        audiofile = convert_video_to_audio(videofile)
        data, duration, bitrate = process_audio(audiofile)
        RESAMPLE_FACTOR = res_factor
        subsampled_data = sub_resample(data, RESAMPLE_FACTOR)
        squared_subsample = np.square(subsampled_data)
        LONG_WINDOW = lw
        SHORT_WINDOW = sw

        assert(LONG_WINDOW > SHORT_WINDOW)

        long_ma  = moving_average(squared_subsample, LONG_WINDOW)
        short_ma = moving_average(squared_subsample, SHORT_WINDOW)
        moments = find_moving_average_highlights(short_ma, long_ma, bitrate, RESAMPLE_FACTOR)
        total_time = 0
        for m in moments:
            if m.duration > minduration and m.duration < maxduration:
                print(f"Start {round(m.start/60, 2)} \t\t Stop {round(m.stop/60, 2)} \t\t Duration {round(m.duration, 2)}")
                total_time = total_time + m.duration

        print(total_time/60)

        if not dry_run:
            clips = get_subclips(videofile, moments)
            clips
            to_render = mp.concatenate_videoclips(clips)
            to_render.write_videofile(outfile)
        os.remove(audiofile) 
    except:
        os.remove(audiofile) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="autoditor",
        description="autoditor is an automatic video editor."
    )
    parser.add_argument(
        "-v",
        "--video",
        required=True,
        metavar="Video file path",
        dest="vpath"
    )
    parser.add_argument(
        "-f",
        "--factor",
        default=16000,
        metavar="Subsampling factor",
        dest="factor",
        type=int
    )
    parser.add_argument(
        "-l",
        "--longwindow",
        default=128,
        metavar="Long moving average time",
        dest="lwindow",
        type=int
    )
    parser.add_argument(
        "-s",
        "--shortwindow",
        default=64,
        metavar="Short moving average time",
        dest="swindow",
        type=int
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        dest="drun",
        action="store_true"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="Output file location",
        dest="opath"
    )
    parser.add_argument(
        "-i",
        "--minduration",
        default=30,
        metavar="Minimum clip duration",
        dest="mindur",
        type=int
    )
    parser.add_argument(
        "-m",
        "--maxduration",
        default=100,
        metavar="Maximum clip duration",
        dest="maxdur",
        type=int
    )

    args = parser.parse_args()
    # def main(vidfilepath, outfile, res_factor, lw, sw, dry_run):
    main(args.vpath, args.opath, args.factor, args.lwindow, args.swindow, args.drun, args.mindur, args.maxdur)