from __future__ import print_function

import mido
from mido import MidiFile

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display


# -----------------------------------------------------------------------------
# Step 1: reading midi file into a dataframe
# -----------------------------------------------------------------------------

def midi_to_df(midi_file:str) -> pd.DataFrame:
    """
        Reads a midi file and returns the content as a pd.DataFrame.\n
        Ignores meta messages.

        Args:
            midi_file (str): filepath of the midi file
    """
    mid = MidiFile(midi_file, clip=True)
    ticks_per_beat = mid.ticks_per_beat
    df = pd.DataFrame(None, columns=[
        "type", "channel", "track", "ticks_per_beat", "note", "velocity", 
        "time (delta)", "time (abs)", "time (sec)"])
    for track_num, track in enumerate(mid.tracks):
        time_abs = 0
        for msg in track:
            try:
                time_abs += msg.time
                df = df.append(
                    {
                        "type":           msg.type,
                        "channel":        msg.channel,
                        "track":          track_num,
                        "ticks_per_beat": ticks_per_beat,
                        "note":           msg.note,
                        "velocity":       msg.velocity,
                        "time (delta)":   msg.time,
                        "time (abs)":     time_abs,
                        "time (sec)":     mido.tick2second(tick=time_abs, ticks_per_beat=ticks_per_beat, tempo=500000),
                    },
                    ignore_index=True
                )
            except Exception:
                pass
    return df


# -----------------------------------------------------------------------------
# Step 2: Compute DTW time mappings
# -----------------------------------------------------------------------------

def get_dtw_plot(x_1, x_2, fs, hop_size, wp):
    # -- plot dtw mappings between bot sequences --
    fig = plt.figure(figsize=(16, 8))

    # Plot x_1
    plt.subplot(2, 1, 1)
    librosa.display.waveplot(x_1, sr=fs)
    plt.title('Audio Baseline $X_1$')
    ax1 = plt.gca()

    # Plot x_2
    plt.subplot(2, 1, 2)
    librosa.display.waveplot(x_2, sr=fs)
    plt.title('Audio Warped $X_2$')
    ax2 = plt.gca()

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 30
    points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))

    # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    for tp1, tp2 in wp[points_idx] * hop_size / fs:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                    (coord1[1], coord2[1]),
                                    transform=fig.transFigure,
                                    color='r')
        lines.append(line)

    fig.lines = lines
    plt.tight_layout()
    plt.show()

def get_time_mappings(wav_file_base, wav_file_warp, show_dtw_plot:bool=False) -> pd.DataFrame:
    # -- read data from file --
    x_1, fs = librosa.load(wav_file_base)
    x_2, fs = librosa.load(wav_file_warp)

    # -- define stepsize --
    n_fft = 4410
    hop_size = 2205 # default...          (0.1 sec steps)
    hop_size = 220  # reduced it a lot... (0.01 sec steps)

    # -- create features --
    x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_fft=n_fft)
    x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_fft=n_fft)

    # -- compute DTW --
    D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma)
    wp_s = np.asarray(wp) * hop_size / fs

    # -- place in df --
    df = pd.DataFrame(wp_s, columns=["audio", "midi"])
    df = df.reindex(index=df.index[::-1])
    df.reset_index(inplace=True, drop=True)
    
    # -- show dtw plot --
    if show_dtw_plot is True:
        get_dtw_plot(x_1=x_1, x_2=x_2, fs=fs, hop_size=hop_size, wp=wp)

    # -- return result --
    return df

# -----------------------------------------------------------------------------
# Step 3: Apply DTW time mappings
# -----------------------------------------------------------------------------

def apply_time_mappings(df_midi) -> pd.DataFrame:
    return df_midi


# -----------------------------------------------------------------------------
# Step 4: Write midi
# -----------------------------------------------------------------------------

def write_midi(df_midi:pd.DataFrame, outfile:str) -> None:
    pass


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Step 1: reading midi file into a dataframe
    midi_file = r'C:\Users\User\Documents\GitHub\DTW\Cataldi_Impromptu.mid'
    df_midi = midi_to_df(midi_file=midi_file)
    df_midi

    # Step 2: DTW time mappings
    wav_file_base = r'C:\Users\User\Documents\GitHub\DTW\Cataldi_Impromptu_in_A_Minor_REAL.wav'
    wav_file_warp = r'C:\Users\User\Documents\GitHub\DTW\Cataldi_ImpromptuMIDI.wav'
    df_mappings = get_time_mappings(wav_file_base=wav_file_base, wav_file_warp=wav_file_warp, show_dtw_plot=False)
    df_mappings

    # Step 3: Apply DTW time mappings


    # Step 4: Write midi



