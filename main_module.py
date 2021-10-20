import mido
from mido import MidiFile

import pandas as pd
import numpy as np
import math
import os

import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display
import libfmp
import libfmp.c3
import libfmp.c7
# import dtw

from scipy.interpolate import interp1d


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

    # Documentation / Variable names:
    # x1 (np.ndarray): First signal
    # x2 (np.ndarray): Second signal
    # fs (scalar): Sampling rate of WAV files
    # n_fft (int): Window size for computing STFT-based chroma features (Default value = 4410)
    # hop_size (int): Hop size for computing STFT-based chroma features (Default value = 2205)
    # ell (int): Smoothing length for computing CENS features (Default value = 21)
    # d (int): Downsampling factor for computing CENS features (Default value = 5)

    # L_smooth (int): Length of filter for enhancing SM (Default value = 12)
    # tempo_rel_set (np.ndarray): Set of relative tempo values for enhancing SM (Default value = np.array([0.66, 0.81, 1, 1.22, 1.5]))
    # shift_set (np.ndarray): Set of shift indices for enhancing SM (Default value = np.array([0]))
    # strategy (str): Thresholding strategy for thresholding SM ('absolute', 'relative', 'local') (Default value = 'relative')
    # scale (bool): If scale=True, then scaling of positive values to range [0,1] for thresholding SM (Default value = True)
    # thresh (float): Treshold (meaning depends on strategy) (Default value = 0.15)
    # penalty (float): Set values below treshold to value specified (Default value = -2.0)
    # binarize (bool): Binarizes final matrix (positive: 1; otherwise: 0) (Default value = False)
    
    # X (np.ndarray): CENS feature sequence for first signal
    # Y (np.ndarray): CENS feature sequence for second signal
    # Fs_feature (scalar): Feature rate
    # S_thresh (np.ndarray): Similarity matrix
    # I (np.ndarray): Index matrix
    
    # -- read data from file --
    x_1, fs = librosa.load(wav_file_base)
    x_2, fs = librosa.load(wav_file_warp)

    # -- define stepsize --
    n_fft = 4410
    hop_size = 2205
    sigma = np.array([[1, 0], [0, 1], [1, 1]])
    # sigma = np.array([[1, 1], [1, 2], [1, 2]])

    # -- create features --
    X = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_fft=n_fft)
    Y = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2,
                                            hop_length=hop_size, n_fft=n_fft)
    
    # ell = 21
    # d = 5
    # X, Fs_cens = libfmp.c7.compute_cens_from_chromagram(X, ell=21, d=5)
    # Y, Fs_cens = libfmp.c7.compute_cens_from_chromagram(Y, ell=21, d=5)

    N, M = X.shape[1], Y.shape[1]
    K = X.shape[0]
    K; N; M # K = 12, N = 2006, M = 1631 || 12, 402, 327
    
    # C_FMP = libfmp.c3.compute_cost_matrix(X, Y, 'euclidean')

    # -- compute DTW --
    # D, wp = librosa.sequence.dtw(X=X, Y=Y)
    D, wp = librosa.sequence.dtw(X=X, Y=Y, subseq=False, metric="euclidian")
    D, wp = librosa.sequence.dtw(X=X, Y=Y, subseq=False, metric="euclidian")
    # D, wp = librosa.sequence.dtw(X=X, Y=Y, step_sizes_sigma=sigma, subseq=False, metric="euclidian")
    # D, wp = librosa.sequence.dtw(C=C_FMP, step_sizes_sigma=sigma, subseq=False, metric="euclidian")
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

def remap_function(df_mappings, x_colname:str="midi", y_colname:str="audio", show_plot:bool=False):
    """
        Returns a function that remaps any arbitrary point in time to match the audio sequence.

        Args:
            df_mappings (pd.DataFrane): mappings between two sequences (one baseline [audio] and one warped [midi])
            x_colname (str): column name of time that should be warped (midi)
            y_colname (str): column name of time that is used as ground truth (audio)
            show_plot (bool): should the resulting function be shown as a plot?
    """

    # -- drop duplicates --
    df_mappings = df_mappings[[x_colname, y_colname]].drop_duplicates(subset=[x_colname], keep="first")

    # -- define mappings --
    # x:       time (sec)
    # f(x), y: time (sec) remapped
    x = df_mappings[x_colname]
    y = df_mappings[y_colname]

    # -- interpolation methods --
    # f1 = interp1d(x, y, kind='linear')
    # f2 = interp1d(x, y, kind='cubic')
    f3 = interp1d(x, y, fill_value='extrapolate')

    # -- generate plot --
    if show_plot is True:
        plt.plot(x, y, 'o', x, f3(x), '--')
        plt.legend(['data', 'linear'], loc='best')
        plt.show()

    return f3

def apply_time_mappings(df_midi:pd.DataFrame, df_mappings:pd.DataFrame, df_midi_time_colname:str="time (sec)", x_colname_map="midi", y_colname_map="audio", show_remap_plot:bool=False) -> pd.DataFrame:
    """
        Adds a column to the pd.DataFrame with a  warped time for each event.

        Args:
            df_midi (pd.DataFrame): dataset containing midi events
            df_mappings (pd.DataFrame): dataset containing time mappings
    """
    # -- get function to remap time --
    f = remap_function(df_mappings=df_mappings, x_colname=x_colname_map, y_colname=y_colname_map, show_plot=show_remap_plot)

    # -- apply remapping --
    df_midi["time (sec) remapped"] = [f(x) for x in df_midi[df_midi_time_colname]]
    
    # -- return result --
    return df_midi


# -----------------------------------------------------------------------------
# Step 4: Write midi
# -----------------------------------------------------------------------------

def write_midi(df_midi:pd.DataFrame, outfile:str, time_colname:str="time (sec) remapped") -> None:
    """
        Writes a midi file based on a pd.DataFrame containing midi events.

        Args:
            df_midi (pd.DataFrame): needs to contain the following columns: type, channel, note, velocity
            outfile (str): name of the filepath where the midi should be written to
            time_colname (str): name of the column containing the time of the event
        
        Comments:
            currently writes all events into track 0
    """
    # -- global tempo --
    tempo:int = 500000
    ticks_per_beat:int = 96 # set to a higher value to reduce rounding errors

    # -- init midi --
    file = MidiFile()
    track = mido.MidiTrack()
    file.tracks.append(track)
    file.ticks_per_beat = ticks_per_beat

    # -- append note events --
    time_t0:float = 0.0
    for row_id, row in df_midi.iterrows():
        try:
            t:int = round(mido.second2tick(second=row[time_colname]-time_t0, ticks_per_beat=ticks_per_beat, tempo=tempo))
            msg = mido.Message(row['type'], channel=row['channel'], note=row['note'], velocity=row['velocity'], time=t)
            track.append(msg)
            time_t0 = row[time_colname]
        except Exception as e:
            print("Error in df_midi row: {row_id}".format(row_id=row_id))
            raise e

    # -- save file --
    file.save(outfile)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Step 0: define data dir
    try:
        data_dir = os.path(__file__)
    except NameError:
        data_dir = os.path.join(r'C:\Users\User\Documents\GitHub\DTW')

    # Step 1: reading midi file into a dataframe
    midi_file = os.path.join(data_dir, 'Cataldi_Impromptu.mid')
    df_midi = midi_to_df(midi_file=midi_file)

    # Step 2: DTW time mappings
    wav_file_base = os.path.join(data_dir, 'Cataldi_Impromptu_in_A_Minor_REAL.wav')
    wav_file_warp = os.path.join(data_dir, 'Cataldi_ImpromptuMIDI.wav')                 # to fix: push notes to the front of the midi first, so the midi file & midi audio match.
    df_mappings = get_time_mappings(wav_file_base=wav_file_base, wav_file_warp=wav_file_warp, show_dtw_plot=True)

    # Step 3: Apply DTW time mappings
    df_midi = apply_time_mappings(df_midi=df_midi, df_mappings=df_mappings, show_remap_plot=True)

    # Step 4: Write midi
    write_midi(df_midi=df_midi, outfile=r"C:\Users\User\Documents\GitHub\DTW\test.mid", time_colname="time (sec) remapped")

#     raise ParameterError(
# librosa.util.exceptions.ParameterError: scipy.spatial.distance.cdist returned an error.
# Please provide your input in the form X.shape=(K, N) and Y.shape=(K, M).
#  1-dimensional sequences should be reshaped to X.shape=(1, N) and Y.shape=(1, M).