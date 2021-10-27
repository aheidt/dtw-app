import mido
from mido import MidiFile

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display
import libfmp
import libfmp.c3
import libfmp.c7

from scipy.interpolate import interp1d

import os
from typing import Optional


# -----------------------------------------------------------------------------
# Classes
# -----------------------------------------------------------------------------

class DTW:
    def __init__(self, file_wav_original, file_wav_from_midi, file_midi) -> None:
        # -- load data --
        print(f"Loading {file_wav_original} ...")
        self.x_raw, self.fs = librosa.load(file_wav_original)
        print(f"Loading {file_wav_from_midi} ...")
        self.y_raw, self.fs = librosa.load(file_wav_from_midi)
        print(f"Loading {file_midi} ...")
        self.df_midi:pd.DataFrame = MidiIO.midi_to_df(file_midi=file_midi)

        # -- extracting chroma features --
        print(f"Extracting chroma features ...")
        self.compute_chroma_features()

        # -- dtw params & weights --
        self.hop_size:int = 512
        self.sigma:Optional[np.array] = np.array([[1,1], [3,4], [4,3], [2,3], [3,2], [1,2], [2,1], [1,3], [3,1], [1,4], [4,1]])
        self.weights_add:Optional[list] = [1.0, 1.625, 1.625, 1.8, 1.8, 2.25, 2.25, 2.7, 2.7, 2.875, 2.875]

        print("Done")

    # -- transform --
    
    def compute_chroma_features(self) -> None:
        x_harm = librosa.effects.harmonic(y=self.x_raw, margin=8)
        x_chroma_harm = librosa.feature.chroma_cqt(y=x_harm, sr=self.fs)
        self.x_chroma = np.minimum(
            x_chroma_harm, librosa.decompose.nn_filter(
                x_chroma_harm, aggregate=np.median, metric='cosine'))

        y_harm = librosa.effects.harmonic(y=self.y_raw, margin=8)
        y_chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=self.fs)
        self.y_chroma = np.minimum(
            y_chroma_harm, librosa.decompose.nn_filter(
                y_chroma_harm, aggregate=np.median, metric='cosine'))

    # -- compute --
    
    def compute_dtw(self) -> None:
        # -- create cost matrix --
        C_FMP = libfmp.c3.compute_cost_matrix(self.x_chroma, self.y_chroma, 'euclidean')

        # -- compute DTW --
        self.D, self.wp = librosa.sequence.dtw(C=C_FMP, step_sizes_sigma=self.sigma, metric="euclidian", weights_add=self.weights_add)
        self.wp_s = np.asarray(self.wp) * self.hop_size / self.fs

        # -- place in df --
        self.df_mappings = pd.DataFrame(self.wp_s, columns=["wav_original", "wav_from_midi"])
        self.df_mappings = self.df_mappings.reindex(index=self.df_mappings.index[::-1])
        self.df_mappings.reset_index(inplace=True, drop=True)

    def compute_remap_function(self) -> None:
        """
            Returns a function that remaps any arbitrary point in time from the midi & wav_from_midi to match the original audio sequence.
        """
        # -- drop duplicates --
        df_mappings = self.df_mappings[["wav_from_midi", "wav_original"]].drop_duplicates(subset=["wav_from_midi"], keep="first")

        # -- define mappings --
        # x:       time (sec)
        # f(x), y: time (sec) remapped
        x = df_mappings["wav_from_midi"]
        y = df_mappings["wav_original"]
        x = np.array(x)
        y = np.array(y)

        # -- interpolation methods --
        self.f = interp1d(x, y, fill_value='extrapolate')

    def compute_remapped_midi(self) -> None:
        """
            Adds a column to self.df_midi with a warped/remapped time for each event.
        """
        # -- apply remapping --
        self.df_midi["time (sec) remapped"] = [self.f(x) for x in self.df_midi["time (sec)"]]

    # -- plot graphs --

    def plot_dtw_mappings(self) -> None:
        """
            Plots both original .wav audio sequences below each other.\n
            Draws lines between both sequences to represent the mappings.
        """
        # -- init plot --
        fig = plt.figure(figsize=(16, 8))

        # -- plot self.x_raw --
        plt.subplot(2, 1, 1)
        librosa.display.waveplot(self.x_raw, sr=self.fs)
        plt.title('Audio Baseline $x_raw$')
        ax1 = plt.gca()

        # -- plot self.y_raw --
        plt.subplot(2, 1, 2)
        librosa.display.waveplot(self.y_raw, sr=self.fs)
        plt.title('Audio Warped $y_raw$')
        ax2 = plt.gca()

        plt.tight_layout()

        trans_figure = fig.transFigure.inverted()
        lines = []
        arrows = 30
        points_idx = np.int16(np.round(np.linspace(0, self.wp.shape[0] - 1, arrows)))

        # -- generates list with mappings --
        for tp1, tp2 in self.wp[points_idx] * self.hop_size / self.fs:
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

    def plot_chroma_features(self) -> None:
        """
            Plots the chroma features of .wav sequences below each other.
        """
        plt.figure(figsize=(16, 8))

        plt.subplot(2, 1, 1)
        plt.title('Chroma Representation of $self.x_raw$')
        librosa.display.specshow(self.x_chroma, x_axis='time', y_axis='chroma', hop_length=self.hop_size)
        plt.colorbar()
        
        plt.subplot(2, 1, 2)
        plt.title('Chroma Representation of $self.y_raw$')
        librosa.display.specshow(self.y_chroma, x_axis='time', y_axis='chroma', hop_length=self.hop_size)
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

    def plot_warping_path(self) -> None:
        """
            Shows the warping path on top of the accumulated cost matrix.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        librosa.display.specshow(self.D, x_axis='time', y_axis='time',
                                cmap='gray_r', hop_length=self.hop_size)
        imax = ax.imshow(self.D, cmap=plt.get_cmap('gray_r'),
                        origin='lower', interpolation='nearest', aspect='auto')
        ax.plot(self.wp_s[:, 1], self.wp_s[:, 0], marker='o', color='r')
        plt.title('Warping Path on Acc. Cost Matrix $self.D$')
        plt.colorbar()
        plt.show()

    def plot_remap_function(self) -> None:
        """
            Shows the function that remaps any arbitrary point in time from the midi & wav_from_midi to match the original audio sequence.
        """
        # -- drop duplicates --
        df_mappings = self.df_mappings[["wav_from_midi", "wav_original"]].drop_duplicates(subset=["wav_from_midi"], keep="first")

        # -- define mappings --
        # x:       time (sec)
        # f(x), y: time (sec) remapped
        x = df_mappings["wav_from_midi"]
        y = df_mappings["wav_original"]
        x = np.array(x)
        y = np.array(y)

        # -- generate plot --
        plt.plot(x, y, 'o', x, self.f(x), '--')
        plt.legend(['data', 'linear'], loc='best')
        plt.show()


class MidiIO:
    def __init__(self) -> None:
        pass

    @staticmethod
    def midi_to_df(file_midi:str, start_t0:bool=True) -> pd.DataFrame:
        """
            Reads a midi file and returns the content as a pd.DataFrame.\n
            Ignores meta messages.

            Args:
                file_midi (str): filepath of the midi file
                start_t0 (bool): should the time be shifted so that the midi starts at time 0?
            
            Returns:
                (pd.DataFrame): contains all midi note events inside a dataframe. 
        """
        mid = MidiFile(file_midi, clip=True)
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

    @staticmethod
    def export_midi(df_midi:pd.DataFrame, outfile:str, time_colname:str="time (sec) remapped") -> None:
        """
            Writes a midi file based on a pd.DataFrame containing midi events.

            Args:
                df_midi (pd.DataFrame): needs to contain the following columns: type, channel, note, velocity
                outfile (str): name of the filepath where the midi should be written to
                time_colname (str): name of the column containing the time of the event (in seconds)
            
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
                print("Error in self.df_midi row: {row_id}".format(row_id=row_id))
                raise e

        # -- save file --
        file.save(outfile)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Step 1: file location
    try:
        data_dir = os.path(__file__)
    except NameError:
        data_dir = os.path.join(r'C:\Users\User\Documents\GitHub\DTW')
    finally:
        file_midi          = os.path.join(data_dir, 'midi_files', 'Cataldi_Impromptu.mid')
        file_wav_original  = os.path.join(data_dir, 'wav_files', 'Cataldi_Impromptu_in_A_Minor_REAL3.wav')
        file_wav_from_midi = os.path.join(data_dir, 'wav_files', 'Cataldi_ImpromptuMIDI.wav')                 # to fix: push notes to the front of the midi first, so the midi file & midi audio match.

    # Step 2: compute DTW time mappings
    dtw_obj = DTW(file_wav_original=file_wav_original, file_wav_from_midi=file_wav_from_midi, file_midi=file_midi)
    dtw_obj.compute_dtw()
    dtw_obj.compute_remap_function()

    # Step 3: Inspect some plots
    dtw_obj.plot_chroma_features()
    dtw_obj.plot_dtw_mappings()
    dtw_obj.plot_remap_function()
    dtw_obj.plot_warping_path()

    # Step 4: Apply DTW time mappings
    dtw_obj.compute_remapped_midi()

    # Step 5: Write midi
    outfile = os.path.join(data_dir, 'midi_files', 'test_stepsize1234+_6x_weight_chroma4_newtest.mid')
    MidiIO.export_midi(df_midi=dtw_obj.df_midi, outfile=outfile, time_colname="time (sec) remapped")

