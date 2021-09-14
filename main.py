# https://www.twilio.com/blog/working-with-midi-data-in-python-using-mido
# https://mido.readthedocs.io/en/latest/parsing.html
# https://www.programcreek.com/python/example/90175/mido.MidiFile

# pip install mido==1.2.9
# pip install librosa==0.8.1
import mido
from mido import MidiFile
import librosa

midi_file = r'C:\Users\User\Documents\GitHub\DTW\Duepree08XP.MID'

mid = MidiFile(midi_file, clip=True)
print(mid)

for track in mid.tracks:
    print(track)

for msg in mid.tracks[0][:15]:
    print(msg)


# length in seconds of the entire midi
mid.length

mid.play()
mid.add_track()
mid.filename
mid.print_tracks()
mid.ticks_per_beat


# -----------------------------------------------------------------------------

cv1 = MidiFile(midi_file, clip=True) # clip the velocity of all notes to 127 if they are higher than that.

message_numbers = []
duplicates = []

for track in cv1.tracks:
    if len(track) in message_numbers:
        duplicates.append(track)
    else:
        message_numbers.append(len(track))

for track in duplicates:
    cv1.tracks.remove(track)

midi_file_edited = r'C:\Users\User\Documents\GitHub\DTW\Duepree08XP_edited.MID'
cv1.save(midi_file_edited)


# -----------------------------------------------------------------------------

import os

from mido import MidiFile


cv1 = MidiFile('new_song.mid', clip=True)
cv3 = MidiFile('VampireKillerCV3.mid', clip=True)

del cv1.tracks[4]
del cv1.tracks[4]

cv1.tracks.append(cv3.tracks[4])
cv1.tracks.append(cv3.tracks[5])

cv1.save('mashup.mid')


# -----------------------------------------------------------------------------

import mido

msg = mido.Message('note_on', note=60)
msg = mido.Message('note_on', channel=0, note=16, velocity=32, time=10)


msg.type
# 'note_on'

msg.note
# 60

msg.bytes()
# [144, 60, 64]

msg.copy(channel=2)
# Message('note_on', channel=2, note=60, velocity=64, time=0)

mido.Message.from_bytes(msg.bytes())


# -----------------------------------------------------------------------------
# working example

file = MidiFile()
track = mido.MidiTrack()
file.tracks.append(track)
file.ticks_per_beat = 480

# events = []
# mido.Message.from_bytes(msg.bytes()) # iterate through messages and adjust time

msg = mido.Message('note_on', channel=0, note=30, velocity=32, time=100)
track.append(msg)
msg = mido.Message('note_off', channel=0, note=30, velocity=32, time=500)
track.append(msg)

msg = mido.Message('note_on', channel=0, note=40, velocity=80, time=600)
track.append(msg)
msg = mido.Message('note_off', channel=0, note=40, velocity=80, time=1000)
track.append(msg)

path = r'C:\Users\User\Documents\GitHub\DTW\test.MID'
file.save(path)


# -----------------------------------------------------------------------------

def save_midi(path, pitches, intervals, velocities):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    track = mido.MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(librosa.hz_to_midi(event['pitch'])))
        track.append(mido.Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)


import numpy as np
path = r'C:\Users\User\Documents\GitHub\DTW\test.MID'
pitches = np.ndarray([1,2,3,4], dtype=float)
intervals = [1,2,4,5]
velocities = [80,60,40]
save_midi(path=path, pitches=pitches, intervals=intervals, velocities=velocities)



# def play_midi(song_path):
#     midiports.pending_queue.append(mido.Message('note_on'))
    
#     if song_path in  saving.is_playing_midi.keys():
#         menu.render_message(song_path, "Already playing", 2000)
#         return
    
#     saving.is_playing_midi.clear()
    
#     saving.is_playing_midi[song_path] = True
#     menu.render_message("Playing: ", song_path, 2000)
#     saving.t = threading.currentThread()    

#     output_time_last = 0
#     delay_debt = 0;
#     try:   
#         mid = mido.MidiFile("Songs/"+song_path)
#         fastColorWipe(ledstrip.strip, True)
#         #length = mid.length        
#         t0 = False        
#         for message in mid:
#             if song_path in saving.is_playing_midi.keys():
#                 if(t0 == False):
#                     t0 = time.time()
#                     output_time_start = time.time()            
#                 output_time_last = time.time() - output_time_start
#                 delay_temp = message.time - output_time_last
#                 delay = message.time - output_time_last - float(0.003) + delay_debt
#                 if(delay > 0):
#                     time.sleep(delay)
#                     delay_debt = 0
#                 else:
#                     delay_debt += delay_temp
#                 output_time_start = time.time()                   
            
#                 if not message.is_meta:
#                     midiports.playport.send(message)
#                     midiports.pending_queue.append(message.copy(time=0))
                
#             else:                
#                 break
#         #print('play time: {:.2f} s (expected {:.2f})'.format(
#                 #time.time() - t0, length))
#         #saving.is_playing_midi = False
#     except:
#         menu.render_message(song_path, "Can't play this file", 2000)

# pip install ffmpeg
# pip install PySoundFile

from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import librosa
import librosa.display

# -----------------------------------------------------------------------------

# Visualize

wav_file_true = r'C:\Users\User\Documents\GitHub\DTW\Grieg_Op12_No1_edited_original.wav'
x_1, fs = librosa.load(wav_file_true)
plt.figure(figsize=(16, 4))
librosa.display.waveplot(x_1, sr=fs)
plt.title('Grieg Op.12 No.1 (True)')
plt.tight_layout()
plt.show()

wav_file_false = r'C:\Users\User\Documents\GitHub\DTW\Grieg_Op12_No1_edited_original_false.wav'
x_2, fs = librosa.load(wav_file_false)
plt.figure(figsize=(16, 4))
librosa.display.waveplot(x_2, sr=fs)
plt.title('Grieg Op.12 No.1 (False)')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------

# Extract Chroma Features 
# List of different features:
# (https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html)

n_fft = 4410
hop_size = 2205

x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2,
                                         hop_length=hop_size, n_fft=n_fft)
x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2,
                                         hop_length=hop_size, n_fft=n_fft)

plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.title('Chroma Representation of $X_1$')
librosa.display.specshow(x_1_chroma, x_axis='time',
                         y_axis='chroma', cmap='gray_r', hop_length=hop_size)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.title('Chroma Representation of $X_2$')
librosa.display.specshow(x_2_chroma, x_axis='time',
                         y_axis='chroma', cmap='gray_r', hop_length=hop_size)
plt.colorbar()
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------

D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma)
wp_s = np.asarray(wp) * hop_size / fs

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
librosa.display.specshow(D, x_axis='time', y_axis='time',
                         cmap='gray_r', hop_length=hop_size)
imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
                 origin='lower', interpolation='nearest', aspect='auto')
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
plt.title('Warping Path on Acc. Cost Matrix $D$')
plt.colorbar()
plt.show()

wp_s[0:10]
wp_s[10:20]
wp_s[40:50]
wp_s[140:150]
wp_s[240:250]
wp_s[340:350]
wp_s[450:460]
wp_s[600:610]
wp_s[800:810]
wp_s[900:910]
wp_s[990:1000]

len(wp_s)
len(wp_s)/60
fs
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 8))

# Plot x_1
plt.subplot(2, 1, 1)
librosa.display.waveplot(x_1, sr=fs)
plt.title('Slower Version $X_1$')
ax1 = plt.gca()

# Plot x_2
plt.subplot(2, 1, 2)
librosa.display.waveplot(x_2, sr=fs)
plt.title('Slower Version $X_2$')
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

# -----------------------------------------------------------------------------

# Example from docs (https://librosa.org/doc/main/generated/librosa.sequence.dtw.html)

import numpy as np
import matplotlib.pyplot as plt
y, sr = librosa.load(librosa.ex('brahms'), offset=10, duration=15)
X = librosa.feature.chroma_cens(y=y, sr=sr)
noise = np.random.rand(X.shape[0], 200)
Y = np.concatenate((noise, noise, X, noise), axis=1)
D, wp = librosa.sequence.dtw(X, Y, subseq=True)
fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                               ax=ax[0])
ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
ax[0].legend()
fig.colorbar(img, ax=ax[0])
ax[1].plot(D[-1, :] / wp.shape[0])
ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
          title='Matching cost function')
plt.show()

# -----------------------------------------------------------------------------