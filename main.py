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

