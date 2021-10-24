import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt


class MenuBar(tk.Menu):
    def __init__(self, parent) -> None:
        # -- init class --
        tk.Menu.__init__(self, parent)
        self.app = parent

        # -- init menu bar --
        menubar = tk.Menu(self, tearoff=False)

        # -- create menu (File) --
        filemenu = tk.Menu(menubar, tearoff=0)

        filemenu.add_command(label="Restart", command=self.restart, accelerator="Ctrl+R")
        filemenu.add_separator()
        filemenu.add_command(label="Open .wav (Original)", command=self.on_open_wav_original, accelerator="Ctrl+I")
        filemenu.add_command(label="Open .wav (from MIDI)", command=self.on_open_wav_from_midi, accelerator="Ctrl+O")
        filemenu.add_command(label="Open .midi", command=self.on_open_midi, accelerator="Ctrl+M")
        filemenu.add_separator()
        filemenu.add_command(label="Save as .midi", command=self.on_save_midi, accelerator="Ctrl+S")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit_app, accelerator="Ctrl+Q")

        self.app.bind('<Control-r>', self.ctrl_r)
        self.app.bind('<Control-i>', self.ctrl_i)
        self.app.bind('<Control-o>', self.ctrl_o)
        self.app.bind('<Control-m>', self.ctrl_m)
        self.app.bind('<Control-s>', self.ctrl_s)
        self.app.bind('<Control-q>', self.ctrl_q)

        # -- create menu (Help) --
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help", command=None)

        # -- attach the menus to the bar --
        self.add_cascade(label="File", menu=filemenu)
        self.add_cascade(label="Help", menu=helpmenu)
        
    # -- functionality for the menu bar -----------------------------
    def restart(self) -> None:
        self.app.destroy()
        app=App()
        app.mainloop()

    def on_open_wav_original(self) -> None:
        self.app.file_wav_original = filedialog.askopenfilename(initialdir = "/",title = "Open file",filetypes = (("wav files","*.wav;"),("All files","*.*")))
        Views(self.app).get_wav_original_plot()

    def on_open_wav_from_midi(self) -> None:
        self.app.file_wav_from_midi = filedialog.askopenfilename(initialdir = "/",title = "Open file",filetypes = (("wav files","*.wav;"),("All files","*.*")))
        Views(self.app).get_wav_from_midi_plot()

    def on_open_midi(self) -> None:
        self.app.file_midi = filedialog.askopenfilename(initialdir = "/",title = "Open file",filetypes = (("MIDI files","*.midi;*.MIDI"),("All files","*.*")))
        Views(self.app).get_midi_plot()
    
    def on_save_midi(self) -> None:
        self.app.file_midi_save = filedialog.asksaveasfilename(initialdir = "/",title = "Save as",filetypes = (("MIDI files","*.midi;*.MIDI"),("All files","*.*")))

    def exit_app(self) -> None:
        self.app.destroy()

    # -- hotkey events ----------------------------------------------
    def ctrl_r(self, event) -> None:
        self.restart()
    
    def ctrl_i(self, event) -> None:
        self.on_open_wav_original()

    def ctrl_o(self, event) -> None:
        self.on_open_wav_from_midi()

    def ctrl_m(self, event) -> None:
        self.on_open_midi()

    def ctrl_s(self, event) -> None:
        self.on_save_midi()

    def ctrl_q(self, event) -> None:
        self.exit_app()


class Editing():
    def __init__(self, parent) -> None:
        # -- init class --
        self.app = parent

        self.bar_on_hand:bool = False # useful for move_bar_pos_1/move_bar_pos_2, to remember if a bar is currently being dragged.
        self.bar_on_hand_pos1:Optional[Tuple[int,int]] = None
        self.bar_on_hand_pos2:Optional[Tuple[int,int]] = None

        # -- mouse click --
        self.app.bind('<Double-Button-1>', self.insert_bar)     # double click: insert bar
        self.app.bind('<Button-3>', self.delete_bar)            # right click: delete bar

        self.app.bind("<Button 1>", self.move_bar_pos_1)        # left click down: locate bar to move (previous position)
        self.app.bind("<ButtonRelease-1>", self.move_bar_pos_2) # left click up: new location of bar (new position)
        
        # could add events to highlight the bar that is being hovered
        # <Enter>           The mouse pointer entered the widget (this event doesn’t mean that the user pressed the Enter key!).
        # <Leave>           The mouse pointer left the widget.

    # -- click events -----------------------------------------------
    def insert_bar(self, event) -> None:
        """Adds a new bar to the graph."""
        bar_exists:bool = self.bar_exists(proximity=3)
        if bar_exists is True:
            print("A bar already exists at: {x} {y}".format(x=event.x, y=event.y))
        elif bar_exists is False:
            print("A new bar is inserted at: {x} {y}".format(x=event.x, y=event.y))
        else:
            raise ValueError("boolean return expected in 'bar_exists'.")
    
    def delete_bar(self, event) -> None:
        """Deletes a bar from the graph."""
        bar_exists:bool = self.bar_exists(proximity=3)
        if bar_exists is True:
            print("A bar is deleted at: {x} {y}".format(x=event.x, y=event.y))
        elif bar_exists is False:
            print("No bar to delete at: {x} {y}".format(x=event.x, y=event.y))
        else:
            raise ValueError("boolean return expected in 'bar_exists'.")

    def move_bar_pos_1(self, event) -> None:
        bar_exists:bool = self.bar_exists(proximity=3)
        if bar_exists is True:
            print(self.bar_on_hand)
            self.bar_on_hand = True
            print("A bar is moved from: {x} {y}".format(x=event.x, y=event.y))
            print(self.bar_on_hand)
        elif bar_exists is False:
            print("No bar to move from: {x} {y}".format(x=event.x, y=event.y))
        else:
            raise ValueError("boolean return expected in 'bar_exists'.")

    def move_bar_pos_2(self, event) -> None:
        if self.bar_on_hand is True:
            print(self.bar_on_hand)
            self.bar_on_hand = False
            print("A bar is moved to: {x} {y}".format(x=event.x, y=event.y))
            print(self.bar_on_hand)
        elif self.bar_on_hand is False:
            print(self.bar_on_hand)
            print("No bar at hand.")
        else:
            raise ValueError("boolean return expected in 'bar_exists'.")

    # -- edit support -----------------------------------------------
    def bar_exists(self, proximity:int=3) -> bool:
        """
            Checks if a bar already exists within a range of +/- proximity.
        """
        return False


class Views():
    def __init__(self, parent) -> None:
        self.app = parent

    def get_wav_original_plot(self) -> None:
        # -- check if the figure has already been successfully loaded. if yes, don't overwrite it --
        if self.app.figure1_loaded is True:
            messagebox.showinfo("Info", "Could not load the figure, because it was already loaded. Consider restarting the app if you want to replace the sequence.")
            return None
        else:
            pass
        
        if self.app.file_wav_original is not None:
            try:
                y, fs = librosa.load(self.app.file_wav_original)

                x_num_steps = len(y)
                time_length = len(y)/fs
                x = [(i/x_num_steps)*time_length for i in range(x_num_steps)]

                self.app.figure1 = plt.Figure(figsize=(6,2), dpi=100)
                self.app.figure1.add_subplot().plot(x,y)
                self.app.figure1.subplots_adjust(left=0.0, bottom=None, right=1.0, top=1.0, wspace=None, hspace=None)
                # self.app.figure1.tight_layout()
                # self.app.figure1.xlim([0,100])
                self.app.figure1_insert = FigureCanvasTkAgg(self.app.figure1, master=self.app.frame_pos_1)
                self.app.figure1_insert.get_tk_widget().pack(fill='x', side='top') # tags='wav_original_plot'
                self.app.figure1_loaded = True
                
            except Exception as e:
                messagebox.showerror("Error Message", "Could not load file: {file_wav_original}".format(file_wav_original=self.app.file_wav_original))
                messagebox.showerror("Internal Error Message", repr(e))
        else:
            pass

    def get_wav_from_midi_plot(self) -> None:
        # -- check if the figure has already been successfully loaded. if yes, don't overwrite it --
        if self.app.figure2_loaded is True:
            messagebox.showinfo("Info", "Could not load the figure, because it was already loaded. Consider restarting the app if you want to replace the sequence.")
            return None
        else:
            pass

        if self.app.file_wav_from_midi is not None:
            try:
                y, fs = librosa.load(self.app.file_wav_from_midi)

                x_num_steps = len(y)
                time_length = len(y)/fs
                x = [(i/x_num_steps)*time_length for i in range(x_num_steps)]

                figure2 = plt.Figure(figsize=(6,2), dpi=100)
                figure2.add_subplot().plot(x,y)
                figure2.subplots_adjust(left=0.0, bottom=None, right=1.0, top=1.0, wspace=None, hspace=None)
                figure2_insert = FigureCanvasTkAgg(figure2, master=self.app.frame_pos_2)
                figure2_insert.get_tk_widget().pack(fill='x', side='top')
            except Exception as e:
                messagebox.showinfo("Could not load file: {file_wav_from_midi}".format(file_wav_from_midi=self.app.file_wav_from_midi))
                messagebox.showerror("Internal Error Message", repr(e))
        else:
            pass

    def get_midi_plot(self) -> None:
        # -- check if the figure has already been successfully loaded. if yes, don't overwrite it --
        if self.app.figure3_loaded is True:
            messagebox.showinfo("Info", "Could not load the figure, because it was already loaded. Consider restarting the app if you want to replace the sequence.")
            return None
        else:
            pass

        if self.app.file_midi is not None:
            try:
                pass
                # ideas: https://colinwren.medium.com/visualising-midi-files-with-python-b221feacd762
            except Exception as e:
                messagebox.showinfo("Could not load file: {file_midi}".format(file_midi=self.app.file_midi))
                messagebox.showerror("Internal Error Message", repr(e))
        else:
            pass

        # ax1.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])


class App(tk.Tk):
    def __init__(self) -> None:
        # -- init class --
        tk.Tk.__init__(self)

        # -- window style --
        self.geometry('1200x600')
        self.title("Dynamic Time Warp Tool")

        self.frame_pos_1 = tk.Frame(self)
        self.frame_pos_1.pack(sid="top", fill='x')
        self.frame_pos_2 = tk.Frame(self)
        self.frame_pos_2.pack(sid="top", fill='x')
        self.frame_pos_3 = tk.Frame(self)
        self.frame_pos_3.pack(sid="top", fill='x')

        # -- loaded filenames --
        self.file_wav_original = None
        self.file_wav_from_midi = None
        self.file_midi = None

        # -- loaded figures --
        self.figure1_loaded:bool = False
        self.figure2_loaded:bool = False
        self.figure3_loaded:bool = False

        # -- menubar --
        menubar = MenuBar(self)
        self.config(menu=menubar)

        # -- add editing events --
        _ = Editing(self)


if __name__ == "__main__":
    # -- init app --
    app=App()

    # -- run app --
    app.mainloop()

