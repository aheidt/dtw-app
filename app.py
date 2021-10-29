import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.collections
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt

from dtw import DTW, MidiIO


class MenuBar(tk.Menu):
    def __init__(self, parent) -> None:
        # -- init class --
        tk.Menu.__init__(self, parent)
        self.app = parent

        # -- init menu bar --
        menubar = tk.Menu(self, tearoff=False)

        # -- create menu (File) -------------------------------------
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.new, accelerator="Ctrl+N")
        filemenu.add_command(label="Restart", command=self.restart, accelerator="Ctrl+R")
        filemenu.add_separator()
        filemenu.add_command(label="Open .wav (Original)", command=self.on_open_wav_original, accelerator="Ctrl+I")
        filemenu.add_command(label="Open .wav (from MIDI)", command=self.on_open_wav_from_midi, accelerator="Ctrl+O")
        filemenu.add_command(label="Open .midi", command=self.on_open_midi, accelerator="Ctrl+M")
        filemenu.add_separator()
        filemenu.add_command(label="Save as .midi", command=self.on_save_midi, accelerator="Ctrl+S")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit_app, accelerator="Ctrl+Q")

        self.app.bind('<Control-n>', self.ctrl_n)
        self.app.bind('<Control-r>', self.ctrl_r)
        self.app.bind('<Control-i>', self.ctrl_i)
        self.app.bind('<Control-o>', self.ctrl_o)
        self.app.bind('<Control-m>', self.ctrl_m)
        self.app.bind('<Control-s>', self.ctrl_s)
        self.app.bind('<Control-q>', self.ctrl_q)

        # -- create menu (DTW) --------------------------------------
        dtwmenu = tk.Menu(menubar, tearoff=0)
        dtwmenu.add_command(label="apply dtw algorithm", command=self.apply_dtw_algo, accelerator="F1")
        dtwmenu.add_separator()
        dtwmenu.add_command(label="show chroma features", command=self.show_chroma_features, accelerator="F2")
        dtwmenu.add_command(label="show dtw mappings", command=self.show_dtw_mappings, accelerator="F3")
        dtwmenu.add_command(label="show remap function", command=self.show_remap_function, accelerator="F4")

        self.app.bind('<F1>', self.f1)
        self.app.bind('<F2>', self.f2)
        self.app.bind('<F3>', self.f3)
        self.app.bind('<F4>', self.f4)

        # -- create menu (View) -------------------------------------
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="zoom in", command=self.zoom_in, accelerator="Ctrl++")
        viewmenu.add_command(label="zoom out", command=self.zoom_out, accelerator="Ctrl+-")
        viewmenu.add_separator()
        viewmenu.add_command(label="scroll right", command=self.scroll_right, accelerator="Ctrl+Right")
        viewmenu.add_command(label="scroll left", command=self.scroll_left, accelerator="Ctrl+Left")

        self.app.bind('<Control-plus>', self.ctrl_plus)
        self.app.bind('<Control-minus>', self.ctrl_minus)
        self.app.bind('<Control-Right>', self.ctrl_right)
        self.app.bind('<Control-Left>', self.ctrl_left)

        # -- create menu (Help) -------------------------------------
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Test Me (1)", command=self.test1)
        helpmenu.add_command(label="Test Me (2)", command=self.test2)
        helpmenu.add_command(label="Help", command=None)

        # -- attach the menus to the bar --
        self.add_cascade(label="File", menu=filemenu)
        self.add_cascade(label="DTW", menu=dtwmenu)
        self.add_cascade(label="View", menu=viewmenu)
        self.add_cascade(label="Help", menu=helpmenu)
    
    # ---------------------------------------------------------------
    # FUNCTIONALITY
    # ---------------------------------------------------------------

    # -- FILE -------------------------------------------------------

    def new(self) -> None:
        self.app.x_min_glob = 0
        self.app.x_max_glob = 60
        self.app.x_lower_bound_glob = 0
        self.app.x_upper_bound_glob = 60
        self.app.frame_1.clear_plot()
        self.app.frame_2.clear_plot()

    def restart(self) -> None:
        self.app.destroy()
        app=App()
        app.mainloop()

    # ---------------------------------

    def on_open_wav_original(self) -> None:
        self.app.data_1.filename = filedialog.askopenfilename(initialdir = "/",title = "Open file",filetypes = (("wav files","*.wav;"),("All files","*.*")))

        try:
            # -- load timeseries --
            self.app.data_1.y_raw, self.app.data_1.fs = librosa.load(self.app.data_1.filename)

            # -- convert time to seconds for x axis--
            x_num_steps = len(self.app.data_1.y_raw)
            time_length = len(self.app.data_1.y_raw) / self.app.data_1.fs
            self.app.data_1.x_raw = [(i/x_num_steps)*time_length for i in range(x_num_steps)]

            # -- update x axis limits & bound --
            self.app.x_min_glob = 0
            if self.app.data_1.x_raw[-1:][0] > self.app.x_max_glob:
                self.app.x_max_glob = self.app.data_1.x_raw[-1:][0]
            if self.app.data_1.x_raw[-1:][0] > self.app.x_upper_bound_glob:
                self.app.x_upper_bound_glob = self.app.data_1.x_raw[-1:][0]
            
            # -- draw graph --
            self.app.frame_1.clear_plot()
            self.app.frame_1.get_plot()

            # -- trigger axis adjustment --
            self.app.reset_bounds()
            self.app.frame_1.reload_axis()
            self.app.frame_2.reload_axis()

        except Exception as e:
            messagebox.showerror("Error Message", "Could not load file: {file_wav_original}".format(file_wav_original=self.app.data_1.filename))
            messagebox.showerror("Python Error", repr(e))

    def on_open_wav_from_midi(self) -> None:
        self.app.data_2.filename = filedialog.askopenfilename(initialdir = "/",title = "Open file",filetypes = (("wav files","*.wav;"),("All files","*.*")))
        
        try:
            # -- load timeseries --
            self.app.data_2.y_raw, self.app.data_2.fs = librosa.load(self.app.data_2.filename)

            # -- convert time to seconds for x axis--
            x_num_steps = len(self.app.data_2.y_raw)
            time_length = len(self.app.data_2.y_raw) / self.app.data_2.fs
            self.app.data_2.x_raw = [(i/x_num_steps)*time_length for i in range(x_num_steps)]

            # -- update x axis limits & bound --
            self.app.x_min_glob = 0
            if self.app.data_2.x_raw[-1:][0] > self.app.x_max_glob:
                self.app.x_max_glob = self.app.data_2.x_raw[-1:][0]
            if self.app.data_2.x_raw[-1:][0] > self.app.x_upper_bound_glob:
                self.app.x_upper_bound_glob = self.app.data_2.x_raw[-1:][0]
            
            # -- draw graph --
            self.app.frame_2.clear_plot()
            self.app.frame_2.get_plot()

            # -- trigger axis adjustment --
            self.app.reset_bounds()
            self.app.frame_1.reload_axis()
            self.app.frame_2.reload_axis()

        except Exception as e:
            messagebox.showinfo("Could not load file: {file_wav_from_midi}".format(file_wav_from_midi=self.app.data_2.filename))
            messagebox.showerror("Internal Error Message", repr(e))
        
        # self.app.frame_2.get_plot()

    def on_open_midi(self) -> None:
        self.app.file_midi = filedialog.askopenfilename(initialdir = "/",title = "Open file",filetypes = (("MIDI files","*.midi;*.MIDI"),("All files","*.*")))
    
        try:
            pass
            # ideas: https://colinwren.medium.com/visualising-midi-files-with-python-b221feacd762
        except Exception as e:
            messagebox.showinfo("Could not load file: {file_midi}".format(file_midi=self.app.file_midi))
            messagebox.showerror("Internal Error Message", repr(e))

        # self.app.frame_3.get_plot()
    
    # ---------------------------------

    def on_save_midi(self) -> None:
        self.app.file_midi_save = filedialog.asksaveasfilename(initialdir = "/",title = "Save as",filetypes = (("MIDI files","*.midi;*.MIDI"),("All files","*.*")))

    # ---------------------------------

    def exit_app(self) -> None:
        self.app.destroy()

    # -- DTW --------------------------------------------------------

    def apply_dtw_algo(self) -> None:
        # Step 2: compute DTW time mappings
        self.app.dtw_obj = DTW(x_raw=self.app.data_1.y_raw, y_raw=self.app.data_2.y_raw, fs=self.app.data_1.fs, df_midi=None)
        print("Computing chroma features...")
        self.app.dtw_obj.compute_chroma_features()
        print("Computing DTW...")
        self.app.dtw_obj.compute_dtw()
        print("Computing remap function...")
        self.app.dtw_obj.compute_remap_function()
        print("Done")

        # Step 4: Apply DTW time mappings
        # Need to write a function: self.app.dtw_obj.compute_remapped_wav_from_midi(f, x_time)
        # print("Remapping midi")
        # self.app.dtw_obj.compute_remapped_midi()

    def show_chroma_features(self) -> None:
        self.app.dtw_obj.plot_chroma_features()

    def show_dtw_mappings(self) -> None:
        self.app.dtw_obj.plot_dtw_mappings()

    def show_remap_function(self) -> None:
        self.app.dtw_obj.plot_remap_function()

    # -- VIEW -------------------------------------------------------

    def zoom_in(self) -> None:
        # -- adjust x axis limits --
        zoom_amount = ((self.app.x_max_glob - self.app.x_min_glob) * 1/3) * 0.5

        self.app.x_min_glob += zoom_amount
        self.app.x_max_glob -= zoom_amount

        # -- trigger axis adjustment --
        self.app.frame_1.reload_axis()
        self.app.frame_2.reload_axis()

    def zoom_out(self) -> None:
        # -- adjust x axis limits --
        zoom_amount = ((self.app.x_max_glob - self.app.x_min_glob) * 0.5) * 0.5

        self.app.x_min_glob -= zoom_amount
        self.app.x_max_glob += zoom_amount

        # -- adjust xlim to avoid going outside the bounds --
        if self.app.x_min_glob < self.app.x_lower_bound_glob and self.app.x_max_glob > self.app.x_upper_bound_glob:
            self.app.x_min_glob = self.app.x_lower_bound_glob
            self.app.x_max_glob = self.app.x_upper_bound_glob
        elif self.app.x_min_glob < self.app.x_lower_bound_glob:
            self.app.x_min_glob = self.app.x_lower_bound_glob
            self.app.x_max_glob += self.app.x_lower_bound_glob - self.app.x_min_glob
        elif self.app.x_max_glob > self.app.x_upper_bound_glob:
            self.app.x_max_glob = self.app.x_upper_bound_glob
            self.app.x_min_glob += self.app.x_upper_bound_glob - self.app.x_max_glob
        else:
            pass

        # -- trigger axis adjustment --
        self.app.frame_1.reload_axis()
        self.app.frame_2.reload_axis()

    def scroll_right(self) -> None:
        # -- adjust x axis limits --
        window_width:Union[int,float] = self.app.x_max_glob - self.app.x_min_glob
        scroll_amount:Union[int,float] = window_width * 0.18

        self.app.x_min_glob += scroll_amount
        self.app.x_max_glob += scroll_amount

        # -- adjust xlim to avoid going outside the bounds --
        if self.app.x_max_glob > self.app.x_upper_bound_glob:
            self.app.x_min_glob = self.app.x_upper_bound_glob - window_width
            self.app.x_max_glob = self.app.x_upper_bound_glob
        else:
            pass

        # -- trigger axis adjustment --
        self.app.frame_1.reload_axis()
        self.app.frame_2.reload_axis()

    def scroll_left(self) -> None:
        # -- adjust x axis limits --
        window_width:Union[int,float] = self.app.x_max_glob - self.app.x_min_glob
        scroll_amount:Union[int,float] = window_width * 0.18

        self.app.x_min_glob -= scroll_amount
        self.app.x_max_glob -= scroll_amount

        # -- adjust xlim to avoid going outside the bounds --
        if self.app.x_min_glob < self.app.x_lower_bound_glob:
            self.app.x_min_glob = self.app.x_lower_bound_glob
            self.app.x_max_glob = self.app.x_lower_bound_glob + window_width
        else:
            pass

        # -- trigger axis adjustment --
        self.app.frame_1.reload_axis()
        self.app.frame_2.reload_axis()

    # -- HELP -------------------------------------------------------

    def test1(self):
        pass

    def test2(self):
        pass

    # ---------------------------------------------------------------
    # HOTKEY EVENTS
    # ---------------------------------------------------------------

    # -- FILE -------------------------------------------------------

    def ctrl_n(self, event) -> None:
        self.new()

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

    # -- DTW --------------------------------------------------------

    def f1(self, event) -> None:
        self.apply_dtw_algo()

    def f2(self, event) -> None:
        self.show_chroma_features()

    def f3(self, event) -> None:
        self.show_dtw_mappings()

    def f4(self, event) -> None:
        self.show_remap_function()

    # -- VIEW -------------------------------------------------------

    def ctrl_plus(self, event) -> None:
        self.zoom_in()

    def ctrl_minus(self, event) -> None:
        self.zoom_out()

    def ctrl_right(self, event) -> None:
        self.scroll_right()

    def ctrl_left(self, event) -> None:
        self.scroll_left()


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


class View1():
    def __init__(self, parent):
        self.app = parent

        self.figure = plt.Figure(figsize=(6,2), dpi=100)
        self.axes = self.figure.add_subplot()

        self.figure.subplots_adjust(left=0.0, bottom=None, right=1.0, top=1.0, wspace=None, hspace=None)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.app.frame_pos_1)
        self.canvas.get_tk_widget().pack(fill='x', side='top')

    def get_plot(self):
        self.axes.plot(self.app.data_1.x_raw, self.app.data_1.y_raw)
        self.axes.set_xlim([self.app.x_min_glob, self.app.x_max_glob])
        self.canvas.draw()
    
    def reload_axis(self):
        self.axes.set_xlim([self.app.x_min_glob, self.app.x_max_glob])
        self.canvas.draw()

    def clear_plot(self):
        self.axes.cla()
        self.canvas.draw()


class View2():
    def __init__(self, parent):
        self.app = parent

        self.figure = plt.Figure(figsize=(6,2), dpi=100)
        self.axes = self.figure.add_subplot()

        self.figure.subplots_adjust(left=0.0, bottom=None, right=1.0, top=1.0, wspace=None, hspace=None)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.app.frame_pos_2)
        self.canvas.get_tk_widget().pack(fill='x', side='top')

    def get_plot(self):
        self.axes.plot(self.app.data_2.x_raw, self.app.data_2.y_raw)
        self.axes.set_xlim([self.app.x_min_glob, self.app.x_max_glob])
        self.canvas.draw()
    
    def reload_axis(self):
        self.axes.set_xlim([self.app.x_min_glob, self.app.x_max_glob])
        self.canvas.draw()

    def clear_plot(self):
        self.axes.cla()
        self.canvas.draw()


class App(tk.Tk):
    def __init__(self) -> None:
        # -- init class --
        tk.Tk.__init__(self)

        # -- window style --
        self.geometry('1200x600')
        self.title("Dynamic Time Warp Tool")

        # -- create master frames
        self.frame_pos_1 = tk.Frame(self)
        self.frame_pos_1.pack(sid="top", fill='x')

        self.frame_pos_2 = tk.Frame(self)
        self.frame_pos_2.pack(sid="top", fill='x')
        
        self.frame_pos_3 = tk.Frame(self)
        self.frame_pos_3.pack(sid="top", fill='x')

        # -- init views --
        self.frame_1 = View1(parent=self)
        self.frame_2 = View2(parent=self)

        # -- init container that holds data for each view --
        self.data_1 = self.data_container()
        self.data_2 = self.data_container()
        self.data_3 = self.data_container()

        # -- init global x axis (in seconds) --
        self.x_min_glob:Union[int,float] = 0
        self.x_max_glob:Union[int,float] = 60

        self.x_lower_bound_glob:Union[int,float] = 0
        self.x_upper_bound_glob:Union[int,float] = 60

        # -- init menubar --
        menubar = MenuBar(self)
        self.config(menu=menubar)

        # -- add editing events --
        # _ = Editing(self)
    
    class data_container():
        def __init__(self) -> None:
            pass

    def reset_bounds(self) -> None:
        try:
            x1 = self.data_1.x_raw[-1:][0]
        except Exception:
            x1 = 0
        
        try:
            x2 = self.data_2.x_raw[-1:][0]
        except Exception:
            x2 = 0

        self.x_min_glob = 0
        self.x_max_glob = max(60, x1, x2)

        self.x_lower_bound_glob = 0
        self.x_upper_bound_glob = max(60, x1, x2)


if __name__ == "__main__":
    # -- init app --
    app=App()

    # -- run app --
    app.mainloop()

