# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

# -- tkinter --
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# -- utils --
import numpy as np
import pandas as pd
import os
from typing import List, Optional, Tuple, Union

# -- plotting --
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.collections
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -- dtw --
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d

# -- custom --
from dtw import DTW, MidiIO


# -------------------------------------------------------------------
# APP
# -------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        tk.Tk.__init__(self)

        # -------------------------------------------------
        # DATA (model)
        # -------------------------------------------------

        # -- init constants --
        self.downsampling_factor:int = 50 # only plot every 50th data point from .wav for performance reasons

        # -- init data containers --
        self.data_1 = Data1(self)
        self.data_2 = Data2(self)
        self.data_3 = Data3(self)

        # -------------------------------------------------
        # VIEWS (view)
        # -------------------------------------------------

        # -- window style --
        self.geometry('1200x600')
        self.title("Dynamic Time Warp Tool")

        # -- init constants --
        self.x_min:Union[int,float] = 0 # xlim of current view
        self.x_max:Union[int,float] = 1 # xlim of current view
        self.x_min_glob:Union[int,float] = 0 # lowest xlim that is allowed
        self.x_max_glob:Union[int,float] = 1 # highest xlim that is allowed

        # -- init views --
        self.view_1 = View1(self)
        self.view_2 = View2(self)
        self.view_3 = View3(self)

        # -------------------------------------------------
        # EVENTS (controller)
        # -------------------------------------------------

        # -- init menubar --
        menubar = MenuBar(self)
        self.config(menu=menubar)

        # -- init events --
        self.events_1 = Events1(self)
        self.events_2 = Events2(self)
        self.events_3 = Events3(self)

    # -- MISC ---------------------------------------------

    def reset_bounds(self) -> None:
        try:
            x1 = self.data_1.x[-1:][0]
        except Exception:
            x1 = 0
        
        try:
            x2 = self.data_2.x[-1:][0]
        except Exception:
            x2 = 0
        
        try:
            x3 = self.data_3.df_midi["time abs (sec)"][-1:].item()
        except Exception:
            x3 = 0

        self.x_min = 0
        self.x_max = max(1, x1, x2, x3)

        self.x_min_glob = 0
        self.x_max_glob = max(1, x1, x2, x3)

    def convert_x_pos(self, x) -> Union[int,float]:
        """
            Converts the x position from widget position to axis position.

            Args:
                x (int): x position on the widget
            
            Returns:
                (int, float): x position on the plot
        """
        return ( x / self.winfo_width() ) * ( self.x_max - self.x_min) + self.x_min


# -------------------------------------------------------------------
# MenuBar
# -------------------------------------------------------------------

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
        self.app.x_min = 0
        self.app.x_max = 1
        self.app.x_min_glob = 0
        self.app.x_max_glob = 1
        self.app.view_1.clear_plot()
        self.app.view_2.clear_plot()

    def restart(self) -> None:
        self.app.destroy()
        app=App()
        app.mainloop()

    # ---------------------------------------------------------------

    def on_open_wav_original(self) -> None:
        self.app.data_1.filename = filedialog.askopenfilename(initialdir="/", title="Open file", filetypes=(("wav files","*.wav;"),("All files","*.*")))

        try:
            # -- load wav --
            self.app.data_1.y, self.app.data_1.fs = librosa.load(self.app.data_1.filename)

            # -- convert time to seconds for x axis--
            x_num_steps = len(self.app.data_1.y)
            time_length = len(self.app.data_1.y) / self.app.data_1.fs
            self.app.data_1.x = [(i/x_num_steps)*time_length for i in range(x_num_steps)]

            # -- create reduced sample for plotting --
            self.app.data_1.y_sm = self.app.data_1.y[::self.app.downsampling_factor]
            self.app.data_1.x_sm = self.app.data_1.x[::self.app.downsampling_factor]

            # -- update x axis limits & bound --
            self.app.x_min = 0
            if self.app.data_1.x[-1:][0] > self.app.x_max:
                self.app.x_max = self.app.data_1.x[-1:][0]
            if self.app.data_1.x[-1:][0] > self.app.x_max_glob:
                self.app.x_max_glob = self.app.data_1.x[-1:][0]
            
            # -- draw graph --
            self.app.view_1.clear_plot()
            self.app.view_1.get_plot()

            # -- trigger axis adjustment --
            self.app.reset_bounds()
            self.app.view_1.reload_axis()
            self.app.view_2.reload_axis()
            self.app.view_3.reload_axis()

        except Exception as e:
            messagebox.showerror("Error Message", "Could not load file: {file_wav_original}".format(file_wav_original=self.app.data_1.filename))
            messagebox.showerror("Python Error", repr(e))

    def on_open_wav_from_midi(self) -> None:
        self.app.data_2.filename = filedialog.askopenfilename(initialdir="/", title="Open file", filetypes=(("wav files","*.wav;"),("All files","*.*")))
        
        try:
            # -- load wav --
            self.app.data_2.y, self.app.data_2.fs = librosa.load(self.app.data_2.filename)

            # -- convert time to seconds for x axis --
            x_num_steps = len(self.app.data_2.y)
            time_length = len(self.app.data_2.y) / self.app.data_2.fs
            self.app.data_2.x = [(i/x_num_steps)*time_length for i in range(x_num_steps)]

            # -- create reduced sample for plotting --
            self.app.data_2.y_sm = self.app.data_2.y[::self.app.downsampling_factor]
            self.app.data_2.x_sm = self.app.data_2.x[::self.app.downsampling_factor]

            # -- update x axis limits & bound --
            self.app.x_min = 0
            if self.app.data_2.x[-1:][0] > self.app.x_max:
                self.app.x_max = self.app.data_2.x[-1:][0]
            if self.app.data_2.x[-1:][0] > self.app.x_max_glob:
                self.app.x_max_glob = self.app.data_2.x[-1:][0]
            
            # -- draw graph --
            self.app.view_2.clear_plot()
            self.app.view_2.get_plot()

            # -- trigger axis adjustment --
            self.app.reset_bounds()
            self.app.view_1.reload_axis()
            self.app.view_2.reload_axis()
            self.app.view_3.reload_axis()

        except Exception as e:
            messagebox.showinfo("Could not load file: {file_wav_from_midi}".format(file_wav_from_midi=self.app.data_2.filename))
            messagebox.showerror("Internal Error Message", repr(e))
        
    def on_open_midi(self) -> None:
        self.app.data_3.filename = filedialog.askopenfilename(initialdir="/", title="Open file", filetypes=(("MIDI files","*.mid;*.MID"),("All files","*.*")))

        try:
            # -- load midi --
            self.app.data_3.df_midi = MidiIO.midi_to_df(file_midi=self.app.data_3.filename, clip_t0=False)

            # -- update x axis limits & bound --
            self.app.x_min = 0
            if self.app.data_3.df_midi["time abs (sec)"][-1:].item() > self.app.x_max:
                self.app.x_max = self.app.data_3.df_midi["time abs (sec)"][-1:].item()
            if self.app.data_3.df_midi["time abs (sec)"][-1:].item() > self.app.x_max_glob:
                self.app.x_max_glob = self.app.data_3.df_midi["time abs (sec)"][-1:].item()
            
            # -- draw graph --
            self.app.view_3.clear_plot()
            self.app.view_3.get_plot()

            # -- trigger axis adjustment --
            self.app.reset_bounds()
            self.app.view_1.reload_axis()
            self.app.view_2.reload_axis()
            self.app.view_3.reload_axis()

        except Exception as e:
            messagebox.showinfo("Could not load file: {file_midi}".format(file_midi=self.app.data_3.filename))
            messagebox.showerror("Internal Error Message", repr(e))
    
    # ---------------------------------------------------------------

    def on_save_midi(self) -> None:
        self.app.data_3.filename_out = filedialog.asksaveasfilename(initialdir="/", title="Save as", filetypes=(("MIDI files","*.mid;*.MID"),("All files","*.*")))
        print(f"Saving midi file to: {self.app.data_3.filename_out}")
        # to do: actually saving the midi file

    # ---------------------------------------------------------------

    def exit_app(self) -> None:
        self.app.destroy()

    # -- DTW --------------------------------------------------------

    def apply_dtw_algo(self) -> None:
        # -- compute DTW time mappings --
        self.app.dtw_obj = DTW(x_raw=self.app.data_1.y_raw, y_raw=self.app.views.view_2.data.y, fs=self.app.data_1.fs, df_midi=None)
        print("Computing chroma features...")
        self.app.dtw_obj.compute_chroma_features()
        print("Computing DTW...")
        self.app.dtw_obj.compute_dtw()
        print("Computing remap function...")
        self.app.dtw_obj.compute_remap_function()

        # -- apply DTW time mappings --
        print("Remapping midi...")
        self.app.data_3.df_midi["time abs (sec)"] = [self.app.dtw_obj.f(x) for x in self.app.data_3.df_midi["time abs (sec)"]]
        print("Remapping wav...")
        self.app.data_2.x_sm = [self.app.dtw_obj.f(x) for x in self.app.data_2.x_sm]

        # -- draw graphs --
        print("Redrawing graphs")
        self.app.view_2.clear_plot()
        self.app.view_2.get_plot()

        self.app.view_3.clear_plot()
        self.app.view_3.get_plot()

        # -- trigger axis adjustment --
        print("Resetting axis")
        self.app.reset_bounds()
        self.app.view_1.reload_axis()
        self.app.view_2.reload_axis()
        self.app.view_3.reload_axis()

        print("Done")

    def show_chroma_features(self) -> None:
        self.app.dtw_obj.plot_chroma_features()

    def show_dtw_mappings(self) -> None:
        self.app.dtw_obj.plot_dtw_mappings()

    def show_remap_function(self) -> None:
        self.app.dtw_obj.plot_remap_function()

    # -- VIEW -------------------------------------------------------

    def zoom_in(self) -> None:
        # -- adjust x axis limits --
        zoom_amount = ((self.app.x_max - self.app.x_min) * 1/3) * 0.5

        self.app.x_min += zoom_amount
        self.app.x_max -= zoom_amount

        # -- trigger axis adjustment --
        self.app.view_1.reload_axis()
        self.app.view_2.reload_axis()
        self.app.view_3.reload_axis()

    def zoom_out(self) -> None:
        # -- adjust x axis limits --
        zoom_amount = ((self.app.x_max - self.app.x_min) * 0.5) * 0.5

        self.app.x_min -= zoom_amount
        self.app.x_max += zoom_amount

        # -- adjust xlim to avoid going outside the bounds --
        if self.app.x_min < self.app.x_min_glob and self.app.x_max > self.app.x_max_glob:
            self.app.x_min = self.app.x_min_glob
            self.app.x_max = self.app.x_max_glob
        elif self.app.x_min < self.app.x_min_glob:
            self.app.x_min = self.app.x_min_glob
            self.app.x_max += self.app.x_min_glob - self.app.x_min
        elif self.app.x_max > self.app.x_max_glob:
            self.app.x_max = self.app.x_max_glob
            self.app.x_min += self.app.x_max_glob - self.app.x_max
        else:
            pass

        # -- trigger axis adjustment --
        self.app.view_1.reload_axis()
        self.app.view_2.reload_axis()
        self.app.view_3.reload_axis()

    def scroll_right(self) -> None:
        # -- adjust x axis limits --
        window_width:Union[int,float] = self.app.x_max - self.app.x_min
        scroll_amount:Union[int,float] = window_width * 0.18

        self.app.x_min += scroll_amount
        self.app.x_max += scroll_amount

        # -- adjust xlim to avoid going outside the bounds --
        if self.app.x_max > self.app.x_max_glob:
            self.app.x_min = self.app.x_max_glob - window_width
            self.app.x_max = self.app.x_max_glob
        else:
            pass

        # -- trigger axis adjustment --
        self.app.view_1.reload_axis()
        self.app.view_2.reload_axis()
        self.app.view_3.reload_axis()

    def scroll_left(self) -> None:
        # -- adjust x axis limits --
        window_width:Union[int,float] = self.app.x_max - self.app.x_min
        scroll_amount:Union[int,float] = window_width * 0.18

        self.app.x_min -= scroll_amount
        self.app.x_max -= scroll_amount

        # -- adjust xlim to avoid going outside the bounds --
        if self.app.x_min < self.app.x_min_glob:
            self.app.x_min = self.app.x_min_glob
            self.app.x_max = self.app.x_min_glob + window_width
        else:
            pass

        # -- trigger axis adjustment --
        self.app.view_1.reload_axis()
        self.app.view_2.reload_axis()
        self.app.view_3.reload_axis()

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


# -------------------------------------------------------------------
# Data (model)
# -------------------------------------------------------------------

class Data1():
    def __init__(self, parent) -> None:
        # -- init parent --
        self.app = parent

        # -- data source --
        self.filename = ""

        # -- init dataset --
        self.x = np.arange(0, 1.1, 0.1).round(2).tolist()
        self.y = np.arange(0, 1.1, 0.1).round(2).tolist()
        self.x_sm = np.arange(0, 1.1, 0.1).round(2).tolist() # small / reduced dataset
        self.y_sm = np.arange(0, 1.1, 0.1).round(2).tolist() # small / reduced dataset
        self.fs = None

        # -- edit memory --
        self.bars:List[Optional[Union[int,float]]] = []


class Data2():
    def __init__(self, parent) -> None:
        # -- init parent --
        self.app = parent

        # -- data source --
        self.filename = ""

        # -- init dataset --
        self.x = np.arange(0, 1.1, 0.1).round(2).tolist()
        self.y = np.arange(0, 1.1, 0.1).round(2).tolist()
        self.x_sm = np.arange(0, 1.1, 0.1).round(2).tolist() # small / reduced dataset
        self.y_sm = np.arange(0, 1.1, 0.1).round(2).tolist() # small / reduced dataset
        self.fs = None

        # -- edit memory --
        self.bars:List[Optional[Union[int,float]]] = []


class Data3():
    def __init__(self, parent) -> None:
        # -- init parent --
        self.app = parent

        # -- data source --
        self.filename = ""

        # -- init dataset --
        self.df_midi:pd.DataFrame = pd.DataFrame()

        # -- edit memory --
        self.bars:List[Optional[Union[int,float]]] = []


# -------------------------------------------------------------------
# Events (controller)
# -------------------------------------------------------------------

class Events1():
    def __init__(self, parent) -> None:
        # -- init class --
        self.app = parent

        # -- click coordinate memory --
        self.button_1_down_coord:Tuple[int, int] = (None, None)
        self.button_1_up_coord:Tuple[int, int] = (None, None)

        self.button_3_down_coord:Tuple[int, int] = (None, None)
        self.button_3_up_coord:Tuple[int, int] = (None, None)

        # -- mouse click --
        self.app.view_1.canvas.get_tk_widget().bind("<Button 1>", self.record_button_1_down)      # left mouse click (down)
        self.app.view_1.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.record_button_1_up) # left mouse click (up)

        self.app.view_1.canvas.get_tk_widget().bind('<Button-3>', self.record_button_3_down)      # right mouse click (down)
        self.app.view_1.canvas.get_tk_widget().bind('<ButtonRelease-3>', self.record_button_3_up) # right mouse click (up)

        # -- hover canvas (hover bar would be nice too...) --
        self.app.view_1.canvas.get_tk_widget().bind("<Enter>", self.hover_canvas_in)  # mouse pointer entered the widget
        self.app.view_1.canvas.get_tk_widget().bind("<Leave>", self.hover_canvas_out) # mouse pointer left the widget

    # -- click events -----------------------------------------------

    # -- LEFT MOUSE CLICK --

    def record_button_1_down(self, event) -> None:
        self.button_1_down_coord:Tuple[int, int] = (event.x, event.y)

    def record_button_1_up(self, event) -> None:
        self.button_1_up_coord:Tuple[int, int] = (event.x, event.y)

        # -- create bar --
        if self.button_1_down_coord == self.button_1_up_coord:
            x_pos = self.app.convert_x_pos(self.button_1_up_coord[0])
            bar_already_exists:bool = self.app.view_1.bar_exists(x=x_pos)
            if bar_already_exists is True:
                print(f"view_1: A bar already exists at: {event.x} {event.y} | {x_pos}")
            else:
                self.app.view_1.insert_bar(x_pos)
                print(f"view_1: A new bar was inserted at: {event.x} {event.y} | {x_pos}")

        # -- move bar --
        else:
            x_from = self.app.convert_x_pos(self.button_1_down_coord[0])
            x_to   = self.app.convert_x_pos(self.button_1_up_coord[0])
            x_closest_bar = self.app.view_1.get_closest_bar(x_from)
            if x_closest_bar is not None:
                if self.app.view_1.validate_new_bar_pos(x_from=x_closest_bar, x_to=x_to) is True:
                    self.app.view_1.move_bar(x_from=x_closest_bar, x_to=x_to)
                    print("view_1: A bar was moved from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to}".format(
                        x0=self.button_1_down_coord[0], y0=self.button_1_down_coord[1],
                        x1=self.button_1_up_coord[0], y1=self.button_1_up_coord[1],
                        x_from=x_from, x_to=x_to
                        )
                    )
                else:
                    print("view_1: A bar could not be moved from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to} (can't cross other bars)".format(
                        x0=self.button_1_down_coord[0], y0=self.button_1_down_coord[1],
                        x1=self.button_1_up_coord[0], y1=self.button_1_up_coord[1],
                        x_from=x_from, x_to=x_to
                        )
                    )
            else:
                print("view_1: No bar to move from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to}".format(
                    x0=self.button_1_down_coord[0], y0=self.button_1_down_coord[1],
                    x1=self.button_1_up_coord[0], y1=self.button_1_up_coord[1],
                    x_from=x_from, x_to=x_to
                    )
                )

    # -- RIGHT MOUSE CLICK --

    def record_button_3_down(self, event) -> None:
        self.button_3_down_coord:Tuple[int, int] = (event.x, event.y)

    def record_button_3_up(self, event) -> None:
        self.button_3_up_coord:Tuple[int, int] = (event.x, event.y)
        # -- delete bar --
        if self.button_3_down_coord == self.button_3_up_coord:
            x = self.app.convert_x_pos(event.x)
            x_bar_pos = self.app.view_1.get_closest_bar(x=x)
            if x_bar_pos is None:
                print(f"view_1: No bar to delete from: {event.x} {event.y} | {x}")
            else:
                self.app.view_1.remove_bar(x_bar_pos)
                print(f"view_1: A bar was deleted from: {event.x} {event.y} | {x} | {x_bar_pos}")
        else:
            pass

    # -- HOVER FRAME --

    def hover_canvas_in(self, event) -> None:
        self.app.view_1.axes.set_facecolor((0.96, 0.96, 0.96))
        self.app.view_1.canvas.draw()

    def hover_canvas_out(self, event) -> None:
        self.app.view_1.axes.set_facecolor((1.0, 1.0, 1.0))
        self.app.view_1.canvas.draw()


class Events2():
    def __init__(self, parent) -> None:
        # -- init class --
        self.app = parent

        # -- click coordinate memory --
        self.button_1_down_coord:Tuple[int, int] = (None, None)
        self.button_1_up_coord:Tuple[int, int] = (None, None)

        self.button_3_down_coord:Tuple[int, int] = (None, None)
        self.button_3_up_coord:Tuple[int, int] = (None, None)


class Events3():
    def __init__(self, parent) -> None:
        # -- init class --
        self.app = parent

        # -- click coordinate memory --
        self.button_1_down_coord:Tuple[int, int] = (None, None)
        self.button_1_up_coord:Tuple[int, int] = (None, None)

        self.button_3_down_coord:Tuple[int, int] = (None, None)
        self.button_3_up_coord:Tuple[int, int] = (None, None)


# -------------------------------------------------------------------
# Views (view)
# -------------------------------------------------------------------

class View1():
    def __init__(self, parent) -> None:
        self.app = parent

        # -- create frame --
        self.frame = tk.Frame(self.app)
        self.frame.pack(sid="top", fill='x')

        # -- init plot --
        self.figure = plt.Figure(figsize=(6,2), dpi=100)
        self.axes = self.figure.add_subplot()

        # -- adjust axes style --
        self.axes.set_yticklabels([])
        self.axes.set_xticklabels([])
        self.axes.grid(axis="x")
        self.figure.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=None, hspace=None)

        # -- init canvas --
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill='x', side='top')

    def get_plot(self) -> None:
        self.reset_bars()
        self.axes.plot(self.app.data_1.x_sm, self.app.data_1.y_sm)
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.axes.grid(axis="x")
        for x in self.app.data_1.bars:
            self.axes.axvline(x=x, color='red', gid=str(x))
        self.canvas.draw()
    
    def insert_bar(self, x:Union[int,float]) -> None:
        self.app.data_1.bars += [x]
        self.axes.axvline(x=x, color='red', gid=str(x))
        self.canvas.draw()
    
    def remove_bar(self, x:Union[int,float]) -> None:
        self.app.data_1.bars.remove(x)
        for c in self.axes.lines:
            if c.get_gid() == str(x):
                c.remove()
        self.canvas.draw()
    
    def move_bar(self, x_from:Union[int,float], x_to:Union[int,float]) -> None:
        # -- update data (bars & time series) --
        self.app.data_1.bars.remove(x_from)
        closest_bars = self.get_closest_bars(x=x_to)
        print(f"closest bars to {x_to} are: {closest_bars[0]} and {closest_bars[1]}")
        self.apply_dtw_from_bars(x_from=x_from, x_to=x_to, x_min_glob=closest_bars[0], x_max_glob=closest_bars[1])
        self.app.data_1.bars += [x_to] # Note: keep this line after 'self.apply_dtw_from_bars' and before 'self.axes.axvline' !!

        # -- update plot --
        self.axes.cla()
        self.axes.plot(self.app.data_1.x_sm, self.app.data_1.y_sm)
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.axes.grid(axis="x")
        for x in self.app.data_1.bars:
            self.axes.axvline(x=x, color='red', gid=str(x))
        self.canvas.draw()

    def reload_axis(self):
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.canvas.draw()

    def clear_plot(self):
        self.axes.cla()
        self.canvas.draw()

    # -- MISC -------------------------------------------------------

    def reset_bars(self) -> None:
        self.app.data_1.bars = []

    def bar_exists(self, x:Union[int,float], window_perc:Optional[float] = 0.013) -> bool:
        """
            Checks if a bar exists within the predefined limits (based on percentage deviation relative to the window size).
            Returns the exact position of the bar if found, otherwise returns None.

            Args:
                x (None,int,float): x position of the bar in the graph
                window_perc (float): how large is the window in percent (relative to the displayed graph limits), in which the bar should exist?
            
            Returns:
                (bool): Returns True if a bar exists within proximity, otherwise returns False.
        """
        if window_perc is None or window_perc == 0 or window_perc == 0.0:
            if x in self.app.data_1.bars:
                return True
            else:
                return False
        
        else:
            deviation = ((0.5 * window_perc) * (self.app.x_max - self.app.x_min))
            lower_bound = x - deviation
            upper_bound = x + deviation
            
            results = [x for x in self.app.data_1.bars if lower_bound < x < upper_bound]
            
            if len(results) == 0:
                return False
            else:
                return True
    
    def get_closest_bar(self, x:Union[int,float], window_perc:float = 0.01) -> Optional[Union[int,float]]:
        """
            Checks if a bar exists within the predefined limits (based on percentage deviation relative to the window size).
            Returns the exact position of the bar if found, otherwise returns None.

            Args:
                x (int,float): x position of the bar in the graph
                window_perc (float): how large is the window in percent (relative to the displayed graph limits), in which the bar should exist?
            
            Returns:
                (None,int,float): Returns the x position of a bar within the specified range in case it was found, otherwise returns None.
        """
        deviation = ((0.5 * window_perc) * (self.app.x_max - self.app.x_min))
        lower_bound = x - deviation
        upper_bound = x + deviation
        
        results = [x for x in self.app.data_1.bars if lower_bound < x < upper_bound]
        
        if len(results) == 0:
            return None
        else:
            return min(results, key=lambda list_item:abs(list_item-x))

    def get_closest_bars(self, x:Union[int,float]) -> Tuple[Union[int,float], Union[int,float]]:
        """
            Returns the closest bars on both sides relative to a specified x position.

            Args:
                x (int,float): x position on the graph

            Returns:
                (Tuple[Union[int,float], Union[int,float]]): returns the closest bars on both sides relative to a specified x position.
        """
        candidates_lower_all = self.app.data_1.bars + [self.app.x_min_glob]
        candidates_upper_all = self.app.data_1.bars + [self.app.x_max_glob]

        candidates_lower = [k for k in candidates_lower_all if k < x]
        candidates_upper = [k for k in candidates_upper_all if k > x]
        
        result_lower = min(candidates_lower, key=lambda list_item:abs(list_item-x))
        result_upper = min(candidates_upper, key=lambda list_item:abs(list_item-x))
        
        return (result_lower, result_upper)

    def validate_new_bar_pos(self, x_from:Union[int,float], x_to:Union[int,float]) -> bool:
        """
            Checks if the new position of the bar is not out bounds, i.e. it can not cross another bar.
            Only useful when moving a bar to a new position.

            Args:
                x_from (int,float): previous x position on the plot
                x_to (int,float): new x position on the plot

            Returns:
                (bool): returns True if the new position fulfills the check, otherwise False if it fails the check.
        """
        for x in self.app.data_1.bars:
            if x <= x_from and x < x_to:
                continue
            elif x >= x_from and x > x_to:
                continue
            else:
                return False
        return True

    def apply_dtw_from_bars(self, x_from:Union[int,float], x_to:Union[int,float], x_min_glob:Union[int,float], x_max_glob:Union[int,float]) -> None:
        """
            Args:
                x_from (int,float): previous position of the bar
                x_to (int,float): new position of the bar
                x_min_glob (int,float): this is the position of the closest bar to the left, relative to x_from and x_to
                x_max_glob (int,float): this is the position of the closest bar to the right, relative to x_from and x_to
        """
        # -- define mappings --
        # x:       time (sec)
        # f(x), y: time (sec) remapped
        x = [self.app.x_min_glob] + self.app.data_1.bars + [self.app.x_max_glob] + [x_from]
        y = [self.app.x_min_glob] + self.app.data_1.bars + [self.app.x_max_glob] + [x_to]
        x = np.array(x)
        y = np.array(y)

        # -- interpolation methods --
        f = interp1d(x, y, fill_value='extrapolate')

        # -- update data -- (only update data within the relevant range!)
        data = self.app.data_1.x_sm

        idx_start = np.searchsorted(data, x_min_glob)
        idx_end = np.searchsorted(data, x_max_glob)

        self.app.data_1.x_sm[idx_start:idx_end] = [f(x) for x in data[idx_start:idx_end]]


class View2():
    def __init__(self, parent) -> None:
        self.app = parent

        # -- create frame --
        self.frame = tk.Frame(self.app)
        self.frame.pack(sid="top", fill='x')

        # -- init plot --
        self.figure = plt.Figure(figsize=(6,2), dpi=100)
        self.axes = self.figure.add_subplot()

        # -- adjust axes style --
        self.axes.set_yticklabels([])
        self.axes.set_xticklabels([])
        self.axes.grid(axis="x")
        self.figure.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=None, hspace=None)

        # -- init canvas --
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill='x', side='top')

    def get_plot(self):
        self.axes.plot(self.app.data_2.x_sm, self.app.data_2.y_sm)
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.axes.grid(axis="x")
        self.canvas.draw()
    
    def reload_axis(self):
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.canvas.draw()

    def clear_plot(self):
        self.axes.cla()
        self.canvas.draw()


class View3():
    def __init__(self, parent) -> None:
        self.app = parent

        # -- create frame --
        self.frame = tk.Frame(self.app)
        self.frame.pack(sid="top", fill='x')

        # -- init plot --
        self.figure = plt.Figure(figsize=(6,2), dpi=100)
        self.axes = self.figure.add_subplot()

        # -- adjust axes style --
        self.axes.grid(axis="x")
        self.figure.subplots_adjust(left=0.0, bottom=None, right=1.0, top=1.0, wspace=None, hspace=None)

        # -- init canvas --
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill='x', side='top')
        
    def get_plot(self):
        segs = []
        colors = []
        my_cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=127)
        notes = self.app.data_3.df_midi["note"].unique()
        for note in notes:
            df_note = self.app.data_3.df_midi.where(self.app.data_3.df_midi["note"] == note).dropna()
            x1, x2, y1, y2, velocity = [None, None, note, note, None]
            for idx, row in df_note.iterrows():
                if row["type"] == "note_on":
                    x1 = row["time abs (sec)"]
                    velocity = row["velocity"]
                elif row["type"] == "note_off":
                    x2 = row["time abs (sec)"]
                    if x1 is not None:
                        segs.append(((x1, y1), (x2, y2)))
                        colors.append(my_cmap(norm(velocity)))
                        x1 = None
                        x2 = None
                else:
                    continue

        ln_coll = matplotlib.collections.LineCollection(segs, colors=colors)

        self.axes.add_collection(ln_coll)
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.axes.set_ylim(min(notes)-1, max(notes)+1)
        self.axes.grid(axis="x")
        self.canvas.draw()

    def reload_axis(self):
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.canvas.draw()

    def clear_plot(self):
        self.axes.cla()
        self.canvas.draw()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    # -- init app --
    app = App()

    # -- run app --
    app.mainloop()

