###################### BEGIN GPL LICENSE BLOCK ######################
#
#     DTW APP TO ALIGN MIDI WITH AUDIO
#     Copyright (C) 2022 Antoine Heidt 
#     
#     This program is free software; you can redistribute it and/or
#     modify it under the terms of the GNU General Public License as 
#     published by the Free Software Foundation; either version 3 
#     of the License, or (at your option) any later version.
#     
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#     
#     You should have received a copy of the GNU General Public License
#     along with this program; if not, write to the Free Software Foundation,
#     Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
###################### END GPL LICENSE BLOCK ########################

# ----------------------<[ SUPPORT / CONTACT ]>----------------------
# Programmed by: Antoine Heidt
# GitHub Repository: https://www.github.com/aheidt/dtw-app
# -------------------------------------------------------------------

# -- package info --
__version__:str = "1.4.0" 


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
from typing import List, Any, Optional, Tuple, Union
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -- plotting --
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.collections
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -- music player --
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True'
from pygame import mixer

# -- dtw --
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d

# -- custom --
from dtw import DTW, MidiIO

# -- Settings --
LOAD_SAMPLE_ON_START:bool = False


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
        self.downsampling_factor_1:int = 100 # only plot every N-th data point from .mp3 for performance reasons
        self.downsampling_factor_2:int = 100 # only plot every N-th data point from .mp3 for performance reasons
        
        # -- init variables --
        self.dtw_enabled:bool = True           # disable dtw once data has been manipulated
        self.dtw_output_available:bool = False # enable dtw stats once dtw has been applied

        # -- init data containers --
        self.bars_1 = Bars1(self)
        self.bars_2 = Bars2(self)
        self.data_1 = Data1(self)
        self.data_2 = Data2(self)
        self.data_3 = Data3(self)
        self.data_4 = Data4(self)
        self.data_5 = Data5(self)

        # -- init project container --
        self.project_data = ProjectData(self)

        # -- load sample data --
        if LOAD_SAMPLE_ON_START is True:
            self.project_data.load_demo()

        # -- dtw object --
        self.dtw_obj:DTW = DTW(x_raw=None, y_raw=None, fs=None, df_midi=None)

        # -------------------------------------------------
        # VIEWS (view)
        # -------------------------------------------------

        # -- window style --
        self.geometry('1200x660')
        self.title("Dynamic Time Warp Tool")
        self.iconbitmap(os.path.join(os.path.dirname(__file__), "logo.ico"))

        # -- init constants --
        self.x_min:Union[int,float] = 0 # xlim of current view
        self.x_max:Union[int,float] = 1 # xlim of current view
        self.x_min_glob:Union[int,float] = 0 # lowest xlim that is allowed
        self.x_max_glob:Union[int,float] = 1 # highest xlim that is allowed
        self.hover_color = (0.96, 0.96, 0.96)

        # -- init views --
        self.view_1 = View1(self)
        self.view_2 = View2(self)
        self.view_3 = View3(self)
        self.view_4 = View4(self)
        self.view_5 = View5(self)

        # -- plot sample data --
        if LOAD_SAMPLE_ON_START is True:
            self.reset_bounds()
            self.view_1.get_plot()
            self.view_2.get_plot()
            self.view_3.get_plot()
            self.view_4.get_plot()
            self.view_5.get_plot()

        # -------------------------------------------------
        # EVENTS (controller)
        # -------------------------------------------------

        # -- init menubar --
        self.menubar = MenuBar(self)
        self.config(menu=self.menubar)

        # -- init events --
        self.click_events = ClickEvents(self)
        self.hover_events = HoverEvents(self)

        # -- init musicplayer --
        self.mp = MusicPlayer(self)
        
        if LOAD_SAMPLE_ON_START is True:
            self.mp.load_track()

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
            x3 = self.data_5.df_midi["time abs (sec)"][-1:].item()
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
    def __init__(self, parent:App) -> None:
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
        filemenu.add_command(label="Load Project", command=self.load_project, accelerator="Ctrl+O")
        filemenu.add_command(label="Save Project as...", command=self.save_project_as)
        filemenu.add_command(label="Save Project", command=self.save_project, accelerator="Ctrl+S")
        filemenu.add_separator()
        filemenu.add_command(label="Open .mp3 (Original)", command=self.on_open_mp3_original, accelerator="Ctrl+I")
        filemenu.add_command(label="Open .mp3 (from MIDI)", command=self.on_open_mp3_from_midi, accelerator="Ctrl+K")
        filemenu.add_command(label="Open .midi", command=self.on_open_midi, accelerator="Ctrl+M")
        filemenu.add_separator()
        filemenu.add_command(label="Export .midi", command=self.on_save_midi, accelerator="Ctrl+E")
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exit_app, accelerator="Ctrl+Q")

        self.app.bind('<Control-n>', self.ctrl_n)
        self.app.bind('<Control-r>', self.ctrl_r)

        self.app.bind('<Control-o>', self.ctrl_o)
        self.app.bind('<Control-s>', self.ctrl_s)
        
        self.app.bind('<Control-i>', self.ctrl_i)
        self.app.bind('<Control-k>', self.ctrl_k)
        self.app.bind('<Control-m>', self.ctrl_m)
        
        self.app.bind('<Control-e>', self.ctrl_e)
        self.app.bind('<Control-q>', self.ctrl_q)

        # -- create menu (DTW) --------------------------------------
        dtwmenu = tk.Menu(menubar, tearoff=0)
        dtwmenu.add_command(label="apply dtw algorithm", command=self.apply_dtw_algo, accelerator="F1")
        dtwmenu.add_separator()
        dtwmenu.add_command(label="show chroma features", command=self.show_chroma_features, accelerator="F2")
        dtwmenu.add_command(label="show dtw mappings", command=self.show_dtw_mappings, accelerator="F3")
        dtwmenu.add_command(label="show remap function", command=self.show_remap_function, accelerator="F4")
        dtwmenu.add_separator()
        dtwmenu.add_command(label="reset bars (track 1)", command=self.reset_bars_track_1)
        dtwmenu.add_command(label="reset bars (track 2)", command=self.reset_bars_track_2)
        dtwmenu.add_command(label="reset all bars", command=self.reset_all_bars)

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

        # -- create menu (Play) -------------------------------------
        self.volume = tk.IntVar()
        self.volume.set(value=100)

        self.start_pos = tk.IntVar()
        self.start_pos.set(value=0)

        playmenu = tk.Menu(menubar, tearoff=0)
        playmenu.add_command(label="Play", command=self.play, accelerator="F9")
        playmenu.add_command(label="Pause/Continue", command=self.pause, accelerator="Space")
        playmenu.add_command(label="Stop", command=self.stop, accelerator="F10")
        playmenu.add_separator()
        volume = tk.Menu(playmenu, tearoff=0)
        volume.add_radiobutton(label="0%", value=0, variable=self.volume, command=self.adjust_volume)
        volume.add_radiobutton(label="25%", value=25, variable=self.volume, command=self.adjust_volume)
        volume.add_radiobutton(label="50%", value=50, variable=self.volume, command=self.adjust_volume)
        volume.add_radiobutton(label="75%", value=75, variable=self.volume, command=self.adjust_volume)
        volume.add_radiobutton(label="100%", value=100, variable=self.volume, command=self.adjust_volume)
        playmenu.add_cascade(label="Volume", menu=volume)
        
        self.app.bind('<space>', self.space)
        self.app.bind('<F9>', self.f9)
        self.app.bind('<F10>', self.f10)

        # -- create menu (Help) -------------------------------------
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help", command=self.help)

        # -- attach the menus to the bar ----------------------------
        self.add_cascade(label="File", menu=filemenu)
        self.add_cascade(label="DTW", menu=dtwmenu)
        self.add_cascade(label="View", menu=viewmenu)
        self.add_cascade(label="Play", menu=playmenu)
        self.add_cascade(label="Help", menu=helpmenu)
    
    # ---------------------------------------------------------------
    # FUNCTIONALITY
    # ---------------------------------------------------------------

    # -- FILE -------------------------------------------------------

    def new(self) -> None:
        # -- load sample data --
        self.app.project_data.load_demo()

        # -- plot sample data --
        self.app.reset_bounds()
        self.app.view_1.get_plot()
        self.app.view_2.get_plot()
        self.app.view_5.get_plot()
        self.app.view_3.get_plot()
        self.app.view_4.get_plot()

    def restart(self) -> None:
        self.app.destroy()
        app=App()
        app.mainloop()

    # ---------------------------------------------------------------

    def load_project(self) -> None:
        # -- load data --
        filename = filedialog.askopenfilename(initialdir="/", title="Open file", filetypes=(("data files","*.data;*.DATA"),("All files","*.*")))
        if filename == "":
            print("Cancelled loading project")
            return None
        self.app.project_data.load_file(filename=filename)

        # -- reset graph limits --
        self.app.reset_bounds()
        
        # -- update graphs --
        self.app.view_1.get_plot()
        self.app.view_2.get_plot()

        if self.app.data_3.x is not None:
            self.app.view_3.get_plot()
        else:
            self.app.view_3.reload_axis()
    
        if self.app.data_4.x is not None:
            self.app.view_4.get_plot()
        else:
            self.app.view_4.reload_axis()
        
        if len(self.app.data_5.df_midi) > 0:
            self.app.view_5.get_plot()
        else:
            self.app.view_5.reload_axis()

        # -- load audiofile --
        try:
            self.app.mp.load_track()
        except Exception:
            print("Could not load audio for playback")

    def save_project_as(self) -> None:
        # -- save data --
        self.app.project_data.filename = filedialog.asksaveasfilename(initialdir="/", title="Save as", filetypes=(("data files","*.data;*.DATA"),("All files","*.*")))
        if self.app.project_data.filename == "":
            print("Cancelled saving project")
            return None
        self.app.project_data.save_file(filename=self.app.project_data.filename)

    def save_project(self) -> None:
        """
            Saves the project without asking for a file location, if it has already been defined previously.
        """
        # -- save data --
        if self.app.project_data.filename is None:
            self.app.project_data.filename = filedialog.asksaveasfilename(initialdir="/", title="Save as", filetypes=(("data files","*.data;*.DATA"),("All files","*.*")))
            if self.app.project_data.filename == "":
                print("Cancelled saving project")
                return None
            self.app.project_data.save_file(filename=self.app.project_data.filename)
        else:
            self.app.project_data.save_file(filename=self.app.project_data.filename)

    # ---------------------------------------------------------------

    def on_open_mp3_original(self) -> None:
        # -- load dataset --
        filename = filedialog.askopenfilename(initialdir="/", title="Open file", filetypes=(("audio files","*.mp3;*.wav;"),("all files","*.*")))
        
        if filename == "":
            print("Cancelled opening original mp3 file")
            return None
        
        print("Loading mp3 file...")
        self.app.data_1.load_file(filename)
        
        print("Computing chroma features...")
        self.app.data_3.load_chroma_features()
        
        print("Adjusting window...")

        # -- reset graph limits --
        self.app.reset_bounds()
        
        # -- update graphs --
        self.app.view_1.get_plot()
        self.app.view_2.reload_axis()
        self.app.view_5.reload_axis()
        self.app.view_3.get_plot()
        self.app.view_4.reload_axis()

        # -- reset that track has not been loaded to audio playback engine yet --
        self.app.mp.track_loaded = False

        print("Done")

    def on_open_mp3_from_midi(self) -> None:
        # -- load dataset --
        filename = filedialog.askopenfilename(initialdir="/", title="Open file", filetypes=(("audio files","*.mp3;*.wav;"),("all files","*.*")))
        
        if filename == "":
            print("Cancelled opening mp3 from midi")
            return None
        
        print("Loading mp3 file...")
        self.app.data_2.load_file(filename)

        print("Computing chroma features...")
        self.app.data_4.load_chroma_features()
               
        print("Adjusting window...")

        # -- reset graph limits --
        self.app.reset_bounds()
        
        # -- update graphs --
        self.app.view_1.reload_axis()
        self.app.view_2.get_plot()
        self.app.view_5.reload_axis()
        self.app.view_3.reload_axis()
        self.app.view_4.get_plot()

        print("Done")
        
    def on_open_midi(self) -> None:
        # -- load dataset --
        filename = filedialog.askopenfilename(initialdir="/", title="Open file", filetypes=(("MIDI files","*.mid;*.MID"),("All files","*.*")))
        
        if filename == "":
            print("Cancelled opening midi file")
            return None
        
        print("Loading file...")

        self.app.data_5.load_file(filename)

        print("Adjusting window...")

        # -- reset graph limits --
        self.app.reset_bounds()
        
        # -- update graphs --
        self.app.view_1.reload_axis()
        self.app.view_2.reload_axis()
        self.app.view_5.get_plot()
        self.app.view_3.reload_axis()
        self.app.view_4.reload_axis()

        print("Done")

    # ---------------------------------------------------------------

    def on_save_midi(self) -> None:
        self.app.data_5.outfile = filedialog.asksaveasfilename(initialdir="/", title="Save as", filetypes=(("MIDI files","*.mid;*.MID"),("All files","*.*")))
        if self.app.data_5.outfile == "":
            print("Cancelled saving midi file")
            return None
        
        if self.app.data_5.outfile.endswith((".mid", ".MID")) is False:
            self.app.data_5.outfile += ".mid"
        MidiIO.export_midi(df_midi=self.app.data_5.df_midi, outfile=self.app.data_5.outfile, time_colname="time abs (sec)")
        print(f"Saved midi file to: {self.app.data_5.outfile}")

    # ---------------------------------------------------------------

    def exit_app(self) -> None:
        self.app.destroy()

    # -- DTW --------------------------------------------------------

    def apply_dtw_algo(self) -> None:
        # -- compute DTW time mappings --
        self.app.dtw_obj = DTW(x_raw=self.app.data_1.y, y_raw=self.app.data_2.y, fs=self.app.data_1.fs, df_midi=self.app.data_5.df_midi)
        
        if self.app.dtw_enabled is False:
            messagebox.showerror(title="Error", message="DTW is not available for already manipulated data.\nTo run it again, restart the app and load fresh audio + midi data.")
            return None

        if any([x is None or len(x) == 0 for x in [self.app.data_1.y, self.app.data_2.y, self.app.data_5.df_midi]]) or self.app.data_1.fs is None:
            messagebox.showerror(title="Error", message="Please load some data first!  (audio + midi)")
            return None

        print("Computing chroma features...")
        # self.app.dtw_obj.compute_chroma_features()
        self.app.dtw_obj.x_chroma = self.app.data_3.chroma
        self.app.dtw_obj.y_chroma = self.app.data_4.chroma
        
        print("Computing DTW...")
        self.app.dtw_obj.compute_dtw()

        print("Computing remap function...")
        self.app.dtw_obj.compute_remap_function()

        # -- apply DTW time mappings --
        print("Remapping midi...")
        self.app.data_5.df_midi["time abs (sec)"] = [self.app.dtw_obj.f(x) for x in self.app.data_5.df_midi["time abs (sec)"]]
        
        print("Remapping mp3...")
        self.app.data_2.x_sm = [self.app.dtw_obj.f(x) for x in self.app.data_2.x_sm]

        print("Remapping chroma features...")
        self.app.data_4.x = [self.app.dtw_obj.f(x) for x in self.app.data_4.x]

        # -- draw graphs --
        print("Redrawing graphs")
        self.app.view_2.get_plot()
        self.app.view_5.get_plot()
        self.app.view_4.get_plot()

        # -- trigger axis adjustment --
        print("Resetting axis")
        self.app.reset_bounds()
        self.app.view_1.reload_axis()
        self.app.view_2.reload_axis()
        self.app.view_5.reload_axis()
        self.app.view_3.reload_axis()
        self.app.view_4.reload_axis()

        # -- disable dtw --
        self.app.dtw_enabled = False

        # -- enable detailed dtw reports ---
        self.app.dtw_output_available = True

        print("Done")

    def show_chroma_features(self) -> None:
        if self.app.dtw_output_available is True:
            self.app.dtw_obj.plot_chroma_features()
        else:
            messagebox.showinfo(title="Info", message="Run DTW first to enable stats.")

    def show_dtw_mappings(self) -> None:
        if self.app.dtw_output_available is True:
            self.app.dtw_obj.plot_dtw_mappings()
        else:
            messagebox.showinfo(title="Info", message="Run DTW first to enable stats.")

    def show_remap_function(self) -> None:
        if self.app.dtw_output_available is True:
            self.app.dtw_obj.plot_remap_function()
        else:
            messagebox.showinfo(title="Info", message="Run DTW first to enable stats.")

    def reset_bars_track_1(self) -> None:
        # -- reset bars data --
        self.app.bars_1.reset_bars()

        # -- update view 1 --
        self.app.view_1.get_plot()

        # -- update view 3 --
        if self.app.data_3.x is not None:
            self.app.view_3.get_plot()
    
    def reset_bars_track_2(self) -> None:
        # -- reset bars data --
        self.app.bars_2.reset_bars()

        # -- update view 2 --
        self.app.view_2.get_plot()

        # -- update view 4 --
        if self.app.data_4.x is not None:
            self.app.view_4.get_plot()

        # -- update view 5 --
        if len(self.app.data_5.df_midi) > 0:
            self.app.view_5.get_plot()
    
    def reset_all_bars(self) -> None:
        # -- reset bars data --
        self.app.bars_1.reset_bars()
        self.app.bars_2.reset_bars()

        # -- update view 1 --
        self.app.view_1.get_plot()

        # -- update view 2 --
        self.app.view_2.get_plot()

        # -- update view 3 --
        if self.app.data_3.x is not None:
            self.app.view_3.get_plot()
        
        # -- update view 4 --
        if self.app.data_4.x is not None:
            self.app.view_4.get_plot()

        # -- update view 5 --
        if len(self.app.data_5.df_midi) > 0:
            self.app.view_5.get_plot()

    # -- VIEW -------------------------------------------------------

    def zoom_in(self) -> None:
        # -- adjust x axis limits --
        zoom_amount = ((self.app.x_max - self.app.x_min) * 1/3) * 0.5

        self.app.x_min += zoom_amount
        self.app.x_max -= zoom_amount

        # -- trigger axis adjustment --
        self.app.view_1.reload_axis()
        self.app.view_2.reload_axis()
        self.app.view_5.reload_axis()
        self.app.view_3.reload_axis()
        self.app.view_4.reload_axis()

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
        self.app.view_5.reload_axis()
        self.app.view_3.reload_axis()
        self.app.view_4.reload_axis()

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
        self.app.view_5.reload_axis()
        self.app.view_3.reload_axis()
        self.app.view_4.reload_axis()

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
        self.app.view_5.reload_axis()
        self.app.view_3.reload_axis()
        self.app.view_4.reload_axis()

    # -- PLAY -------------------------------------------------------

    def play(self) -> None:
        self.app.mp.play()

    def pause(self) -> None:
        self.app.mp.pause()

    def stop(self) -> None:
        self.app.mp.stop()

    def adjust_volume(self) -> None:
        self.app.mp.adjust_volume()

    # -- HELP -------------------------------------------------------

    def help(self):
        messagebox.showinfo(title="Help", message="Documentation available at:\nhttps://www.github.com/aheidt/dtw-app")

    # ---------------------------------------------------------------
    # HOTKEY EVENTS
    # ---------------------------------------------------------------

    # -- FILE -------------------------------------------------------

    def ctrl_n(self, event) -> None:
        self.new()

    def ctrl_r(self, event) -> None:
        self.restart()

    # ---------------------------------------------------------------

    def ctrl_o(self, event) -> None:
        self.load_project()

    def ctrl_s(self, event) -> None:
        self.save_project()

    # ---------------------------------------------------------------

    def ctrl_i(self, event) -> None:
        self.on_open_mp3_original()

    def ctrl_k(self, event) -> None:
        self.on_open_mp3_from_midi()

    def ctrl_m(self, event) -> None:
        self.on_open_midi()

    # ---------------------------------------------------------------

    def ctrl_e(self, event) -> None:
        self.on_save_midi()

    def ctrl_q(self, event) -> None:
        self.exit_app()

    # -- DTW --------------------------------------------------------

    def f1(self, event) -> None:
        self.apply_dtw_algo()

    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------

    def ctrl_right(self, event) -> None:
        self.scroll_right()

    def ctrl_left(self, event) -> None:
        self.scroll_left()

    # -- PLAY -------------------------------------------------------

    def space(self, event) -> None:
        self.pause()
    
    def f9(self, event) -> None:
        self.play()

    def f10(self, event) -> None:
        self.stop()


# -------------------------------------------------------------------
# Data (model)
# -------------------------------------------------------------------

class ProjectData():
    """Packages the data of Data1, Data2 and Data5 into a .data file and vice versa."""
    def __init__(self, parent:App) -> None:
        # -- init parent --
        self.app = parent
        self.filename:Optional[str] = None
    
    def load_demo(self) -> None:
        # -- data source --
        data_dir = os.path.dirname(__file__)
        filename = os.path.join(data_dir, "demo.data")

        # -- load data --
        self.load_file(filename=filename)
    
    def load_file(self, filename:str) -> None:
        # -- load data --
        if filename.endswith((".data", ".DATA")) is False:
            filename += ".data"
        
        with open(filename, 'rb') as filehandle:
            data = pickle.load(filehandle)
        
        # -- unpack data --
        settings = data["settings"]
        bars_1   = data["bars_1"]
        bars_2   = data["bars_2"]
        data_1   = data["data_1"]
        data_2   = data["data_2"]
        data_3   = data["data_3"]
        data_4   = data["data_4"]
        data_5   = data["data_5"]

        # -- check if data meets version requirements --
        if settings["version"].split(sep=".")[:2] != __version__.split(sep=".")[:2]:
            raise IOError("Couldn't load project, incompatible versions (data: {data_version}, app: {app_version})".format(
                data_version=settings["version"], app_version=__version__))

        # -- settings --
        self.app.dtw_enabled          = settings["dtw_enabled"]
        self.app.dtw_output_available = settings["dtw_output_available"]

        # -- bars --
        self.app.bars_1.bars = bars_1["bars_1"]
        self.app.bars_2.bars = bars_2["bars_2"]

        # -- data_1 --
        self.app.data_1.filename = data_1["filename"]
        self.app.data_1.x        = data_1["x"]
        self.app.data_1.y        = data_1["y"]
        self.app.data_1.x_sm     = data_1["x_sm"]
        self.app.data_1.y_sm     = data_1["y_sm"]
        self.app.data_1.fs       = data_1["fs"]

        # -- data_2 --
        self.app.data_2.filename = data_2["filename"]
        self.app.data_2.x        = data_2["x"]
        self.app.data_2.y        = data_2["y"]
        self.app.data_2.x_sm     = data_2["x_sm"]
        self.app.data_2.y_sm     = data_2["y_sm"]
        self.app.data_2.fs       = data_2["fs"]

        # -- data_3 --
        self.app.data_3.chroma     = data_3["chroma"]
        self.app.data_3.x          = data_3["x"]
        self.app.data_3.y          = data_3["y"]
        self.app.data_3.fs         = data_3["fs"]
        self.app.data_3.hop_length = data_3["hop_length"]

        # -- data_4 --
        self.app.data_4.chroma     = data_4["chroma"]
        self.app.data_4.x          = data_4["x"]
        self.app.data_4.y          = data_4["y"]
        self.app.data_4.fs         = data_4["fs"]
        self.app.data_4.hop_length = data_4["hop_length"]

        # -- data_5 --
        self.app.data_5.filename = data_5["filename"]
        self.app.data_5.outfile  = data_5["outfile"]
        self.app.data_5.df_midi  = data_5["df_midi"]

    def save_file(self, filename:str) -> None:
        # -- datasets --
        data_1 = {
            "filename": self.app.data_1.filename,
            "x": self.app.data_1.x,
            "y": self.app.data_1.y,
            "x_sm": self.app.data_1.x_sm,
            "y_sm": self.app.data_1.y_sm,
            "fs": self.app.data_1.fs,
        }
        data_2 = {
            "filename": self.app.data_2.filename,
            "x": self.app.data_2.x,
            "y": self.app.data_2.y,
            "x_sm": self.app.data_2.x_sm,
            "y_sm": self.app.data_2.y_sm,
            "fs": self.app.data_2.fs,
        }
        data_3 = {
            "chroma": self.app.data_3.chroma,
            "x": self.app.data_3.x,
            "y": self.app.data_3.y,
            "fs": self.app.data_3.fs,
            "hop_length": self.app.data_3.hop_length,
        }
        data_4 = {
            "chroma": self.app.data_4.chroma,
            "x": self.app.data_4.x,
            "y": self.app.data_4.y,
            "fs": self.app.data_4.fs,
            "hop_length": self.app.data_4.hop_length,
        }
        data_5 = {
            "filename": self.app.data_5.filename,
            "outfile": self.app.data_5.outfile,
            "df_midi": self.app.data_5.df_midi,
        }
        bars_1 = {
            "bars_1": self.app.bars_1.bars,
        }
        bars_2 = {
            "bars_2": self.app.bars_2.bars,
        }

        # -- miscellaneous data --
        settings = {
            "version": __version__,
            "dtw_enabled": self.app.dtw_enabled,
            "dtw_output_available": self.app.dtw_output_available,
        }

        # -- bundle data --
        data = {
            "settings": settings,
            "bars_1": bars_1,
            "bars_2": bars_2,
            "data_1": data_1,
            "data_2": data_2,
            "data_3": data_3,
            "data_4": data_4,
            "data_5": data_5,
        }

        # -- adjust filename --
        if filename.endswith((".data", ".DATA")) is False:
            filename += ".data"

        # -- write file --        
        with open(filename, 'wb') as filehandle:
            pickle.dump(data, filehandle)

        # -- log / ui message --
        print(f"Successfully saved project to {filename}")
        messagebox.showinfo("Info", "Project successfully saved")


class Bars1():
    def __init__(self, parent:App) -> None:
        # -- init parent --
        self.app = parent

        # -- edit memory --
        self.bars:List[Union[int,float]] = []

    # -- edit bar dataset -------------------------------------------

    def insert_bar(self, x) -> None:
        self.bars += [x]

    def delete_bar(self, x) -> None:
        self.bars.remove(x)

    def reset_bars(self) -> None:
        self.bars = []
    
    # -- get bar relations ------------------------------------------

    def bar_exists(self, x:Union[int,float], window_perc:float = 0.013) -> bool:
        """
            Checks if a bar exists within the predefined limits (based on percentage deviation relative to the window size).
            Returns the exact position of the bar if found, otherwise returns None.

            Args:
                x (None,int,float): x position of the bar in the graph
                window_perc (float): how large is the window in percent (relative to the displayed graph limits), in which the bar should exist?
            
            Returns:
                (bool): Returns True if a bar exists within proximity, otherwise returns False.
        """
        if window_perc == 0 or window_perc == 0.0:
            if x in self.bars:
                return True
            else:
                return False
        
        else:
            deviation = ((0.5 * window_perc) * (self.app.x_max - self.app.x_min))
            lower_bound = x - deviation
            upper_bound = x + deviation
            
            results = [x for x in self.bars if lower_bound < x and x < upper_bound]
            
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
        
        results = [x for x in self.bars if lower_bound < x < upper_bound]
        
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
        candidates_lower_all = self.bars + [self.app.x_min_glob]
        candidates_upper_all = self.bars + [self.app.x_max_glob]

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
        for x in self.bars:
            if x <= x_from and x < x_to:
                continue
            elif x >= x_from and x > x_to:
                continue
            else:
                return False
        return True


class Bars2():
    def __init__(self, parent:App) -> None:
        # -- init parent --
        self.app = parent

        # -- edit memory --
        self.bars:List[Union[int,float]] = []

    # -- edit bar dataset -------------------------------------------

    def insert_bar(self, x) -> None:
        self.bars += [x]

    def delete_bar(self, x) -> None:
        self.bars.remove(x)

    def reset_bars(self) -> None:
        self.bars = []
    
    # -- get bar relations ------------------------------------------

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
            if x in self.bars:
                return True
            else:
                return False
        
        else:
            deviation = ((0.5 * window_perc) * (self.app.x_max - self.app.x_min))
            lower_bound = x - deviation
            upper_bound = x + deviation
            
            results = [x for x in self.bars if lower_bound < x < upper_bound]
            
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
        
        results = [x for x in self.bars if lower_bound < x < upper_bound]
        
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
        candidates_lower_all = self.bars + [self.app.x_min_glob]
        candidates_upper_all = self.bars + [self.app.x_max_glob]

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
        for x in self.bars:
            if x <= x_from and x < x_to:
                continue
            elif x >= x_from and x > x_to:
                continue
            else:
                return False
        return True


class Data1():
    def __init__(self, parent:App) -> None:
        # -- init parent --
        self.app = parent

        # -- data source --
        self.filename:str = ""

        # -- init dataset --
        self.x = np.arange(0, 1.001, 0.001).round(2).tolist()
        self.y = np.arange(0, 1.001, 0.001).round(2).tolist()
        self.x_sm = np.arange(0, 1.001, 0.001).round(2).tolist() # small / reduced dataset
        self.y_sm = np.arange(0, 1.001, 0.001).round(2).tolist() # small / reduced dataset
        self.fs = None
    
    # -- load from file ---------------------------------------------

    def load_file(self, filename) -> None:
        self.filename = filename
        try:
            # -- load mp3 --
            self.y, self.fs = librosa.load(self.filename)

            # -- convert time to seconds for x axis--
            x_num_steps = len(self.y)
            time_length = len(self.y) / self.fs
            self.x = [(i/x_num_steps)*time_length for i in range(x_num_steps)]

            # -- create reduced sample for plotting --
            self.y_sm = self.y[::self.app.downsampling_factor_1]
            self.x_sm = self.x[::self.app.downsampling_factor_1]

        except Exception as e:
            messagebox.showerror("Error Message", f"Could not load file: {self.filename}, because: {repr(e)}")


class Data2():
    def __init__(self, parent:App) -> None:
        # -- init parent --
        self.app = parent

        # -- data source --
        self.filename:str = ""

        # -- init dataset --
        self.x = np.arange(0, 1.001, 0.001).round(2).tolist()
        self.y = np.arange(0, 1.001, 0.001).round(2).tolist()
        self.x_sm = np.arange(0, 1.001, 0.001).round(2).tolist() # small / reduced dataset
        self.y_sm = np.arange(0, 1.001, 0.001).round(2).tolist() # small / reduced dataset
        self.fs = None
    
    # -- load from file ---------------------------------------------

    def load_file(self, filename) -> None:
        self.filename = filename
        try:
            # -- load mp3 --
            self.y, self.fs = librosa.load(self.filename)

            # -- convert time to seconds for x axis--
            x_num_steps = len(self.y)
            time_length = len(self.y) / self.fs
            self.x = [(i/x_num_steps)*time_length for i in range(x_num_steps)]

            # -- create reduced sample for plotting --
            self.y_sm = self.y[::self.app.downsampling_factor_2]
            self.x_sm = self.x[::self.app.downsampling_factor_2]

        except Exception as e:
            messagebox.showerror("Error Message", f"Could not load file: {self.filename}, because: {repr(e)}")

    # -- alter time series ------------------------------------------

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
        x = [self.app.x_min_glob] + self.app.bars_2.bars + [self.app.x_max_glob] + [x_from]
        y = [self.app.x_min_glob] + self.app.bars_2.bars + [self.app.x_max_glob] + [x_to]
        x = np.array(x)
        y = np.array(y)

        # -- interpolation methods --
        f = interp1d(x, y, fill_value='extrapolate')

        # -- update data -- (only update data within the relevant range!)
        data = self.x_sm

        idx_start = np.searchsorted(data, x_min_glob)
        idx_end = np.searchsorted(data, x_max_glob)

        self.x_sm[idx_start:idx_end] = [f(x) for x in data[idx_start:idx_end]]


class Data3():
    def __init__(self, parent:App) -> None:
        # -- init parent --
        self.app = parent

        # -- init dataset --
        self.chroma = None
        self.x:List[Any] = []
        self.y = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'H', '']
        self.fs = None
        self.hop_length:int = 512
    
    # -- load chroma features ---------------------------------------

    def load_chroma_features(self):
        self.fs = self.app.data_1.fs
        harm = librosa.effects.harmonic(y=self.app.data_1.y, margin=8)
        chroma_harm = librosa.feature.chroma_cqt(y=harm, sr=self.fs)
        self.chroma = np.minimum(
            chroma_harm, librosa.decompose.nn_filter(
                chroma_harm, aggregate=np.median, metric='cosine'))

        x_num_steps = len(self.chroma[0])
        self.x = librosa.frames_to_time(np.arange(x_num_steps + 1), sr=self.fs, hop_length=self.hop_length)


class Data4():
    def __init__(self, parent:App) -> None:
        # -- init parent --
        self.app = parent

        # -- init dataset --
        self.chroma = None
        self.x:List[Any] = []
        self.y = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'H', '']
        self.fs = None
        self.hop_length = 512

    # -- load chroma features ---------------------------------------

    def load_chroma_features(self):
        self.fs = self.app.data_2.fs
        harm = librosa.effects.harmonic(y=self.app.data_2.y, margin=8)
        chroma_harm = librosa.feature.chroma_cqt(y=harm, sr=self.fs)
        self.chroma = np.minimum(
            chroma_harm, librosa.decompose.nn_filter(
                chroma_harm, aggregate=np.median, metric='cosine'))

        x_num_steps = len(self.chroma[0])
        self.x = librosa.frames_to_time(np.arange(x_num_steps + 1), sr=self.fs, hop_length=self.hop_length)

    # -- alter time series ------------------------------------------

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
        x = [self.app.x_min_glob] + self.app.bars_2.bars + [self.app.x_max_glob] + [x_from]
        y = [self.app.x_min_glob] + self.app.bars_2.bars + [self.app.x_max_glob] + [x_to]
        x = np.array(x)
        y = np.array(y)

        # -- interpolation methods --
        f = interp1d(x, y, fill_value='extrapolate')

        # -- update data -- (only update data within the relevant range!)
        data = self.x

        idx_start = np.searchsorted(data, x_min_glob)
        idx_end = np.searchsorted(data, x_max_glob)

        self.x[idx_start:idx_end] = [f(x) for x in data[idx_start:idx_end]]


class Data5():
    def __init__(self, parent:App) -> None:
        # -- init parent --
        self.app = parent

        # -- data source --
        self.filename:Optional[str] = None
        self.outfile:Optional[str] = None

        # -- init dataset --
        self.df_midi:pd.DataFrame = pd.DataFrame()

    # -- load from file ---------------------------------------------

    def load_file(self, filename) -> None:
        self.filename = filename
        try:
            # -- load midi --
            self.df_midi = MidiIO.midi_to_df(file_midi=self.filename, clip_t0=False)
        except Exception as e:
            messagebox.showerror("Error Message", f"Could not load file: {self.filename}, because: {repr(e)}")

    # -- alter time series ------------------------------------------

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
        x = [self.app.x_min_glob] + self.app.bars_2.bars + [self.app.x_max_glob] + [x_from]
        y = [self.app.x_min_glob] + self.app.bars_2.bars + [self.app.x_max_glob] + [x_to]
        x = np.array(x)
        y = np.array(y)

        # -- interpolation methods --
        f = interp1d(x, y, fill_value='extrapolate')

        # -- update data -- (only update data within the relevant range!)
        idx_start = np.searchsorted(self.df_midi["time abs (sec)"], x_min_glob)
        idx_end = np.searchsorted(self.df_midi["time abs (sec)"], x_max_glob)

        self.df_midi.loc[idx_start:idx_end,"time abs (sec)"] = [f(x) for x in self.df_midi.loc[idx_start:idx_end,"time abs (sec)"]]


# -------------------------------------------------------------------
# Views (view)
# -------------------------------------------------------------------

class View():
    def __init__(
        self, parent:App, figsize:Tuple[Union[float,int],Union[float,int]]=(6,1), 
        drop_axis:bool=True, margins:dict={'left':0.0, 'bottom':0.0, 'right':1.0, 'top':1.0}
        ) -> None:

        # -- pass data --
        self.app = parent

        # -- create frame --
        self.frame = tk.Frame(self.app)
        self.frame.pack(sid="top", fill='x')

        # -- init plot --
        self.figure = plt.Figure(figsize=figsize, dpi=100)
        self.axes = self.figure.add_subplot()

        # -- adjust axes style --
        self.axes.grid(axis="x")
        if drop_axis is True:
            self.axes.set_yticklabels([])
            self.axes.set_xticklabels([])
        self.figure.subplots_adjust(left=margins['left'], bottom=margins['bottom'], right=margins['right'], top=margins['top'], wspace=None, hspace=None)

        # -- init canvas --
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def insert_bar(self, x:Union[int,float], color:str='red') -> None:
        self.axes.axvline(x=x, color=color, gid=str(x))
        self.canvas.draw()
    
    def delete_bar(self, x:Union[int,float]) -> None:
        for c in self.axes.lines:
            if c.get_gid() == str(x):
                c.remove()
        self.canvas.draw()
        
    def reload_axis(self):
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.canvas.draw()

    def clear_plot(self):
        self.axes.cla()
        self.canvas.draw()

    def set_bg_color(self, color=(1.0, 1.0, 1.0)):
        self.axes.set_facecolor(color)
        self.canvas.draw()


class View1(View):
    def __init__(self, parent:App) -> None:
        # -- init inherited methods from view --
        super().__init__(parent=parent, figsize=(6,1.5))
    
    def get_plot(self) -> None:
        """Loads a fresh version of the plot."""
        self.axes.cla()
        self.axes.plot(self.app.data_1.x_sm, self.app.data_1.y_sm)
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.axes.grid(axis="x")
        for x in self.app.bars_1.bars:
            self.axes.axvline(x=x, color='orange', gid=str(x))
        self.canvas.draw()


class View2(View):
    def __init__(self, parent:App) -> None:
        # -- init inherited methods from view --
        super().__init__(parent=parent, figsize=(6,1.5))

    def get_plot(self):
        """Loads a fresh version of the plot."""
        self.axes.cla()
        self.axes.plot(self.app.data_2.x_sm, self.app.data_2.y_sm)
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.axes.grid(axis="x")
        for x in self.app.bars_2.bars:
            self.axes.axvline(x=x, color='red', gid=str(x))
        self.canvas.draw()


class View3(View):
    def __init__(self, parent:App) -> None:
        # -- init inherited methods from view --
        super().__init__(parent=parent, figsize=(6,1))
        
    def get_plot(self):
        # -- clear plot --
        self.axes.cla()

        # -- create plot --
        if self.app.data_3.chroma is not None:
            self.axes.pcolormesh(self.app.data_3.x, self.app.data_3.y, self.app.data_3.chroma, shading='auto')

        # -- adjust axis style --
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.axes.grid(axis="x")

        for x in self.app.bars_1.bars:
            self.axes.axvline(x=x, color='orange', gid=str(x))

        # -- draw graph --
        self.canvas.draw()


class View4(View):
    def __init__(self, parent:App) -> None:
        # -- init inherited methods from view --
        super().__init__(parent=parent, figsize=(6,1))
        
    def get_plot(self):
        # -- clear plot --
        self.axes.cla()

        # -- create plot --
        if self.app.data_4.chroma is not None:
            self.axes.pcolormesh(self.app.data_4.x, self.app.data_4.y, self.app.data_4.chroma, shading='auto')

        # -- adjust axis style --
        self.axes.set_xlim([self.app.x_min, self.app.x_max])
        self.axes.grid(axis="x")

        # -- add bars --
        for x in self.app.bars_2.bars:
            self.axes.axvline(x=x, color='red', gid=str(x))

        # -- draw graph --
        self.canvas.draw()


class View5(View):
    def __init__(self, parent:App) -> None:
        # -- init inherited methods from view --
        super().__init__(parent=parent, figsize=(6,1.4), drop_axis=False, margins={'left':0.0, 'bottom':0.16, 'right':1.0, 'top':1.0})

    def get_plot(self):
        # -- clear plot --
        self.axes.cla()

        # -- init variables --
        segs = []
        colors = []
        notes = self.app.data_5.df_midi["note"].unique()

        # -- create colormap --
        my_cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=127)
        
        # Note: We are running into trouble if note on & note off events don't perfectly alternate for a given pitch. 
        #       Unlikely to happen for piano music though.
        for note in notes:
            df_note = self.app.data_5.df_midi.where(self.app.data_5.df_midi["note"] == note).dropna()
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
        for x in self.app.bars_2.bars:
            self.axes.axvline(x=x, color='red', gid=str(x))
        self.canvas.draw()


# -------------------------------------------------------------------
# Events (controller)
# -------------------------------------------------------------------

class ClickEvents():
    def __init__(self, parent:App) -> None:
        # -- init class --
        self.app = parent

        # -----------------------------------------------------------
        # TRACK 1 MIDI
        # -----------------------------------------------------------

        # -- click coordinate memory --
        self.track_1_left_down_coord:Tuple[Optional[int], Optional[int]] = (None, None)
        self.track_1_left_up_coord:Tuple[Optional[int], Optional[int]] = (None, None)

        self.track_1_right_down_coord:Tuple[Optional[int], Optional[int]] = (None, None)
        self.track_1_right_up_coord:Tuple[Optional[int], Optional[int]] = (None, None)

        # -- mouse click [view_1] --
        self.app.view_1.canvas.get_tk_widget().bind("<Button 1>", self.track_1_left_down)      # left mouse click (down)
        self.app.view_1.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.track_1_left_up) # left mouse click (up)

        self.app.view_1.canvas.get_tk_widget().bind('<Button-3>', self.track_1_right_down)      # right mouse click (down)
        self.app.view_1.canvas.get_tk_widget().bind('<ButtonRelease-3>', self.track_1_right_up) # right mouse click (up)

        # -- mouse click [view_3] --
        self.app.view_3.canvas.get_tk_widget().bind("<Button 1>", self.track_1_left_down)      # left mouse click (down)
        self.app.view_3.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.track_1_left_up) # left mouse click (up)

        self.app.view_3.canvas.get_tk_widget().bind('<Button-3>', self.track_1_right_down)      # right mouse click (down)
        self.app.view_3.canvas.get_tk_widget().bind('<ButtonRelease-3>', self.track_1_right_up) # right mouse click (up)

        # -----------------------------------------------------------
        # TRACK 2 MIDI
        # -----------------------------------------------------------

        # -- click coordinate memory --
        self.track_2_left_down_coord:Tuple[Optional[int], Optional[int]] = (None, None)
        self.track_2_left_up_coord:Tuple[Optional[int], Optional[int]] = (None, None)

        self.track_2_right_down_coord:Tuple[Optional[int], Optional[int]] = (None, None)
        self.track_2_right_up_coord:Tuple[Optional[int], Optional[int]] = (None, None)

        # -- mouse click [view_2] --
        self.app.view_2.canvas.get_tk_widget().bind("<Button 1>", self.track_2_left_down)      # left mouse click (down)
        self.app.view_2.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.track_2_left_up) # left mouse click (up)

        self.app.view_2.canvas.get_tk_widget().bind('<Button-3>', self.track_2_right_down)      # right mouse click (down)
        self.app.view_2.canvas.get_tk_widget().bind('<ButtonRelease-3>', self.track_2_right_up) # right mouse click (up)

        # -- mouse click [view_4] --
        self.app.view_4.canvas.get_tk_widget().bind("<Button 1>", self.track_2_left_down)      # left mouse click (down)
        self.app.view_4.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.track_2_left_up) # left mouse click (up)

        self.app.view_4.canvas.get_tk_widget().bind('<Button-3>', self.track_2_right_down)      # right mouse click (down)
        self.app.view_4.canvas.get_tk_widget().bind('<ButtonRelease-3>', self.track_2_right_up) # right mouse click (up)

        # -- mouse click [view_5] --
        self.app.view_5.canvas.get_tk_widget().bind("<Button 1>", self.track_2_left_down)      # left mouse click (down)
        self.app.view_5.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.track_2_left_up) # left mouse click (up)

        self.app.view_5.canvas.get_tk_widget().bind('<Button-3>', self.track_2_right_down)      # right mouse click (down)
        self.app.view_5.canvas.get_tk_widget().bind('<ButtonRelease-3>', self.track_2_right_up) # right mouse click (up)

    # ---------------------------------------------------------------
    # LEFT MOUSE CLICK
    # ---------------------------------------------------------------

    def track_1_left_down(self, event) -> None:
        self.track_1_left_down_coord = (event.x, event.y)

    def track_1_left_up(self, event) -> None:
        self.track_1_left_up_coord = (event.x, event.y)

        # -- incomplete action --------------------------------------
        if self.track_1_left_down_coord == (None, None) or self.track_1_left_up_coord == (None, None):
            pass
        
        # -- create bar ---------------------------------------------
        elif self.track_1_left_down_coord == self.track_1_left_up_coord:
            x_pos = self.app.convert_x_pos(self.track_1_left_up_coord[0])
            bar_exists:bool = self.app.bars_1.bar_exists(x=x_pos)
            if bar_exists is True:
                print(f"A bar already exists at: {event.x} {event.y} | {x_pos}")
            else:
                self.app.bars_1.insert_bar(x_pos)

                self.app.view_1.insert_bar(x_pos, color='orange')
                self.app.view_3.insert_bar(x_pos, color='orange')

                print(f"A new bar was inserted at: {event.x} {event.y} | {x_pos}")

        # -- move bar -----------------------------------------------
        elif any([x is None or len(x) == 0 for x in [self.app.data_1.y, self.app.data_2.y, self.app.data_5.df_midi]]) or self.app.data_1.fs is None:
            messagebox.showerror(title="Error", message="Please load some data first! (audio + midi)")
            return None
        else:
            # -- convert coordinates --
            x_from = self.app.convert_x_pos(self.track_1_left_down_coord[0])
            x_to   = self.app.convert_x_pos(self.track_1_left_up_coord[0])

            # -- get closest bar (from previous position) --
            x_closest_bar = self.app.bars_1.get_closest_bar(x_from)

            if x_closest_bar is not None:
                if self.app.bars_1.validate_new_bar_pos(x_from=x_closest_bar, x_to=x_to) is True:
                    # -- delete bar & insert new bar --
                    self.app.bars_1.delete_bar(x_closest_bar)
                    self.app.bars_1.insert_bar(x_to)

                    # -- update graph --
                    self.app.view_1.delete_bar(x_closest_bar)
                    self.app.view_3.delete_bar(x_closest_bar)
                    self.app.view_1.insert_bar(x_to, color='orange')
                    self.app.view_3.insert_bar(x_to, color='orange')

                    # -- log message --
                    print("A bar was moved from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to}".format(
                        x0=self.track_1_left_down_coord[0], y0=self.track_1_left_down_coord[1],
                        x1=self.track_1_left_up_coord[0], y1=self.track_1_left_up_coord[1],
                        x_from=x_from, x_to=x_to
                        )
                    )
                else:
                    print("A bar could not be moved from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to} (can't cross other bars)".format(
                        x0=self.track_1_left_down_coord[0], y0=self.track_1_left_down_coord[1],
                        x1=self.track_1_left_up_coord[0], y1=self.track_1_left_up_coord[1],
                        x_from=x_from, x_to=x_to
                        )
                    )

                # -- disable dtw --
                self.app.dtw_enabled = False

            else:
                print("No bar to move from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to}".format(
                    x0=self.track_1_left_down_coord[0], y0=self.track_1_left_down_coord[1],
                    x1=self.track_1_left_up_coord[0], y1=self.track_1_left_up_coord[1],
                    x_from=x_from, x_to=x_to
                    )
                )
        
        # -- reset coord --------------------------------------------
        self.track_1_left_down_coord == (None, None)
        self.track_1_left_up_coord == (None, None)

    def track_2_left_down(self, event) -> None:
        self.track_2_left_down_coord = (event.x, event.y)
    
    def track_2_left_up(self, event) -> None:
        self.track_2_left_up_coord = (event.x, event.y)

        # -- incomplete action --------------------------------------
        if self.track_2_left_down_coord == (None, None) or self.track_2_left_up_coord == (None, None):
            pass
        
        # -- create bar ---------------------------------------------
        elif self.track_2_left_down_coord == self.track_2_left_up_coord:
            x_pos = self.app.convert_x_pos(self.track_2_left_up_coord[0])
            bar_exists:bool = self.app.bars_2.bar_exists(x=x_pos)
            if bar_exists is True:
                print(f"A bar already exists at: {event.x} {event.y} | {x_pos}")
            else:
                self.app.bars_2.insert_bar(x_pos)

                self.app.view_2.insert_bar(x_pos)
                self.app.view_4.insert_bar(x_pos)
                self.app.view_5.insert_bar(x_pos)

                print(f"A new bar was inserted at: {event.x} {event.y} | {x_pos}")

        # -- move bar -----------------------------------------------
        elif any([x is None or len(x) == 0 for x in [self.app.data_1.y, self.app.data_2.y, self.app.data_5.df_midi]]) or self.app.data_1.fs is None:
            messagebox.showerror(title="Error", message="Please load some data first!  (audio + midi)")
            return None
        else:
            # -- convert coordinates --
            x_from = self.app.convert_x_pos(self.track_2_left_down_coord[0])
            x_to   = self.app.convert_x_pos(self.track_2_left_up_coord[0])

            # -- get closest bar (from previous position) --
            x_closest_bar = self.app.bars_2.get_closest_bar(x_from)

            if x_closest_bar is not None:
                if self.app.bars_2.validate_new_bar_pos(x_from=x_closest_bar, x_to=x_to) is True:
                    # -- delete bar --
                    self.app.bars_2.delete_bar(x_closest_bar)
                    
                    # -- get the closest bars (left & right), used as range in which to apply DTW --
                    closest_bars = self.app.bars_2.get_closest_bars(x=x_to)
                    print(f"closest bars to {x_to} are: {closest_bars[0]} and {closest_bars[1]}")

                    # -- update data (bars & time series) [data_2, data_4, data_5] --
                    self.app.data_2.apply_dtw_from_bars(x_from=x_from, x_to=x_to, x_min_glob=closest_bars[0], x_max_glob=closest_bars[1])
                    self.app.data_4.apply_dtw_from_bars(x_from=x_from, x_to=x_to, x_min_glob=closest_bars[0], x_max_glob=closest_bars[1])
                    self.app.data_5.apply_dtw_from_bars(x_from=x_from, x_to=x_to, x_min_glob=closest_bars[0], x_max_glob=closest_bars[1])

                    # -- insert the new bar --
                    self.app.bars_2.insert_bar(x_to) # Note: keep this line after 'self.apply_dtw_from_bars' and before 'self.get_plot()' !!

                    # -- update graph --
                    self.app.view_2.get_plot()
                    self.app.view_4.get_plot()
                    self.app.view_5.get_plot()

                    # -- log message --
                    print("A bar was moved from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to}".format(
                        x0=self.track_2_left_down_coord[0], y0=self.track_2_left_down_coord[1],
                        x1=self.track_2_left_up_coord[0], y1=self.track_2_left_up_coord[1],
                        x_from=x_from, x_to=x_to
                        )
                    )
                else:
                    print("A bar could not be moved from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to} (can't cross other bars)".format(
                        x0=self.track_2_left_down_coord[0], y0=self.track_2_left_down_coord[1],
                        x1=self.track_2_left_up_coord[0], y1=self.track_2_left_up_coord[1],
                        x_from=x_from, x_to=x_to
                        )
                    )
    
                # -- disable dtw --
                self.app.dtw_enabled = False
    
            else:
                print("No bar to move from: {x0} {y0} to {x1} {y1} | {x_from} -> {x_to}".format(
                    x0=self.track_2_left_down_coord[0], y0=self.track_2_left_down_coord[1],
                    x1=self.track_2_left_up_coord[0], y1=self.track_2_left_up_coord[1],
                    x_from=x_from, x_to=x_to
                    )
                )
        
        # -- reset coord --------------------------------------------
        self.track_2_left_down_coord == (None, None)
        self.track_2_left_up_coord == (None, None)
    
    # ---------------------------------------------------------------
    # RIGHT MOUSE CLICK
    # ---------------------------------------------------------------

    def track_1_right_down(self, event) -> None:
        self.track_1_right_down_coord = (event.x, event.y)

    def track_1_right_up(self, event) -> None:
        self.track_1_right_up_coord = (event.x, event.y)

        # -- incomplete action --------------------------------------
        if self.track_1_right_down_coord == (None, None) or self.track_1_right_up_coord == (None, None):
            pass

        # -- delete bar ---------------------------------------------
        elif self.track_1_right_down_coord == self.track_1_right_up_coord:
            x = self.app.convert_x_pos(event.x)
            x_bar_pos = self.app.bars_1.get_closest_bar(x=x)
            if x_bar_pos is None:
                print(f"No bar to delete from: {event.x} {event.y} | {x}")
            else:
                self.app.bars_1.delete_bar(x_bar_pos)
                
                self.app.view_1.delete_bar(x_bar_pos)
                self.app.view_3.delete_bar(x_bar_pos)

                print(f"A bar was deleted from: {event.x} {event.y} | {x} | {x_bar_pos}")
        else:
            pass

        # -- reset coord --------------------------------------------
        self.track_1_right_down_coord == (None, None)
        self.track_1_right_up_coord == (None, None)

    def track_2_right_down(self, event) -> None:
        self.track_2_right_down_coord = (event.x, event.y)
    
    def track_2_right_up(self, event) -> None:
        self.track_2_right_up_coord = (event.x, event.y)

        # -- incomplete action --------------------------------------
        if self.track_2_right_down_coord == (None, None) or self.track_2_right_up_coord == (None, None):
            pass

        # -- delete bar ---------------------------------------------
        elif self.track_2_right_down_coord == self.track_2_right_up_coord:
            x = self.app.convert_x_pos(event.x)
            x_bar_pos = self.app.bars_2.get_closest_bar(x=x)
            if x_bar_pos is None:
                print(f"No bar to delete from: {event.x} {event.y} | {x}")
            else:
                self.app.bars_2.delete_bar(x_bar_pos)
                
                self.app.view_2.delete_bar(x_bar_pos)
                self.app.view_4.delete_bar(x_bar_pos)
                self.app.view_5.delete_bar(x_bar_pos)

                print(f"A bar was deleted from: {event.x} {event.y} | {x} | {x_bar_pos}")
        else:
            pass

        # -- reset coord --------------------------------------------
        self.track_2_right_down_coord == (None, None)
        self.track_2_right_up_coord == (None, None)
    

class HoverEvents():
    def __init__(self, parent:App) -> None:
        # -- init class --
        self.app = parent

        # -- hover canvas [view_1] --
        self.app.view_1.canvas.get_tk_widget().bind("<Enter>", self.view_1_in)  # mouse pointer entered the widget
        self.app.view_1.canvas.get_tk_widget().bind("<Leave>", self.view_1_out) # mouse pointer left the widget

        # -- hover canvas [view_2] --
        self.app.view_2.canvas.get_tk_widget().bind("<Enter>", self.view_2_in)  # mouse pointer entered the widget
        self.app.view_2.canvas.get_tk_widget().bind("<Leave>", self.view_2_out) # mouse pointer left the widget

        # -- hover canvas [view_5] --
        self.app.view_5.canvas.get_tk_widget().bind("<Enter>", self.view_5_in)  # mouse pointer entered the widget
        self.app.view_5.canvas.get_tk_widget().bind("<Leave>", self.view_5_out) # mouse pointer left the widget

        # -- hover canvas [view_3] --
        self.app.view_3.canvas.get_tk_widget().bind("<Enter>", self.view_3_in)  # mouse pointer entered the widget
        self.app.view_3.canvas.get_tk_widget().bind("<Leave>", self.view_3_out) # mouse pointer left the widget

        # -- hover canvas [view_4] --
        self.app.view_4.canvas.get_tk_widget().bind("<Enter>", self.view_4_in)  # mouse pointer entered the widget
        self.app.view_4.canvas.get_tk_widget().bind("<Leave>", self.view_4_out) # mouse pointer left the widget

    # ---------------------------------------------------------------
    # HOVER FRAME
    # ---------------------------------------------------------------

    def view_1_in(self, event) -> None:
        self.app.view_1.set_bg_color(color=self.app.hover_color)

    def view_1_out(self, event) -> None:
        self.app.view_1.set_bg_color(color=(1.0, 1.0, 1.0))

    # ---------------------------------------------------------------

    def view_2_in(self, event) -> None:
        self.app.view_2.set_bg_color(color=self.app.hover_color)

    def view_2_out(self, event) -> None:
        self.app.view_2.set_bg_color(color=(1.0, 1.0, 1.0))

    # ---------------------------------------------------------------

    def view_5_in(self, event) -> None:
        self.app.view_5.set_bg_color(color=self.app.hover_color)

    def view_5_out(self, event) -> None:
        self.app.view_5.set_bg_color(color=(1.0, 1.0, 1.0))

    # ---------------------------------------------------------------

    def view_3_in(self, event) -> None:
        self.app.view_3.set_bg_color(color=self.app.hover_color)

    def view_3_out(self, event) -> None:
        self.app.view_3.set_bg_color(color=(1.0, 1.0, 1.0))

    # ---------------------------------------------------------------

    def view_4_in(self, event) -> None:
        self.app.view_4.set_bg_color(color=self.app.hover_color)

    def view_4_out(self, event) -> None:
        self.app.view_4.set_bg_color(color=(1.0, 1.0, 1.0))


class MusicPlayer():
    def __init__(self, parent:App) -> None:
        # -- init class --
        self.app = parent

        # -- init constants --
        self.track_loaded:bool = False           # the music file was loaded and is ready to be played
        self.track_length:Optional[float] = None # length of the audio is seconds
        self.play_button_pressed:bool = False    # was the play button pressed? (i.e. is the track active?) (required self.track_loaded)
        self.playing_state:bool = False          # is the track currently being played or paused? (required self.play_button_pressed)
        self.steps:float = 0.2                   # step size in seconds to display track progression in the wave plot

        self.slider_pos:Union[int,float] = 0      # current position of the track progression slider in the wave plot in seconds
        self.slider_pos_last:Union[int,float] = 0 # last position of the track progression slider in the wave plot in seconds
        self.override_time:bool = False           # ugly variable to avoid bug in pygame.mixer.music.get_pos() where first result after pausing is incorrect & should be dropped

        # -- init music --
        mixer.init()

        # -- ability to insert bars on current slider pos during playback --
        # # too much lag to be useful
        self.app.bind('<b>', self.insert_bar)

    def load_track(self) -> None:
        # -- clean up previous track --
        if self.play_button_pressed:
            self.stop()

        # -- loading track --
        if self.app.data_1.filename:
            mixer.music.load(self.app.data_1.filename) # 32-bit wav files are not supported !! use 16-bit wav files or mp3 instead !!
            self.get_track_len()
            self.track_loaded = True
        else:
            self.track_loaded = False
            print("No music file to play")

    def get_track_len(self) -> None:
        if self.app.data_1.filename:
            if self.app.data_1.filename[-4:] in [".mp3", ".wav", ".ogg"]:
                self.track_length = mixer.Sound(self.app.data_1.filename).get_length()
            else:
                print("get_track_len only supports mp3, wav and ogg")
        else:
            self.track_loaded = False
            print("No music file to get length of")

    def play(self) -> None:
        # -- load audiofile if not loaded yet --
        if self.track_loaded is False:
            try:
                self.app.mp.load_track()
            except Exception:
                if self.app.data_1.filename.endswith('.wav'):
                    messagebox.showinfo(title="Info", message="Use 16-bit wav files or mp3 files to enable audio playback.")
                else:
                    messagebox.showinfo(title="Info", message="Use wav or mp3 files to enable audio playback.")
                return None
        
        # -- play music --
        mixer.music.play(start=self.app.menubar.start_pos.get()) # Note: you can only start at a different position with mp3 & ogg files, NOT with wav files !!
        self.play_button_pressed = True
        self.playing_state = True
        self.playing()
    
    def playing(self) -> None:
        """
            This method calls itself repeatedly while the track is playing, to update various things based on track playback progression.
            This method does not support the playback itself, it is just used to obtain useful info.

            Bug:
                The mixer.music.get_pos method has a bug. The first get_pos call after unpausing track is incorrect, it returns the time we would be at if we hadn't paused !!
                To circumvent this bug, we use the last time before pausing when unpausing the track, instead of the current time.
        """
        # -- exit function if track is suddenly paused --
        if self.playing_state is False:
            return None

        # -- set position of last slider position --
        self.slider_pos_last = self.slider_pos
        if self.slider_pos_last is None:
            self.slider_pos_last = self.app.menubar.start_pos.get()
        
        # -- set position of current slider position --
        current_time = mixer.music.get_pos() / 1000 # seconds
        if self.override_time is True:
            self.slider_pos = self.slider_pos_last
        else:
            self.slider_pos = current_time + self.app.menubar.start_pos.get()
        self.override_time = False

        # -- exit function if track is over --
        if current_time == -0.001:
            print("End of track")
            self.stop()
            return None

        # -- print current timestamp --
        print(datetime.fromtimestamp(self.slider_pos).strftime('%M:%S.%f'))

        # -- insert the new bar --
        self.app.view_1.insert_bar(x=self.slider_pos, color='green')
        
        # -- remove the last bar --
        try:
            self.app.view_1.delete_bar(x=self.slider_pos_last)
        except Exception:
            pass

        # -- move view --
        if self.slider_pos > ((self.app.x_max - self.app.x_min) * 0.9 + self.app.x_min):
            self.app.menubar.scroll_right()

        # -- call the function again to update the slider --            
        self.app.after(int(self.steps*1000), self.playing)

    def pause(self) -> None:
        if self.play_button_pressed is False:
            self.play()
        elif self.playing_state is True:
            self.playing_state = False
            mixer.music.pause()
        elif self.playing_state is False:
            self.playing_state = True
            mixer.music.unpause() # track needs to be paused !! (play button pressed once, otherwise throws an exception)
            self.override_time = True
            self.playing()
        else:
            raise ValueError("Expecting a value of True or False for self.playing_state")
    
    def stop(self) -> None:
        # -- stop music player --
        mixer.music.stop()

        # -- remove the slider --
        try:
            self.app.view_1.delete_bar(x=self.slider_pos)
        except Exception:
            pass
        try:
            self.app.view_2.delete_bar(x=self.slider_pos)
        except Exception:
            pass
        try:
            self.app.view_1.delete_bar(x=self.slider_pos_last)
        except Exception:
            pass
        try:
            self.app.view_2.delete_bar(x=self.slider_pos_last)
        except Exception:
            pass
            
        # -- reset constants --
        self.play_button_pressed = False
        self.playing_state = False
        self.slider_pos = 0
    
    def adjust_volume(self) -> None:
        mixer.music.set_volume(self.app.menubar.volume.get()/100)

    # too much lag & reaction time to be useful (approx. 0.2 sec delay?)
    def insert_bar(self, event) -> None:
        if self.playing_state is True:
            x_pos = mixer.music.get_pos() / 1000 + self.app.menubar.start_pos.get() # seconds
            x_pos = max(0.2, x_pos - 0.2) # try to counter the lag (but may depend on the system)
            self.app.bars_1.insert_bar(x_pos)
            
            self.app.view_1.insert_bar(x_pos, 'blue')
            self.app.view_3.insert_bar(x_pos, 'blue')


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":
    # -- init app --
    app = App()

    # -- run app --
    app.mainloop()

