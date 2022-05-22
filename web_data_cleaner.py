from __future__ import print_function
import random
import numpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import tkinter as tk
from tkinter import *
from tkcalendar import *
from tkinter import ttk


class data_cleaner(tk.Frame):
    def main(self):
        while 1:
            data_cleaner.input()
        filename = "unprocessed/CON Flux.csv"
        n20filename = filename
        datatype = ""
        if "flux" in filename.lower():
            datatype = "n2o_flux"
        elif "precip" in filename.lower():
            datatype = "precipitation_mm"
        elif "air" in filename.lower():
            datatype = "air_temp_c"
        elif "soil t" in filename.lower():
            datatype = "soil_temp_c"
        elif "wfps" in filename.lower():
            datatype = "soil_wfps"
        elif "smc" in filename.lower() or "vwc" in filename.lower():
            datatype = "soil_vwc"
        elif "nit" in filename.lower():
            datatype = "nitrogen_applied_kg"
        fluxdf = pd.read_csv(n20filename, header=None, names=['Date', 'Data'])
        predictordf = pd.read_csv(filename, header=None, names=['Date', datatype])
        originaldf = predictordf
        fluxstart = list(fluxdf.loc[:,"Date"])[0]
        fluxend = list(fluxdf.loc[:, "Date"])[-1]
        try:
            fluxstart = datetime.strptime(fluxstart, "%Y/%m/%d %H:%M")
            fluxend = datetime.strptime(fluxend, "%Y/%m/%d %H:%M")
        except:
            fluxstart = datetime.strptime(fluxstart, "%Y/%m/%d")
            fluxend = datetime.strptime(fluxend, "%Y/%m/%d")
        fluxstart = fluxstart.date()
        fluxstart = datetime.combine(fluxstart, datetime.min.time())
        fluxend = fluxend.date()
        fluxend = datetime.combine(fluxend, datetime.min.time())

        # Make sure predictor start and end dates match flux start and end dates
        predend = list(predictordf.loc[:, "Date"])[-1]
        predstart = list(predictordf.loc[:, "Date"])[0]
        print(predstart)
        print(predend)
        if predend != list(fluxdf.loc[:, "Date"])[-1]:
            dict = {'Date':[list(fluxdf.loc[:, "Date"])[-1]],datatype:['0']}
            df2 = pd.DataFrame(dict)
            predictordf = pd.concat([predictordf, df2], ignore_index=True)
        if predstart != list(fluxdf.loc[:, "Date"])[0]:
            dict = {'Date':[list(fluxdf.loc[:, "Date"])[0]],datatype:['0']}
            df2 = pd.DataFrame(dict)
            predictordf = pd.concat([predictordf, df2], ignore_index=True)
        print(predictordf)

        dailydf = data_cleaner.daily_avg(fluxstart, fluxend, predictordf, datatype)
        df_reindexed, df_plotting = data_cleaner.interpolator(fluxstart, fluxend, dailydf, datatype, fluxdf)
        data_cleaner.write_csv(df_reindexed, filename)
        shutil.move(filename, "original-files/")

        # Plot data before and after
        fig = plt.figure()
        for frame in [df_plotting, originaldf]:
            plt.plot(frame['Date'], frame[datatype])
        #plt.xlim(fluxstart, fluxend)
        plt.ylim(bottom=-1)
        #plt.yscale('log')
        #plt.show()

    def input(self):
        root = tk.Tk()
        root.title('Data Entry')
        window_width = 600
        window_height = 300
        # get the screen dimension
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        # find the center point
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        # set the position of the window to the center of the screen
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        frame = ttk.Frame(root)
        options = {'padx': 5, 'pady': 5}

        # Experiment ID
        id_label = ttk.Label(frame, text='Experiment ID')
        id_label.grid(column=0, row=0, sticky='W', **options)
        exp_id_var = tk.StringVar()
        id_input = ttk.Entry(frame, textvariable=exp_id_var)
        id_input.grid(column=1, row=0, **options)

        # N2O Units
        n2o_label = ttk.Label(frame, text='N2O Flux Units')
        n2o_label.grid(column=0, row=1, sticky='W', **options)
        n2o_units_var = tk.StringVar()
        r1 = ttk.Radiobutton(frame, text='g-N/ha/d', value='g-N/ha/d', variable=n2o_units_var)
        r2 = ttk.Radiobutton(frame, text='ug/m^2/hr', value='ug/m^2/hr', variable=n2o_units_var)
        r3 = ttk.Radiobutton(frame, text='Cumulative g-N/ha', value='cumulative', variable=n2o_units_var)
        r1.grid(column=0, row=2, **options)
        r2.grid(column=1, row=2, **options)
        r3.grid(column=2, row=2, **options)

        # NO3 Units
        n03_label = ttk.Label(frame, text='NO3 Units')
        n03_label.grid(column=0, row=3, sticky='W', **options)
        no3_units_var = tk.StringVar()
        r4 = ttk.Radiobutton(frame, text='NA', value='NA', variable=no3_units_var)
        r5 = ttk.Radiobutton(frame, text='mg-N/kg', value='mg-N/kg', variable=no3_units_var)
        r6 = ttk.Radiobutton(frame, text='kg-N/ha', value='kg-N/ha', variable=no3_units_var)
        r4.grid(column=0, row=4, **options)
        r5.grid(column=1, row=4, **options)
        r6.grid(column=2, row=4, **options)

        # NH4 Units
        nh4_label = ttk.Label(frame, text='NH4 Units')
        nh4_label.grid(column=0, row=5, sticky='W', **options)
        nh4_units_var = tk.StringVar()
        r7 = ttk.Radiobutton(frame, text='NA', value='NA', variable=nh4_units_var)
        r8 = ttk.Radiobutton(frame, text='mg-N/kg', value='mg-N/kg', variable=nh4_units_var)
        r9 = ttk.Radiobutton(frame, text='kg-N/ha', value='kg-N/ha', variable=nh4_units_var)
        r7.grid(column=0, row=6, **options)
        r8.grid(column=1, row=6, **options)
        r9.grid(column=2, row=6, **options)

        # Fertilizer Applications
        fert_label = ttk.Label(frame, text='Fertilizer Applications (#)')
        fert_label.grid(column=0, row=7, sticky='W', **options)
        fert_apps = tk.StringVar()
        fert_input = ttk.Entry(frame, textvariable=fert_apps)
        fert_input.grid(column=1, row=7, **options)

        # Planting Dates
        plant_label = ttk.Label(frame, text='Planting Dates (#)')
        plant_label.grid(column=0, row=8, sticky='W', **options)
        plant_apps = tk.StringVar()
        plant_input = ttk.Entry(frame, textvariable=fert_apps)
        plant_input.grid(column=1, row=8, **options)

        # Tillage Dates
        till_label = ttk.Label(frame, text='Tillage Dates (#)')
        till_label.grid(column=0, row=9, sticky='W', **options)
        till_apps = tk.StringVar()
        till_input = ttk.Entry(frame, textvariable=fert_apps)
        till_input.grid(column=1, row=9, **options)

        def enter_button_clicked():
            global exp_id
            exp_id = exp_id_var.get()
            global n2o_units
            n2o_units = n2o_units_var.get()
            global no3_units
            no3_units = no3_units_var.get()
            global nh4_units
            nh4_units = nh4_units_var.get()
            global fert_num
            fert_num = float(fert_input.get())
            global plant_num
            plant_num = float(plant_input.get())
            global till_num
            till_num = float(till_input.get())
            root.destroy()

        enter_button = ttk.Button(frame, text='Enter', command=enter_button_clicked)
        enter_button.grid(column=1, row=10, sticky='W', **options)

        # add padding to the frame and show it
        frame.grid(padx=10, pady=10)
        root.mainloop()


        if no3_units == "kg-N/ha" or nh4_units == "kg-N/ha":
            # Secondary Input Box
            root = tk.Tk()
            root.title('Data Entry')
            window_width = 500
            window_height = 300
            # get the screen dimension
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            # find the center point
            center_x = int(screen_width / 2 - window_width / 2)
            center_y = int(screen_height / 2 - window_height / 2)
            # set the position of the window to the center of the screen
            root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            frame = ttk.Frame(root)
            options = {'padx': 5, 'pady': 5}

            # Soil Bulk Density
            bd_label = ttk.Label(frame, text='Bulk Density (g/cm3)')
            bd_label.grid(column=0, row=0, sticky='W', **options)
            bulk_d = tk.StringVar()
            bd_input = ttk.Entry(frame, textvariable=bulk_d)
            bd_input.grid(column=1, row=0, **options)
            bd_input.focus()

            # Soil sample depth
            depth_label = ttk.Label(frame, text='Soil Sampling Depth (cm)')
            depth_label.grid(column=0, row=1, sticky='W', **options)
            sampling_depth = tk.StringVar()
            depth_input = ttk.Entry(frame, textvariable=sampling_depth)
            depth_input.grid(column=1, row=1, **options)
            depth_input.focus()
            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

            def second_enter_button_clicked():
                global bulk_density
                bulk_density = float(bulk_d.get())
                global sampling_depth_cm
                sampling_depth_cm = float(sampling_depth.get())
                root.destroy()

            enter_button = ttk.Button(frame, text='Enter', command=second_enter_button_clicked)
            enter_button.grid(column=1, row=10, sticky='W', **options)
        else:
            bulk_density = ''
            sampling_depth_cm = ''

        # Fertilizer Date Input
        global fert_dates
        fert_dates = []
        for num in np.arange(1, int(fert_num) + 1):
            root = tk.Tk()
            root.title('Fertilization Date #{}'.format(num))
            window_width = 500
            window_height = 200
            # get the screen dimension
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            # find the center point
            center_x = int(screen_width / 2 - window_width / 2)
            center_y = int(screen_height / 2 - window_height / 2)
            # set the position of the window to the center of the screen
            root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            frame = ttk.Frame(root)
            options = {'padx': 5, 'pady': 5}

            def date_enter():
                fert_date = dentry.get()
                fert_dates.append(fert_date)
                print(fert_dates)
                root.destroy()

            enter_button = ttk.Button(frame, text='Enter', command=date_enter)
            enter_button.grid(column=2, row=3, sticky='W', **options)

            dentry = DateEntry(root, font=('Helvetica', 40, tk.NORMAL), border=0)
            dentry.grid(column=3, row=0, sticky='W', **options)
            root.bind('<Return>', lambda e: print(dentry.get()))
            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

        # Planting Date Input
        global plant_dates
        plant_dates = []
        for num in np.arange(1, int(plant_num) + 1):
            root = tk.Tk()
            root.title('Planting Date #{}'.format(num))
            window_width = 500
            window_height = 200
            # get the screen dimension
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            # find the center point
            center_x = int(screen_width / 2 - window_width / 2)
            center_y = int(screen_height / 2 - window_height / 2)
            # set the position of the window to the center of the screen
            root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            frame = ttk.Frame(root)
            options = {'padx': 5, 'pady': 5}

            def plant_date_enter():
                plant_date = dentry.get()
                plant_dates.append(plant_date)
                print(plant_date)
                root.destroy()

            enter_button = ttk.Button(frame, text='Enter', command=plant_date_enter)
            enter_button.grid(column=2, row=3, sticky='W', **options)

            dentry = DateEntry(root, font=('Helvetica', 40, tk.NORMAL), border=0)
            dentry.grid(column=3, row=0, sticky='W', **options)
            root.bind('<Return>', lambda e: print(dentry.get()))
            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

        # Tillage Date Input
        global till_dates
        till_dates = []
        for num in np.arange(1, int(plant_num) + 1):
            root = tk.Tk()
            root.title('Tillage Date #{}'.format(num))
            window_width = 500
            window_height = 200
            # get the screen dimension
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            # find the center point
            center_x = int(screen_width / 2 - window_width / 2)
            center_y = int(screen_height / 2 - window_height / 2)
            # set the position of the window to the center of the screen
            root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            frame = ttk.Frame(root)
            options = {'padx': 5, 'pady': 5}

            def till_date_enter():
                till_date = dentry.get()
                till_dates.append(till_date)
                print(till_dates)
                root.destroy()

            enter_button = ttk.Button(frame, text='Enter', command=till_date_enter)
            enter_button.grid(column=2, row=3, sticky='W', **options)

            dentry = DateEntry(root, font=('Helvetica', 40, tk.NORMAL), border=0)
            dentry.grid(column=3, row=0, sticky='W', **options)
            root.bind('<Return>', lambda e: print(dentry.get()))
            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

        return exp_id, n2o_units, no3_units, nh4_units, bulk_density, sampling_depth_cm, fert_dates, \
               plant_dates, till_dates


    def __init__(self, master=None, frame_look={}, **look):
        root = Tk()
        root.withdraw()
        args = dict(relief=tk.SUNKEN, border=1)
        args.update(frame_look)
        tk.Frame.__init__(self, master, **args)

        args = {'relief': tk.FLAT}
        args.update(look)

        self.entry_1 = tk.Entry(self, width=2, **args)
        self.label_1 = tk.Label(self, text='/', **args)
        self.entry_2 = tk.Entry(self, width=2, **args)
        self.label_2 = tk.Label(self, text='/', **args)
        self.entry_3 = tk.Entry(self, width=4, **args)

        self.entry_1.pack(side=tk.LEFT)
        self.label_1.pack(side=tk.LEFT)
        self.entry_2.pack(side=tk.LEFT)
        self.label_2.pack(side=tk.LEFT)
        self.entry_3.pack(side=tk.LEFT)

        self.entries = [self.entry_1, self.entry_2, self.entry_3]

        self.entry_1.bind('<KeyRelease>', lambda e: self._check(0, 2))
        self.entry_2.bind('<KeyRelease>', lambda e: self._check(1, 2))
        self.entry_3.bind('<KeyRelease>', lambda e: self._check(2, 4))

    def _backspace(self, entry):
        cont = entry.get()
        entry.delete(0, tk.END)
        entry.insert(0, cont[:-1])

    def _check(self, index, size):
        entry = self.entries[index]
        next_index = index + 1
        next_entry = self.entries[next_index] if next_index < len(self.entries) else None
        data = entry.get()

        if len(data) > size or not data.isdigit():
            self._backspace(entry)
        if len(data) >= size and next_entry:
            next_entry.focus()

    def get(self):
        return [e.get() for e in self.entries]

    def daily_avg(self, fluxstart, fluxend, predictordf, datatype):
        delta = timedelta(days=1)
        date = fluxstart
        datetime_series = pd.to_datetime(predictordf['Date'])
        predictordf['Date'] = datetime_series
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        predictordf = predictordf.set_index(datetime_index)
        print(predictordf.head())
        daily_list = []
        while date <= fluxend:
            nextday = date + delta
            nextday = nextday - timedelta(seconds=1)
            day_data = predictordf[predictordf['Date'].between(date, nextday)]
            if day_data.empty:
                date += delta
                continue
            day_data = list(day_data.loc[:, datatype])
            print(day_data)
            day_data = [float(i) for i in day_data]
            day_average = statistics.mean(day_data)
            day_list = [date, day_average]
            daily_list.append(day_list)
            date += delta
        dailydf = pd.DataFrame(daily_list, columns=['Date', datatype])
        print(dailydf.head())
        return dailydf

    def interpolator(self, fluxstart, fluxend, dailydf, datatype, fluxdf):
        date = fluxstart
        delta = timedelta(days=1)
        datetime_series = pd.to_datetime(dailydf['Date'])
        dailydf['Date'] = datetime_series
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        dailydf = dailydf.set_index(datetime_index)
        if datatype == "precipitation_mm" or datatype == "nitrogen_applied_kg":
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="0")
            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            print(datadf.head())
            datadf[datatype] = datadf[datatype].astype(float)
        else:
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="NaN")
            print(df_reindexed.head())
            data_list = list(df_reindexed.loc[:,datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            print(datadf.head())
            datadf[datatype] = datadf[datatype].astype(float)
            datadf = datadf.interpolate(method='values').ffill().bfill()

        df_reindexed[datatype] = list(datadf.loc[:,datatype])
        df_plotting = df_reindexed
        df_plotting['Date'] = df_plotting.index
        print(df_plotting.head())
        print(df_reindexed.head())
        df_reindexed = df_reindexed.drop('Date', 1)
        df_reindexed.index.name = 'Date'
        print(df_reindexed.head())
        return df_reindexed, df_plotting

    def write_csv(self, df_reindexed, filename):
        df_reindexed.to_csv("processed/" + filename[12:-4] + "_cleaned.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_cleaner = data_cleaner()
    data_cleaner.main()


