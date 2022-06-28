from __future__ import print_function
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
from src.data.db_conn import load_db_table


class data_cleaner(tk.Frame):
    def main(self):
        nitrogen_formdf = load_db_table(config_db='database.ini', query='SELECT nitrogen_form FROM "DailyPredictors"')
        nitrogen_formdf['nitrogen_form'] = nitrogen_formdf['nitrogen_form'].str.lower()
        n_forms = nitrogen_formdf.nitrogen_form.unique()
        n_forms = np.append(n_forms, "other")
        print(n_forms)
        exp_id, n2o_units, no3_units, nh4_units, bulk_density, sampling_depth_cm, fert_dates, \
            plant_dates, till_dates, harvest_dates, plant_crops = data_cleaner.input(n_forms)
        dir_list = os.listdir("unprocessed/")
        for filename in dir_list:
            filename = "unprocessed/" + filename
            n20filename = filename
            datatype = ""
            if "flux" in filename.lower() or "n2o" in filename.lower():
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
                datatype = "nitrogen_applied_kg_ha"
            elif "no3" in filename.lower():
                datatype = "no3_mg_n_kg"
            elif "nh4" in filename.lower():
                datatype = "nh4_mg_n_kg"
            print(datatype)
            fluxdf = pd.read_csv(n20filename, header=None, names=['Date', 'Data'])
            predictordf = pd.read_csv(filename, header=None, names=['Date', datatype])
            if predictordf["Date"].iloc[0] == "Date" or predictordf["Date"].iloc[0] == " Date":
                predictordf = predictordf.iloc[1:]
            predictordf.sort_values(by='Date', inplace=True)
            predictordf = predictordf.reset_index(drop=True)
            originaldf = predictordf

            # Make sure predictor start and end dates match flux start and end dates
            predend = list(predictordf.loc[:, "Date"])[-1]
            predstart = list(predictordf.loc[:, "Date"])[0]
            print(predstart)
            print(predend)

            print(predictordf)

            # Make sure data is sorted in chronological order
            predictordf['Date'] = pd.to_datetime(predictordf['Date'])
            predictordf.sort_values(by='Date')

            dailydf = data_cleaner.daily_avg(predictordf, datatype)
            df_reindexed, df_plotting = data_cleaner.interpolator(dailydf, datatype, fluxdf)
            data_cleaner.write_csv(df_reindexed, filename)
            try:
                shutil.move(filename, "original-files/")
            except:
                pass

        # Combine into single file
        dir_list = os.listdir("processed/")
        print(dir_list)
        fulldf = ''
        for filename in dir_list:
            filename = "processed/" + filename
            df = pd.read_csv(filename)
            print(df.head())
            if isinstance(fulldf, str):
                fulldf = df
            else:
                fulldf = fulldf.merge(df, how='inner', on=['Date'])
        print(fulldf.head())

        # Perform unit conversions
        fulldf = data_cleaner.unit_conversion(fulldf)

        # Add management and fertilization columns
        print(fert_quantities)
        print(fert_forms)
        fulldf = data_cleaner.add_management(fulldf)
        print(fulldf.head())

        # Reorganize column order
        db_columns = ['experiment_id', 'Date', 'n2o_flux', 'soil_vwc', 'soil_wfps', 'soil_temp_c', 'air_temp_c',
                      'precipitation_mm', 'nitrogen_applied_kg_ha', 'nitrogen_form', 'mgmt', 'nh4_mg_n_kg',
                      'no3_mg_n_kg', 'planted_crop']
        df_columns = fulldf.columns
        shared_columns = []
        for column in db_columns:
            if column in df_columns:
                shared_columns.append(column)
        print(shared_columns)
        fulldf = fulldf.reindex(columns=shared_columns)
        print(fulldf.head())

        # Write to CSV
        fulldf.to_csv("processed/alldata.csv")

    def unit_conversion(self, fulldf):
        # N2O conversion function
        print(n2o_units)
        if n2o_units == "ug/m^2/hr":
            b = fulldf[["n2o_flux"]].apply(lambda a: a * 0.24)
            print(b)
            fulldf.drop("n2o_flux", axis=1, inplace=True)
            fulldf["n2o_flux"] = b
        if no3_units == "kg-N/ha":
            b = fulldf[["no3_mg_n_kg"]].apply(lambda a: a / (float(bulk_density) * float(sampling_depth_cm / 100) * 10))
            print(b)
            fulldf.drop("no3_mg_n_kg", axis=1, inplace=True)
            fulldf["no3_mg_n_kg"] = b
        if nh4_units == "kg-N/ha":
            b = fulldf[["nh4_mg_n_kg"]].apply(lambda a: a / (float(bulk_density) * float(sampling_depth_cm / 100) * 10))
            print(b)
            fulldf.drop("nh4_mg_n_kg", axis=1, inplace=True)
            fulldf["nh4_mg_n_kg"] = b
        if fulldf.soil_wfps[0] > 1:
            b = fulldf[["soil_wfps"]].apply(lambda a: a / 100)
            fulldf.drop("soil_wfps", axis=1, inplace=True)
            fulldf["soil_wfps"] = b
        return fulldf

    def add_management(self, fulldf):
        fulldf = fulldf.assign(experiment_id=exp_id)
        fulldf = fulldf.assign(mgmt=None)
        fulldf = fulldf.assign(nitrogen_form=None)
        fulldf = fulldf.assign(nitrogen_applied_kg_ha=0)
        fulldf = fulldf.assign(planted_crop=None)
        print(fulldf.head())
        for date in till_dates:
            if date[1] == "/":
                date = "0" + date
            date = datetime.strptime(date, "%m/%d/%y")
            date = date.strftime("%Y-%m-%d")
            date = str(date)
            index_list = fulldf.query("Date == @date").index.tolist()
            fulldf.at[index_list[0], 'mgmt'] = "tillage"
        i = 0
        for date in plant_dates:
            print(date)
            if date[1] == "/":
                date = "0" + date
            date = datetime.strptime(date, "%m/%d/%y")
            date = date.strftime("%Y-%m-%d")
            date = str(date)
            index_list = fulldf.query("Date == @date").index.tolist()
            fulldf.at[index_list[0], 'mgmt'] = "planting"
            fulldf.at[index_list[0], 'planted_crop'] = plant_crops[i]
            i += 1
        i = 0
        for date in fert_dates:
            if date[1] == "/":
                date = "0" + date
            date = datetime.strptime(date, "%m/%d/%y")
            date = date.strftime("%Y-%m-%d")
            date = str(date)
            try:
                index_list = fulldf.query("Date == @date").index.tolist()
                fulldf.at[index_list[0], 'mgmt'] = "fertilizer"
                fulldf.at[index_list[0], 'nitrogen_applied_kg_ha'] = fert_quantities[i]
                fulldf.at[index_list[0], 'nitrogen_form'] = fert_forms[i]
            except:
                pass
            i += 1
        for date in harvest_dates:
            if date[1] == "/":
                date = "0" + date
            date = datetime.strptime(date, "%m/%d/%y")
            date = date.strftime("%Y-%m-%d")
            date = str(date)
            index_list = fulldf.query("Date == @date").index.tolist()
            fulldf.at[index_list[0], 'mgmt'] = "harvest"
        return fulldf

    def input(self, n_forms):
        root = tk.Tk()
        root.title('Data Entry')
        window_width = 600
        window_height = 400
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
        def selection1():
            n2o_units_var.set("g-N/ha/d")
        def selection2():
            n2o_units_var.set("ug/m^2/hr")
        def selection3():
            n2o_units_var.set("Cumulative g-N/ha")
        n2o_label = ttk.Label(frame, text='N2O Flux Units')
        n2o_label.grid(column=0, row=1, sticky='W', **options)
        n2o_units_var = tk.StringVar()
        r1 = ttk.Radiobutton(frame, text='g-N/ha/d', value=1, variable=n2o_units_var, command=selection1)
        r2 = ttk.Radiobutton(frame, text='ug/m^2/hr', value=2, variable=n2o_units_var, command=selection2)
        r3 = ttk.Radiobutton(frame, text='Cumulative g-N/ha', value=3, variable=n2o_units_var, command=selection3)
        r1.grid(column=0, row=2, **options)
        r2.grid(column=1, row=2, **options)
        r3.grid(column=2, row=2, **options)

        # NO3 Units
        def selection1():
            no3_units_var.set("NA")
        def selection2():
            no3_units_var.set("mg-N/kg")
        def selection3():
            no3_units_var.set("kg-N/ha")
        no3_label = ttk.Label(frame, text='NO3 Units')
        no3_label.grid(column=0, row=3, sticky='W', **options)
        no3_units_var = tk.StringVar()
        r4 = ttk.Radiobutton(frame, text='NA', value='NA', variable=no3_units_var, command=selection1)
        r5 = ttk.Radiobutton(frame, text='mg-N/kg', value='mg-N/kg', variable=no3_units_var, command=selection2)
        r6 = ttk.Radiobutton(frame, text='kg-N/ha', value='kg-N/ha', variable=no3_units_var, command=selection3)
        r4.grid(column=0, row=4, **options)
        r5.grid(column=1, row=4, **options)
        r6.grid(column=2, row=4, **options)

        # NH4 Units
        def selection1():
            nh4_units_var.set("NA")
        def selection2():
            nh4_units_var.set("mg-N/kg")
        def selection3():
            nh4_units_var.set("kg-N/ha")
        nh4_label = ttk.Label(frame, text='NH4 Units')
        nh4_label.grid(column=0, row=5, sticky='W', **options)
        nh4_units_var = tk.StringVar()
        r7 = ttk.Radiobutton(frame, text='NA', value='NA', variable=nh4_units_var, command=selection1)
        r8 = ttk.Radiobutton(frame, text='mg-N/kg', value='mg-N/kg', variable=nh4_units_var, command=selection2)
        r9 = ttk.Radiobutton(frame, text='kg-N/ha', value='kg-N/ha', variable=nh4_units_var, command=selection3)
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
        plant_input = ttk.Entry(frame, textvariable=plant_apps)
        plant_input.grid(column=1, row=8, **options)

        # Tillage Dates
        till_label = ttk.Label(frame, text='Tillage Dates (#)')
        till_label.grid(column=0, row=9, sticky='W', **options)
        till_apps = tk.StringVar()
        till_input = ttk.Entry(frame, textvariable=till_apps)
        till_input.grid(column=1, row=9, **options)

        # Harvest Dates
        harvest_label = ttk.Label(frame, text='Harvest Dates (#)')
        harvest_label.grid(column=0, row=10, sticky='W', **options)
        harvest_apps = tk.StringVar()
        harvest_input = ttk.Entry(frame, textvariable=harvest_apps)
        harvest_input.grid(column=1, row=10, **options)

        global exp_id
        global n2o_units
        global no3_units
        global nh4_units
        global fert_num
        global plant_num
        global till_num
        global harvest_num

        def enter_button_clicked():
            global exp_id
            exp_id = id_input.get()
            global n2o_units
            n2o_units = n2o_units_var.get()
            print(n2o_units)
            global no3_units
            no3_units = no3_units_var.get()
            print(no3_units)
            global nh4_units
            nh4_units = nh4_units_var.get()
            global fert_num
            fert_num = float(fert_input.get())
            global plant_num
            plant_num = float(plant_input.get())
            global till_num
            till_num = float(till_input.get())
            global harvest_num
            harvest_num = float(harvest_input.get())
            root.destroy()
            root.quit()

        enter_button = ttk.Button(frame, text='Enter', command=enter_button_clicked)
        enter_button.grid(column=1, row=11, sticky='W', **options)

        # add padding to the frame and show it
        frame.grid(padx=10, pady=10)
        root.mainloop()

        global bulk_density
        global sampling_depth_cm
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

            def second_enter_button_clicked():
                global bulk_density
                bulk_density = float(bd_input.get())
                print(bulk_density)
                global sampling_depth_cm
                sampling_depth_cm = float(depth_input.get())
                root.destroy()
                root.quit()

            enter_button = ttk.Button(frame, text='Enter', command=second_enter_button_clicked)
            enter_button.grid(column=1, row=10, sticky='W', **options)

            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

        else:
            bulk_density = ''
            sampling_depth_cm = ''

        # Fertilizer Date Input
        global fert_dates
        global fert_quantities
        global fert_forms
        fert_dates = []
        fert_quantities = []
        fert_forms = []
        for num in np.arange(1, int(fert_num) + 1):
            root = tk.Tk()
            root.title('Fertilization Date #{}'.format(num))
            window_width = 600
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
                app_quant_i = quant_input.get()
                fert_quantities.append(app_quant_i)
                fert_form = form_var.get()
                print(fert_form)
                if fert_form == "other":
                    fert_form = other_form_input.get()
                fert_forms.append(fert_form)
                root.destroy()
                root.quit()

            dentry = DateEntry(root, font=('Helvetica', 40, tk.NORMAL), border=0)
            dentry.grid(column=0, row=0, sticky='W', **options)
            root.bind('<Return>', lambda e: print(dentry.get()))

            # Fertilizer Quantities
            app_label = ttk.Label(frame, text='Fertilizer Application (kg-N/ha)')
            app_label.grid(column=0, row=4, sticky='W', **options)
            app_quant = tk.StringVar()
            quant_input = ttk.Entry(frame, textvariable=app_quant)
            quant_input.grid(column=1, row=4, sticky='W', **options)

            # Nitrogen Form dropdown
            global hidden
            hidden = True
            form_var = tk.StringVar(root)
            form_var.set('urea')
            popupMenu = ttk.OptionMenu(frame, form_var, *n_forms)
            ttk.Label(frame, text="Fertilizer Form").grid(row=5, column=0)
            popupMenu.grid(row=5, column=1)
            other_form_var = tk.StringVar()
            other_form_input = ttk.Entry(frame, textvariable=other_form_var)
            # on change dropdown value
            def change_dropdown(*args):
                print(form_var.get())
                fert_form = form_var.get()
                if fert_form == "other":
                    other_form_input.grid(row=6, column=0)
                else:
                    other_form_input.grid_remove()
            # link function to change dropdown
            form_var.trace('w', change_dropdown)

            # Enter button
            enter_button = ttk.Button(frame, text='Enter', command=date_enter)
            enter_button.grid(column=2, row=6, sticky='W', **options)

            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

        # Planting Date Input
        global plant_dates
        plant_dates = []
        global plant_crops
        plant_crops = []
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

            # Planted crop
            app_label = ttk.Label(frame, text='Planted Crop')
            app_label.grid(column=0, row=4, sticky='W', **options)
            planted_crop_var = tk.StringVar()
            crop_input = ttk.Entry(frame, textvariable=planted_crop_var)
            crop_input.grid(column=1, row=4, sticky='W', **options)

            def plant_date_enter():
                plant_date = dentry.get()
                plant_dates.append(plant_date)
                print(plant_date)
                planted_crop = crop_input.get()
                plant_crops.append(planted_crop)
                root.destroy()
                root.quit()

            enter_button = ttk.Button(frame, text='Enter', command=plant_date_enter)
            enter_button.grid(column=2, row=5, sticky='W', **options)

            dentry = DateEntry(root, font=('Helvetica', 40, tk.NORMAL), border=0)
            dentry.grid(column=0, row=0, sticky='W', **options)
            root.bind('<Return>', lambda e: print(dentry.get()))
            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

        # Tillage Date Input
        global till_dates
        till_dates = []
        for num in np.arange(1, int(till_num) + 1):
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
                root.quit()

            enter_button = ttk.Button(frame, text='Enter', command=till_date_enter)
            enter_button.grid(column=2, row=3, sticky='W', **options)

            dentry = DateEntry(root, font=('Helvetica', 40, tk.NORMAL), border=0)
            dentry.grid(column=0, row=0, sticky='W', **options)
            root.bind('<Return>', lambda e: print(dentry.get()))
            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

        # Harvest Date Input
        global harvest_dates
        harvest_dates = []
        for num in np.arange(1, int(harvest_num) + 1):
            root = tk.Tk()
            root.title('Harvest Date #{}'.format(num))
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

            def harvest_date_enter():
                harvest_date = dentry.get()
                harvest_dates.append(harvest_date)
                print(harvest_dates)
                root.destroy()
                root.quit()

            enter_button = ttk.Button(frame, text='Enter', command=harvest_date_enter)
            enter_button.grid(column=2, row=3, sticky='W', **options)

            dentry = DateEntry(root, font=('Helvetica', 40, tk.NORMAL), border=0)
            dentry.grid(column=3, row=0, sticky='W', **options)
            root.bind('<Return>', lambda e: print(dentry.get()))
            # add padding to the frame and show it
            frame.grid(padx=10, pady=10)
            root.mainloop()

        return exp_id, n2o_units, no3_units, nh4_units, bulk_density, sampling_depth_cm, fert_dates, \
               plant_dates, till_dates, harvest_dates, plant_crops


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

    def daily_avg(self, predictordf, datatype):
        fluxstart = predictordf.iloc[:1]
        fluxstart = list(fluxstart.Date)
        fluxstart = fluxstart[0]
        date = fluxstart
        iterdate = fluxstart.date()
        my_time = datetime.min.time()
        enddate = predictordf.iloc[-1:]
        enddate = list(enddate.Date)
        enddate = enddate[0].date()
        delta = timedelta(days=1)
        datetime_series = pd.to_datetime(predictordf['Date'])
        predictordf['Date'] = datetime_series
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        predictordf = predictordf.set_index(datetime_index)
        daily_list = []
        while iterdate <= enddate:
            nextday = iterdate + delta
            my_time = datetime.min.time()
            nextday = datetime.combine(nextday, my_time)
            iterdatetime = datetime.combine(iterdate, my_time)
            nextday = nextday - timedelta(seconds=1)
            # Get all data points occuring on a single day
            print(date)
            print(nextday)
            day_data = predictordf[predictordf['Date'].between(iterdatetime, nextday)]
            day_df = day_data
            if day_data.empty:
                iterdate += delta
                date += delta
                continue
            # Retreive only measurement data as a list
            day_data = list(day_data.loc[:, datatype])
            day_data = [float(i) for i in day_data]

            # Find the average value of measurements on that day
            start_second = day_df.iloc[:1]
            start_second = list(start_second.Date)[0]
            end_second = day_df.iloc[-1:]
            end_second = list(end_second.Date)[0]
            # If data has only a date stamp, average all values
            if start_second == end_second:
                day_average = statistics.mean(day_data)
                day_list = [iterdate, day_average]
            # If full timestamp is available, find the time-weighted average
            else:
                seconds_list = []
                iter_sec = start_second
                print("HERE")
                print(start_second)
                print(end_second)
                while iter_sec <= end_second:
                    seconds_list.append(iter_sec)
                    iter_sec = iter_sec + timedelta(seconds=1)
                secondsdf = pd.DataFrame(seconds_list, columns=["Date"])
                print(secondsdf.head())
                day_df = day_df.merge(secondsdf, how='outer', on=['Date'])
                day_df.sort_values(by='Date', inplace=True)
                day_df = day_df.reset_index(drop=True)
                print(day_df)
                day_df[datatype] = day_df[datatype].astype(float)
                day_df[datatype].interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)
                daily_average = day_df[datatype].mean()
                day_list = [iterdate, daily_average]
                print(day_list)

            # Update new list of average daily values
            daily_list.append(day_list)
            date += delta
            iterdate += delta

        # Convert daily values list into dataframe
        dailydf = pd.DataFrame(daily_list, columns=['Date', datatype])
        print(dailydf.head())
        return dailydf

    def interpolator(self, dailydf, datatype, fluxdf):
        delta = timedelta(days=1)
        sample_dates = list(dailydf.loc[:, 'Date'])
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
            datadf[datatype] = datadf[datatype].astype(float)
        elif datatype == "nh4_mg_n_kg" or datatype == "no3_mg_n_kg":
            # for nitrogen sample data, if fertilizer has been applied between two data points, interpolate
            # as a stepwise function between those datapoints around the fertilization date
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="NaN")
            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            datadf[datatype] = datadf[datatype].astype(float)
            print(df_reindexed.head())
            print(sample_dates)
            for i in np.arange(0, len(sample_dates) - 1):
                n_date = sample_dates[i]
                np1_date = sample_dates[i+1]
                for date in fert_dates:
                    date = datetime.strptime(date, "%m/%d/%y")
                    date= date.date()
                    if date > n_date and date < np1_date:
                        print("here")
                        low_value = df_reindexed.query("Date == @n_date")[datatype].tolist()[0]
                        high_value = df_reindexed.query("Date == @np1_date")[datatype].tolist()[0]
                        df_reindexed.loc[n_date:date, datatype] = low_value
                        df_reindexed.loc[date:np1_date, datatype] = high_value

            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            datadf[datatype] = datadf[datatype].astype(float)
            datadf = datadf.interpolate(method='values').ffill().bfill()
            print(datadf)
        else:
            df_reindexed = dailydf.reindex(pd.date_range(start=dailydf.index.min(),
                                                         end=dailydf.index.max(),
                                                         freq='1D'), fill_value="NaN")
            data_list = list(df_reindexed.loc[:, datatype])
            datadf = pd.DataFrame(data_list, columns=[datatype])
            datadf[datatype] = datadf[datatype].astype(float)
            datadf = datadf.interpolate(method='values').ffill().bfill()

        df_reindexed[datatype] = list(datadf.loc[:,datatype])
        df_plotting = df_reindexed
        df_plotting['Date'] = df_plotting.index
        df_reindexed = df_reindexed.drop('Date', 1)
        df_reindexed.index.name = 'Date'
        return df_reindexed, df_plotting

    def write_csv(self, df_reindexed, filename):
        df_reindexed.to_csv("processed/" + filename[12:-4] + "_cleaned.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_cleaner = data_cleaner()
    data_cleaner.main()


