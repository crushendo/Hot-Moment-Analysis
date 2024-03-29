import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os

class gn_reporter():
    def main(self):
        originaldf = pd.read_csv("GRACEnet Processing Report.csv")
        greenlight = pd.DataFrame(columns=["Site", "Treatment", "Rep", "Series"])
        uploaded = pd.DataFrame(columns=["Site", "Treatment", "Rep", "Series"])
        green = originaldf[originaldf.Greenlight == "*"]
        upload = originaldf[originaldf.Uploaded == "*"]
        print(green)
        print(upload)
        report_df = pd.DataFrame(columns=["Site", "Treatment", "Rep", "Series", "Avg Flux", "Avg SMC", "Avg Soil T", "Avg Air T",
                                          "# Years", "# Ferts", "# Plantings", "# Harvests", "# Tills", "Greenlight",
                                          "Uploaded"])
        site_list = os.listdir("GRACEnet/")
        for site in site_list:
            treatment_list = os.listdir("GRACEnet/" + site)
            for treatment in treatment_list:
                rep_list = os.listdir("GRACEnet/" + site + "/" + treatment)
                for rep in rep_list:
                    series_list = os.listdir("GRACEnet/" + site + "/" + treatment + "/" + rep)
                    if len(series_list) == 0:
                        continue
                    if ".csv" in series_list[0]:
                        working_dir = "GRACEnet/" + site + "/" + treatment + "/" + rep + "/processed"
                        print(working_dir)
                        series = None
                        report_df = gn_reporter.reporter(working_dir, site, treatment, rep, series, report_df, green,
                                                         upload)
                    else:
                        for series in series_list:
                            working_dir = "GRACEnet/" + site + "/" + treatment + "/" + rep + "/" + series + "/processed"
                            print(working_dir)
                            report_df = gn_reporter.reporter(working_dir, site, treatment, rep, int(series), report_df,
                                                             green, upload)
        report_df.to_csv("GRACEnet Processing Report.csv", index=False)


    def reporter(self, working_dir, site, treatment, rep, series, report_df, green, upload):
        alldata = pd.read_csv(working_dir + "/alldata.csv")
        greencheck = green.query("Site == @site & Treatment == @treatment & Rep == @rep & Series == @series")
        print(greencheck)
        uploadcheck = upload.query("Site == @site & Treatment == @treatment & Rep == @rep & Series == @series")
        print(uploadcheck)
        if greencheck.shape[0] > 0:
            greenval = "*"
        else:
            greenval = None
        if uploadcheck.shape[0] > 0:
            uploadval = "*"
        else:
            uploadval = None
        try:
            flux = list(alldata.loc[:, "n2o_flux"])
            avg_flux = statistics.mean(flux)
        except:
            avg_flux = None
        try:
            vwc = list(alldata.loc[:, "soil_vwc"])
            avg_vwc = statistics.mean(vwc)
        except:
            avg_vwc = None
            pass
        try:
            soil_t = list(alldata.loc[:, "soil_temp_c"])
            avg_soil_t = statistics.mean(soil_t)
        except:
            avg_soil_t = None
            pass
        try:
            air_t = list(alldata.loc[:, "air_temp_c"])
            avg_air_t = statistics.mean(air_t)
        except:
            avg_air_t = None
            pass
        years = len(alldata.index) / 365
        fert_num = alldata["nitrogen_form"].notnull().sum()
        plant_num = alldata["planted_crop"].notnull().sum()
        till_num = alldata[alldata.mgmt == "tillage"].shape[0]
        harvest_num = alldata[alldata.mgmt == "harvest"].shape[0]
        addrow = pd.Series([site, treatment, rep, series, avg_flux, avg_vwc, avg_soil_t, avg_air_t,
                            years, fert_num, plant_num, harvest_num, till_num, greenval, uploadval], index=report_df.columns)
        report_df = report_df.append(addrow, ignore_index=True)
        return report_df



if __name__ == '__main__':
    gn_reporter = gn_reporter()
    gn_reporter.main()