import pandas as pd
import statistics
import os

class gn_greenlight():
    def main(self):
        originaldf = pd.read_csv("GRACEnet Processing Report.csv")
        for index, row in originaldf.iterrows():
            print(row)
            if float(row["# Years"]) > 0.25:
                originaldf.loc[index, "Greenlight"] = "*"
            else:
                originaldf.loc[index, "Greenlight"] = ""
        originaldf.to_csv("GRACEnet Processing Report.csv", index=False)

if __name__ == '__main__':
    gn_greenlight = gn_greenlight()
    gn_greenlight.main()