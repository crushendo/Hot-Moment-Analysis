import openpyxl
import csv
import os

def csv_from_excel():
    os.chdir("unprocessed/")
    dir_list = os.listdir()
    for filename in dir_list:
        if filename[-1] == "v":
            continue
        print(filename)
        wb = openpyxl.load_workbook(filename).active
        your_csv_file = csv.writer(open(filename[:-5] + ".csv", 'w', newline = ""))
        for r in wb.rows:
            row = [a.value for a in r]
            your_csv_file.writerow(row)
        os.remove(filename)

# runs the csv_from_excel function:
csv_from_excel()