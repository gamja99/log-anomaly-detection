import anom_gen as anomaly
import normal_gen as normal
import timestamp as ts
import csv

SALES = ["WAREHOUSES", "PRODUCTS", "DISTRIBUTORS", "ORDERS"]

def genLogs(n, csv_file):
    normal.genTimeStamps(n//2)
    anomaly.genanomalyTimeTimeStamps(n//4)
    anomaly.genanomalyDateTimeStamps(n//4)
    with open(csv_file, 'w') as file:
        csv_writer = csv.writer(file)
        row = ["System","User", "Operation", "Object", "Date", "Time"]
        csv_writer.writerow(row)

        for i in range(n//2):
            user, operation, obj = anomaly.genanomalyOperObjs()
            if obj in SALES:
                sys = "SALES"
            else:
                sys = "PM"

            # Gets the date string from previously generated timestamps
            date = ts.timeStamp.timestamps[i].getstrDate()

            # Gets the time string from previously generated timestamps
            time = ts.timeStamp.timestamps[i].getstrTime()

            row = [sys, user.name, operation, obj, date, time]
            csv_writer.writerow(row)

        for j in range(n//2, n):
            user, operation, obj = normal.genData()
            # Gets the date string from previously generated timestamps
            date = ts.timeStamp.timestamps[j].getstrDate()

            # Gets the time string from previously generated timestamps
            time = ts.timeStamp.timestamps[j].getstrTime()
            if obj in SALES:
                sys = "SALES"
            else:
                sys = "PM"
            row = [sys, user.name, operation, obj, date, time]
            csv_writer.writerow(row)
"""
genLogs(100000, "logs/normal_anomaly_logs.csv")
"""