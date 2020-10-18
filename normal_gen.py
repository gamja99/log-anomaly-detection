# Imports
import timestamp as ts
import roles_and_users as ru
import csv
import anom_gen as anomaly

SALES = ["WAREHOUSES", "PRODUCTS", "DISTRIBUTORS", "ORDERS"]

def genTimeStamps(n):
    """Generates n number of timestamps and orders them by time"""
    # Creates n number of timestamp objects
    timestamps = ts.timeStamp.generatetimeStamp(n)

    # Adds a date and time for each timestamp object
    ## Checks to see if it is a weekday
    for timestamp in timestamps:
        # Which day of the week: 0 -> No value, 1-7 -> Days
        day_of_week = 0
        # Runs loop until created timestamp is on a weekday
        while day_of_week not in range(1, 6):
            # Creates a time stamp
            timestamp.createtimeStamp()
            # Finds what day of the week it is
            day_of_week = timestamp.getDayNum()

    # Orders all the timestamps in chronological order
    ts.timeStamp.ordertimeStamp()


def genData():
    """Gets a random role, user, privilege pair (random operation and random respective object)"""
    # Get a random role object
    role_obj = ru.Role.getrandRole()
    
    # Get a random user from the chosen random role
    user_obj = role_obj.getrandUser()

    # Generates random operation and random object (from the set of corresponding objects) from the chosen random user
    operation, one_object = role_obj.getrandPrivilege(user_obj.login)
    
    # Updates the log in of that chosen random user to reflect which operation/object pairs can be chosen for the next time around
    user_obj.updateLogin(operation)
    
    # Returns the chosen user, chosen operation, chosen object
    return user_obj, operation, one_object

def createLogData(num_rows, location):
    """Creates the log data based on the provided number of rows"""
    # Generates num_rows number of timestamps
    results = []
    #anomaly.genanomalyTimeTimeStamps(num_rows // 4)
    #anomaly.genanomalyDateTimeStamps(num_rows // 4)
    genTimeStamps(num_rows)
    # Creates log_database csv file at location
    with open(location, "w") as file:
        csv_writer = csv.writer(file)

        # Column headers
        row = ["System","User", "Operation", "Object", "Date", "Time"]

        # Writes the column headers
        csv_writer.writerow(row)

        # Loops through the number of rows
        for i in range(num_rows):
            # Generates random user, operation, and object
            user_obj, operation, one_object = genData()

            # Generates the system
            if one_object in SALES:
                sys = "SALES"
            else:
                sys = "PM"

            # Gets the date string from previously generated timestamps
            date = ts.timeStamp.timestamps[i].getstrDate()

            # Gets the time string from previously generated timestamps
            time = ts.timeStamp.timestamps[i].getstrTime()
            print(len(results))
            if i < num_rows // 2:
                result = 1
            else:
                result = 0
            # Creates row with all the data above
            #row = [sys, user_obj.name, operation, one_object, date, time, result]
            row = [sys, user_obj.name, operation, one_object, date, time]
            #print(row)
            # Writes the row to the csv file
            csv_writer.writerow(row)

"""
# Calls the function that creates the csv file with given number of rows and given location of created file
createLogData(100000, "logs/normal_logs.csv")
"""