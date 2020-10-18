# Imports
import timestamp as ts
import roles_and_users as ru
import random
# import n_generate as ng
import csv

SALES = ["WAREHOUSES", "PRODUCTS", "DISTRIBUTORS", "ORDERS"]


def genanomalyDateTimeStamps(n):
    """Generates n number of timestamps with anomaly dates and orders them by time"""
    # Creates n number of anomaly timestamp objects
    timestamps = ts.timeStamp.generatetimeStamp(n)

    # Adds a date and time for each anomaly timestamp object
    ## Checks to see if it is a weekday
    for timestamp in timestamps:
        # Which day of the week: 0 -> No value, 1-7 -> Days
        day_of_week = 0
        # Runs loop until created timestamp is on a weekend
        while day_of_week not in [6, 7]:
            # Creates a time stamp
            timestamp.createtimeStamp()
            # Finds what day of the week it is
            day_of_week = timestamp.getDayNum()
    return timestamps
    # Orders all the timestamps in chronological order
    # ts.timeStamp.ordertimeStamp()


def genanomalyTimeTimeStamps(n):
    """Generates n number of timestamps with anomaly times and orders them by time"""
    # Creates n number of anomaly timestamp objects
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
    # Adds a date and time for each anomaly timestamp object
    for timestamp in timestamps:
        # Which day of the week: 0 -> No value, 1-7 -> Days
        day_of_week = 0
        # Runs loop until created timestamp is on a weekday
        while day_of_week not in range(1, 6):
            # Creates a time stamp
            timestamp.createtimeStamp(True)
            # Finds what day of the week it is
            day_of_week = timestamp.getDayNum()
    return timestamps
    # Orders all the timestamps in chronological order
    # ts.timeStamp.ordertimeStamp()


def genanomalyOperObjs():
    role = ru.Role.getrandRole()
    role2 = ru.Role.getrandRole()

    while role == role2:
        role2 = ru.Role.getrandRole()
    user = role.getrandUser()
    operation, one_object = role2.getrandPrivilege(True)
    return user, operation, one_object


def genbruteLogin():
    num = random.randint(5, 10)
    # Generates a random user
    user = []
    operation = []
    obj = []
    while num != 0:
        user.append(ru.Role.getrandRole().getrandUser())
        operation.append("LOGON")
        obj.append("none")
        num -= 1
    return user, operation, obj
