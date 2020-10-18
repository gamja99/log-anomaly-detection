# Imports
import csv

# List of what the day of the week is for the first day in Jan-Aug
first_day = [3, 6, 7, 3, 5, 1, 3, 6]

# Class Log
class Log:
    # List of all log objects
    logs = []
    # Folder which holds csv files of list of user/operation/objects
    folder = "row_data/"
    folder2 = "operations_object/"

    def __init__(self, row):
        """Initialize object"""
        # Grabs all the information from a log
        self.system = row[0]
        self.user = row[1]
        self.operation = row[2]
        self.object = row[3]
        self.date = row[4]
        # Initializes the day of the week
        self.day_num = 0
        self.time = row[5]
        #self.result = row[6]
        # Adds object to list of objects
        Log.logs.append(self)
    
    def preProcess(self):
        """Pre-processes all the data from a log"""
        self.system = self.preProcessSystem()
        self.user = self.preProcessUser()
        self.operation = self.preProcessOperation()
        self.object = self.preProcessObject()
        self.preProcessDate()
        self.preProcessTime()

    def preProcessSystem(self):
        """Pre-processes the System column"""
        return 1 + self.preProcessFunction(self.system, Log.folder + "system.csv")
    
    def preProcessUser(self):
        """Pre-processes the User column"""
        return 1 + self.preProcessFunction(self.user, Log.folder + "users.csv")
    
    def preProcessOperation(self):
        """Pre-processes the Operation column"""
        return 1 + self.preProcessFunction(self.operation, Log.folder2 + "operations.csv")
    
    def preProcessObject(self):
        """Pre-processes the Object column"""
        # Checks to see if there is an object
        if self.object != "none":
            return 1 + self.preProcessFunction(self.object, Log.folder2 + "objects.csv")
        # Gives the object value None if there is no object for the operation
        return 0
    
    def preProcessDate(self):
        """Pre-processes the Date column"""
        # Gets the year, month, and day from the date part of timestamp
        print(self.date)
        year, month, day = self.date.split('-')
        # Combines them into a number which represents the date
        ## YearMonthDay
        self.year = int(year)
        self.month = int(month)
        self.day = int(day)
        self.date = int(year + month + day)
        # Loops through 0-7
        for i in range(8):
            # Checks to see what month it is
            if (i+1) == int(month):
                # Finds the day of the week for that specific timestamp
                self.day_num = ((int(day) - 1) + first_day[i]) %  7
                # Makes sure Sunday is properly represented by a 7
                if self.day_num == 0:
                    self.day_num = 7
                # Returns the new number representation of the date part of the timestamp and the day of the week
                return self.date, self.day_num
        # Returns 0, 0 if any errors occur
        return 0, 0

    def preProcessTime(self):
        """Pre-processes the Time column"""
        # Gets the hour, minute, and second from the time part of timestamp
        hour, minute, second = self.time.split(':')
        # Combines them into a number which represents the time
        ## HourMinuteSecond
        self.hour = int(hour)
        self.minute = int(minute)
        self.second = int(second)
        self.time = int(hour + minute + second)
        # Returns the new number representation of the time part of the timestamp
        return self.time

    def preProcessFunction(self, element, file):
        """Pre-process Function that assists with pre-processing some columns"""
        # Opens a certain file in read-only format
        with open(file, "r") as file:
            csv_reader = csv.reader(file)
            # Has an index counter
            self.index = 0
            for row in csv_reader:
                # Checks to see if the row's element matches the file's element (through loop)
                if row[0] == element:
                    # Assigns the numbered matched file element to represent element as a number
                    element = self.index
                # Increments index counter
                self.index += 1
            # Resets index counter for future use
            self.index = 0
            # Returns numbered element
            return element

    def getRow(self):
        """Returns the row"""
        return [self.system, self.user, self.operation, self.object, self.date, self.day_num, self.time]

    def getUserOperObjRow(self):
        """Returns the system, user, operation, object in a row"""
        return [self.system, self.user, self.operation, self.object]

    def gettimestampRow(self):
        """Returns the timestamp in a row format"""
        return [self.year, self.month, self.day, self.day_num, self.hour, self.minute, self.second, self.result]

    def getLargeRow(self):
        """Returns the larger row"""
        return [self.system, self.user, self.operation, self.object, self.year, self.month, self.day, self.day_num, self.hour, self.minute, self.second, self.result]

# Opens the normal logs that were generated in read-only format
with open("logs/normal_logs.csv", "r") as file:
    csv_reader = csv.reader(file)
    # Skips to the next line in the csv file
    next(csv_reader)
    # Loops through all the lines in the csv file
    for row in csv_reader:
        # Generates a log object and preprocesses it
        Log(row).preProcess()

# Creates a new file to hold the preprocessed normal logs
with open("logs/normal_preprocessed_logs.csv", "w") as file:
    csv_writer = csv.writer(file)
    # Column headers
    row = ["System", "User", "Operation", "Object", "Year", "Month", "Day" "DayNum", "Hour", "Minute", "Second"]
    # Writes the column headers
    csv_writer.writerow(row)
    # Loops through all the log objects
    for log in Log.logs:
        # Gets all the preprocessed data as a log row
        row = log.getLargeRow()
        # Writes the log row in the csv file
        csv_writer.writerow(row)
