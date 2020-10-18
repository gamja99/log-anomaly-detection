# Imports
import random
import operator

# List of months
months = ["January", "February", "March", "April", "May", "June", "July", "August"]
first_day = [3, 6, 7, 3, 5, 1, 3, 6]

# Class for TimeStamps
class timeStamp:
    # Initialize list of class objects
    timestamps = []

    def __init__(self):
        """Initializes object"""
        # Adds new object to list of objects
        timeStamp.timestamps.append(self)
        # Initialize some values for timestamp
        self.year = 2020
        self.month = 0
        self.day = 0
        self.day_num = 0

    def createtimeStamp(self, anomaly=False):
        """Generates all the necessary information from one method"""
        self.createMonth()
        self.createDay()
        if anomaly:
            self.createanomalyTime()
        else:
            self.createTime()
        self.createtimestampNum()


    def createMonth(self):
        """Generates a random month from: January, February, March, April, May, June, July, August"""
        # Random number from (1, 8)
        self.month = random.randint(1, 8)

    def getnumMonth(self):
        """Returns the 2-digit string representation of the integer month"""
        # If it is a one-digit number, then you add a 0 to the front and return the string representation
        if self.month == 0:
            return "You need to run the createMonth method first"
        if len(str(self.month)) == 1:
            return "0" + str(self.month)
        # If it is a two-digit number, return the string representation of the integer
        else:
            return str(self.month)

    def getstrMonth(self):
        """Returns the month from the integer representation"""
        # Returns the month from the integer representation
        return months[self.month - 1]

    def createDay(self):
        """Generates a random day based on the month"""
        # If a random month has not been generated
        if self.month == 0:
            return "You need to run the createMonth method first"
        # If the month is February
        elif self.month == 2:
            self.day = random.randint(1, 29)
        # If the month has 30 days
        elif self.month in [4, 6]:
            self.day = random.randint(1, 30)
        # If the month has 31 days
        else:
            self.day = random.randint(1, 31)

    def getnumDay(self):
        """Returns the 2-digit string representation of the integer day"""
        # If it is a one-digit number, then you add a 0 to the front and return the string representation
        if self.day == 0:
            return "You need to run the createDay method first"
        if len(str(self.day)) == 1:
            return "0" + str(self.day)
        # If it is a two-digit number, return the string representation of the integer
        else:
            return str(self.day)

    def getstrDate(self):
        """Returns the string date representation"""
        return '%04d-%02d-%02d' % (self.year, self.month, self.day)

    def createTime(self):
        """Generates a random hour, minute, second"""
        # Total seconds: 8 hours from 9am-5pm, 60 minutes per hour, 60 seconds per minute
        self.t_seconds = 8*60*60
        # Generate a random number of seconds from 0 to t_seconds
        self.r_time = int(random.random() * self.t_seconds)
        # Gets the hour
        self.hour = int(self.r_time / 3600)
        # Gets the number of minutes
        self.minute = int((self.r_time - (self.hour * 3600)) / 60)
        # Gets the number of seconds
        self.second = (self.r_time - (self.hour * 3600) - (self.minute * 60))
        # Updates the hour to reflect actual time from 9am-5pm
        self.hour += 9

    def createanomalyTime(self):
        """Generates an anomaly time: hour, minute, and second"""
        # Total seconds: 16 hours from 5pm - 9am, 60 minutes per hour, 60 seconds per minute
        self.t_seconds = 16*60*60
        # Generate a random number of seconds from 0 to t_seconds
        self.r_time = int(random.random() * self.t_seconds)
        # Gets the hour
        self.hour = int(self.r_time / 3600)
        # Gets the number of minutes
        self.minute = int((self.r_time - (self.hour * 3600)) / 60)
        # Gets the number of seconds
        self.second = (self.r_time - (self.hour * 3600) - (self.minute * 60))
        # Updates the hour to reflect actual time from 5pm-9am
        self.hour = (self.hour + 17) % 24

    def getstrSecond(self):
        """Returns the 2-digit string representation of integer second"""
        # If it is a one-digit number, then you add a 0 to the front and return the string representation
        if len(str(self.second)) == 1:
            return "0" + str(self.second)
        # If it is a two-digit number, return the string representation of the integer
        else:
            return str(self.second)
    
    def getstrMinute(self):
        """Returns the 2-digit string representation of integer minute"""
        # If it is a one-digit number, then you add a 0 to the front and return the string representation
        if len(str(self.minute)) == 1:
            return "0" + str(self.minute)
        # If it is a two-digit number, return the string representation of the integer
        else:
            return str(self.minute)

    def getstrHour(self):
        """Returns the 2-digit string representation of integer hour"""
        # If it is a one-digit number, then you add a 0 to the front and return the string representation
        if len(str(self.hour)) == 1:
            return "0" + str(self.hour)
        # If it is a two-digit number, return the string representation of the integer
        else:
            return str(self.hour)

    def getstrTime(self):
        """Returns the string time representation"""
        return '%02d:%02d:%02d' % (self.hour, self.minute, self.second)
    
    def getstrtimeStamp(self):
        """Returns the string timestamp representation"""
        return self.getstrDate() + ' ' + self.getstrTime()
    
    def getDayNum(self):
        """Returns what day of the week it is"""
        # If the month has not be generated yet, it will return False
        if self.month == 0:
            return False
        # Loops through the possible months
        for i in range(len(months)):
            # Finds which month it is
            if (i+1) == self.month:
                # Finds what day of the week it is
                self.day_num = ((self.day - 1) + first_day[i]) %  7
                # Makes sure Sunday is properly represented by a 7
                if self.day_num == 0:
                    self.day_num = 7
                # Returns the day of the week
                return self.day_num
        # Returns False if any errors
        return False

    def createtimestampNum(self):
        """Generate a number that represents the entire timestamp"""
        self.time_num = int(self.getnumMonth() + self.getnumDay() + self.getstrHour() + self.getstrMinute() + self.getstrSecond())

    def ordertimeStamp():
        """Return the timestamps in chronological order"""
        # Sorts the timestamps by the time number
        timeStamp.timestamps = sorted(timeStamp.timestamps, key=operator.attrgetter('time_num'))
        return timeStamp.timestamps

    def generatetimeStamp(n):
        """Generates n number of timestamps"""
        return [timeStamp() for _ in range(n)]
