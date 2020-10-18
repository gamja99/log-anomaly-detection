# Imports
import csv
import random

# Class for Roles
class Role:
    # Initialize list of class objects
    roles = []

    def __init__(self, name):
        """Initialize object"""
        # Assign name to object
        self.name = name
        # Assign list of users
        self.users = []
        # Assign list of privileges in the form of {operation: [objects]}
        self.privileges = {}
        # Assign list of privileges in the form of {operation: [objects]} if the user is logged on
        self.logonprivileges = {}
        # Assign list of privileges in the form of {operation: [objects]} if the user is logged off
        self.logoffprivileges = {}
        # Adds new object to list of objects
        Role.roles.append(self)

    def checkroleExist(name):
        """Checks to see if a specific role with given name already exists"""
        return any([name == role.name for role in Role.roles])

    def addUsers(self, user_objs):
        """Adds a list of users in the form of user objects"""
        for user in user_objs:
            self.users.append(user)
    
    def getallUsers(self):
        """Returns list of all users under a specific role"""
        return self.users
    
    def checkuserExist(self, username):
        """Checks to see if the user has been added for a specific role"""
        return any([username == user.name for user in self.users])
    
    def addPrivileges(self, operations, objects):
        """Adds privileges in the form of {operation: [objects]} to the specific role"""
        # Checks to see if each operation has corresponding objs or empty obj list
        if len(operations) != len(objects):
            return "Make sure that the operations and objects have the same length. If an operation does not have an object, place an empty string"
        # Adds operation, object relationship to list of privileges
        for i, operation in enumerate(operations):
            self.privileges[operation] = objects[i]

    def getallPrivileges(self):
        """Returns back list of privileges for a specific role"""
        return self.privileges
    
    def checkprivilegeExist(self, operation):
        """Checks to see if a specific privilege with given operation name already exists"""
        return operation in self.privileges.keys()

    def getobjectsofOperation(self, operation):
        """Returns the objects under the specific operation for a specific role"""
        return self.privileges[operation]

    def getrandRole():
        return random.choice(Role.roles)

    def getrandUser(self):
        return random.choice(self.users)

    def getrandPrivilege(self, login):
        if login:
            privilege_obj = random.choice(list(self.logonprivileges.items()))
        else:
            privilege_obj = random.choice(list(self.logoffprivileges.items()))
        if privilege_obj[1] != []:
            one_obj = random.choice(privilege_obj[1])
        else:
            one_obj = "none"
        return privilege_obj[0], one_obj


# Class for Users
class User:
    # Intialize list of class objects
    users = []
    def __init__(self, name):
        """Initializes object"""
        # Assigns name to object
        self.name = name
        # Initializes variable that checks whether the user has logged in or logged out in log data
        self.login = False
        # Adds new object to list of objects
        User.users.append(self)

    def checkuserExist(name):
        """Returns whether a certain user with the given name already exists"""
        return any([name == user.name for user in User.user])

    def createUsers(csv_file):
        """Creates users from a csv_file"""
        with open(csv_file, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Makes the username into a single string from array
                username = "".join(row)
                # Creates object
                User(username)

    def getUser(name):
        """Returns the user object from the given name or returns False"""
        for user in User.users:
            if name == user.name:
                return user
        return False
    
    def checkLogin(self):
        """Returns True if the user is logged in and False if the user is logged out"""
        return self.login
    
    def updateLogin(self, operation):
        """Updates whether the user is logged in or logged out"""
        if operation == "LOGON" or operation == "LOGOFF":
            self.login = not self.login



def getroleUsers(csv_file):
    """Creates Role and User Objects and adds the list of users to the respective role"""
    with open(csv_file, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Gets the list of users and splits them by spaces
            list_users = row[1].split()
            # Initializes list of user objects
            user_objs = []
            # Loops through all users
            for user in list_users:
                # Creates user object for each user and adds it to the list of user objects
                user_objs.append(User(user))
            # Creates the role object and adds the list of user objects as its list of users
            Role(row[0]).addUsers(user_objs)

def getrolePrivileges(folder):
    for role in Role.roles:
        file_name = folder + "/" + role.name + ".csv"
        with open(file_name, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                list_objs = row[1].split()
                role.privileges[row[0]] = list_objs
                for key, values in role.privileges.items():
                    if key != "LOGON":
                        role.logonprivileges[key] = values
                    if key == "LOGON":
                        role.logoffprivileges[key] = values

"""
getroleUsers('row_data/user_roles.csv')
getrolePrivileges("operations_object")
"""