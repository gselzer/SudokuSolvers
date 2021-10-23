from datetime import datetime

startTime = None
totalTime = None

def startTimer():
    global startTime
    startTime = datetime.now()

def stopTimer():
    global totalTime
    if (startTime == None):
        exit
    if (totalTime == None):
        totalTime = datetime.now() - startTime
        exit
    totalTime += datetime.now() - startTime

def getTotal():
    return totalTime

def printTotalTime():
    if (totalTime == None):
        print("Timer has not been run yet")
        exit
    out = ""
    if (totalTime.seconds > 0):
        out += str(totalTime.seconds) + " seconds, "
    out += str(totalTime.microseconds) + " microseconds"
    print(out)
    

def reset():
    global totalTime
    totalTime = None