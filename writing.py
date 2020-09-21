import os
import datetime
class Writelogs():
    if not os.path.exists("logs"):
        os.mkdir("logs")
    logfile = open('logs/log.txt', 'a')
    logfile.writelines('\n')
    logfile.write("<-----" + str(datetime.datetime.now()) + "----->" + ' \n')

    def writing(self,*args,**kwargs):
        for arg in args:
            Writelogs.logfile.write(str(arg)+'\n')
        #return
