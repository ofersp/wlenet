import sys

class Logger(object):

    def __init__(self, logfile_path, terminal=sys.stdout):
        self.terminal = terminal
        self.log = open(logfile_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()