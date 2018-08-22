# A progress bar based on: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

import sys
import time


def progress(count, total, status=''):
    
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))    
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)    
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    
    
if __name__ == '__main__':
    
    total = 100
    for i in range(total):
        progress(i, total, status='Sleeping')
        time.sleep(0.1)