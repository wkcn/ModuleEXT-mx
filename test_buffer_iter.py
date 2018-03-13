import mxnet as mx
import time
from bufferIter import BufferIter

class SlowIter(mx.io.DataIter):
    def __init__(self, lst, sleep_time):
        self.lst = lst
        self.sleep_time = sleep_time
        self.reset()
    def reset(self):
        self.i = 0
    def next(self):
        time.sleep(self.sleep_time)
        if self.i < len(self.lst):
            e = self.lst[self.i]
            self.i += 1
            return e
        else:
            raise StopIteration


N = 3
def get_check_iter():
    while 1:
        for i in range(N):
            yield i

slow_iter = SlowIter(list(range(N)), 1.0)
slow_iter = BufferIter(slow_iter)
check_iter = get_check_iter()

st = time.time()
for k in range(20):
    print ("batch %d" % k)
    try:
        e = slow_iter.next()
    except StopIteration:
        slow_iter.reset()
        e = slow_iter.next()
        print ("reset")
    assert next(check_iter) == e
    time.sleep(2)


print ("Use Time", time.time() - st)
