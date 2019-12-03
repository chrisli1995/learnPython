from threading import Thread
from queue import Queue

class ThreadedGenerator(object):
    def __init__(self,iterator,
                 sentinenl=object(),
                 quene_maxsize=0,
                 daemon=False):
        self._iterator=iterator
        self._sentinenl=sentinenl
        self._queue=Queue(maxsize=quene_maxsize)
        self._thread=Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.daemon=daemon
        self._started=False

    # 定义迭代器输出的格式
    def __repr__(self):
        return 'ThreadedGenerator{!r}'.format(self._iterator)

    # 定义想要完成的操作
    def _run(self):
        try:
            for value in self._iterator:
                if not self._started:
                    return
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinenl)


    def close(self):
        self._started=False
        try:
            while True:
                self._queue.get(timeout=30)
        except KeyboardInterrupt as e:
            raise e
        except:
            pass

    def __iter__(self):
        self._started=True
        self._thread.start()
        for value in iter(self._queue.get(),self._sentinenl):
            yield value
        self._thread.join()
        self._started=False

    def __next__(self):
        if not self._started:
            self._started=True
            self._thread.start()
        value=self._queue.get(timeout=30)
        if value==self._sentinenl:
            raise StopIteration()
        return value

def test():
    def gene():
        i=0
        while True:
            yield i
            i+=1

    t=gene()
    print(type(t))
    test=ThreadedGenerator(t)
    print('tread',type(t))

    for _ in range(10):
        print(next(test))
    test.close()

if __name__ == '__main__':
    test()

