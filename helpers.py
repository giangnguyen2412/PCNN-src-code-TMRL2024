import psutil
import os


class HelperFunctions(object):
    def __init__(self):
        pass

    @staticmethod
    def is_program_running(script):
        """
        Check if a script is already running
        :param script:
        :return:
        """
        for q in psutil.process_iter():
            if q.name().startswith('python'):
                if len(q.cmdline())>1 and script in q.cmdline()[1] and q.pid !=os.getpid():
                    print("'{}' Process is already running".format(script))
                    return True

        return False