import logging


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class logger:
    def __init__(self):
        self.log = logging.getLogger('logger')
        self.log.setLevel(logging.DEBUG)

        # 设定输出流
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s 🎙️ > %(message)s')
        handler.setFormatter(formatter)

        self.log.addHandler(handler)
        # 设定debug 日志
        handler_debug = logging.FileHandler('debug.log')
        handler_debug.setLevel(logging.DEBUG)
        handler_debug.setFormatter(formatter)

        self.log.addHandler(handler_debug)
        # 设定error日志
        handler_error = logging.FileHandler('error.log')
        handler_error.setLevel(logging.ERROR)
        handler_error.setFormatter(formatter)

        self.log.addHandler(handler_error)

    def debug(self, message):
        self.log.debug(message)

    def error(self, message):
        self.log.error(message)
