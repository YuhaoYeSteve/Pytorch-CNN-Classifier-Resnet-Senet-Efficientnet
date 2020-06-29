import logging
from logging.handlers import TimedRotatingFileHandler
import datetime
import os


def createPathDirectory(path):
    try:
        path = path.replace("\\", "/")
        p0 = path.rfind('/')
        if p0 != -1 and not os.path.exists(path[:p0]):
            os.makedirs(path[:p0])

    except Exception as e:
        print(e)


def getLogger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    createPathDirectory(path)

    rf_handler = logging.handlers.TimedRotatingFileHandler(path, when='midnight', interval=1, backupCount=7,
                                                           atTime=datetime.time(0, 0, 0, 0))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s')
    rf_handler.setFormatter(formatter)
    rf_handler.setLevel(logging.INFO)
    logger.addHandler(rf_handler)

    sh_handler = logging.StreamHandler()
    sh_handler.setLevel(logging.DEBUG)
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    return logger



if __name__ == "__main__":
    logger = getLogger(__name__, "logs/log_test.txt")
    logger.info("this is info")
    logger.debug("this is debug")

