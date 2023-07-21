
import logging
import os

def create_logger(logdir, name):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    log_path = os.path.join(os.getcwd(), logdir, 'Log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    logfile = log_path  +'/{}.log'.format(name)
    fh = logging.FileHandler(logfile, mode='a')   # 'a' -> append, 'w'
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
                fmt="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                datefmt='%a, %d %b %Y %H:%M:%S'
                )

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console = logging.StreamHandler()
    # console.setLevel(logging.ERROR)
    # console.setFormatter(formatter)
    # logger.addHandler(console)

    return logger

def close_logger(logger):
    if logger is None:
        return
    else:
        for handler in logger.handlers[:]:
            handler.stream.close()
            logger.removeHandler(handler)
        return
