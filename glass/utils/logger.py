import os
import atexit
import functools
import logging.config
import sys
from termcolor import colored
from iopath.common.file_io import PathManager


# We abbreviate this names according to the following table by searching the key and replacing with value
DefaultAbbrevs = {
    '__main__': 'main',
    'detectron2': 'd2',
}


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(output=None, distributed_rank=0, *, level=logging.INFO, color=True, name='__main__', log_color='green'):
    """ Configures the __main__ logger. This method should be called from the main script that runs
    :param output: The output dir for saving the log file, if not provided the log file will not be saved
    :param distributed_rank: if >= 0 than all the data is saved to file and not printed
    :param level: The logging level
    :param color: Whether to use a colored output or not
    :param name: The modules' name, should be usually __main__, can be discovered with the __name__ suffix
    :param log_color: The color of the log text displayed on screed: ['blue', 'yellow', 'green', 'cyan', 'magenta']
    :return: The logger for __main__
    """
    # Getting the root logger
    logger = logging.getLogger()
    # Clearing previous handlers to avoid duplicate messages
    logger.handlers = []
    # Setting log level DEBUG/INFO/WARNING/ERROR/CRITICAL
    logger.setLevel(level)
    # logger.propagate = True
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", 'green') + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                abbrev_dict=DefaultAbbrevs,
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManager().mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        # We always write with DEBUG level to file
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    # We return the logger that the user requested, giving flexibility in case we call from outside of '__main__'
    return logging.getLogger(name)


class _ColorfulFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        # self._root_name = kwargs.pop("root_name") + "."
        # self._abbrev_name = kwargs.pop("abbrev_name", "")
        # if len(self._abbrev_name):
        #     self._abbrev_name = self._abbrev_name + "."
        self._abbrev_dict = kwargs.pop("abbrev_dict", {})
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        for root_name, abbrev in self._abbrev_dict.items():
            record.name = record.name.replace(root_name, abbrev)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored('WARNING', 'yellow', attrs=['bold'])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored('ERROR', 'red', attrs=['bold'])
        elif record.levelno == logging.DEBUG:
            prefix = colored('DEBUG', 'magenta')
        else:
            return log
        return prefix + " " + log

@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = PathManager().open(filename, "a")
    atexit.register(io.close)
    return io
