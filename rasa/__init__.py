import logging

from rasa import version  # noqa: F401
from rasa.api import run, train, train_dist, test  # noqa: F401

# define the version before the other imports since these need it
__version__ = version.__version__


logging.getLogger(__name__).addHandler(logging.NullHandler())
