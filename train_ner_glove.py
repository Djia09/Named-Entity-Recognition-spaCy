from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import time
from conllPrepro import main
from prepro import readfile
TRAIN_DATA = main('./CoNLL-2003/eng.train')
# TRAIN_DATA = readfile('./CoNLL-2003/eng.train')
print(TRAIN_DATA)