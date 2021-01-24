import sys
import utils.utils as utils

d = utils.read_all_config(sys.argv[1:])
print(d)