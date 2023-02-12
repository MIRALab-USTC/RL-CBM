from warnings import simplefilter, filterwarnings 
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action="ignore",category=DeprecationWarning)
filterwarnings("ignore",category = DeprecationWarning)
from cbm.utils.launch_utils import parse_cmd, run_experiments
run_experiments(*parse_cmd())  
