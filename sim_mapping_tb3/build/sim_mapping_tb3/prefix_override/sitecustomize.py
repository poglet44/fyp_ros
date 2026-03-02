import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/thor-usr/Sam/fyp_ws/src/sim_mapping_tb3/install/sim_mapping_tb3'
