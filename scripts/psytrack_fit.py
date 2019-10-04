#!/usr/bin/env python

import sys
import psy_tools as ps

if __name__ == '__main__':
    name_of_this_file   = sys.argv[0]
    experiment_id       = sys.argv[1]
    
    ps.process_session(experiment_id,complete=True,format_options={'timing0/1':True})

