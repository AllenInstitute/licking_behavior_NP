import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob 

VERSION = '10'

python_file = r"/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/psytrack_fit_v"+VERSION+".py"
jobdir = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v'+VERSION+'/psytrack_logs'
job_settings = {'queue': 'braintv',
                'mem': '10g',
                'walltime': '96:00:00',
                'ppn':1,
                'jobdir': jobdir,
                }

# Load list of sessions
id_file = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/psy_ids_v'+VERSION+'.txt'
with open(id_file) as f:
    string_ids = f.read()

# convert string to list of ints
string_ids = string_ids.split('\n')
behavior_session_ids=[]
for s in string_ids[0:-1]:
    behavior_session_ids.append(int(float(s)))

for behavior_session_id in behavior_session_ids:
    PythonJob(
        python_file,
        python_executable='/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python', 
        python_args=behavior_session_id,
        conda_env=None,
        jobname = 'psy_{}'.format(behavior_session_id),
        **job_settings
    ).run(dryrun=False)



