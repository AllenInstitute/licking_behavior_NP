import licking_behavior_NP.psy_tools as ps 
import argparse

parser = argparse.ArgumentParser(description='fit behavioral model for session_id')
parser.add_argument(
    '--bsid', 
    type=int, 
    default=0,
    metavar='bsid',
    help='behavior session id'
)
parser.add_argument(
    '--version', 
    type=str, 
    default='',
    metavar='behavior model version',
    help='model version to use'
)

if __name__ == '__main__':
    args = parser.parse_args()
    ps.process_session(args.bsid, version = int(args.version))
