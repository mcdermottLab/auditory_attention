

import os
import argparse
import re


def get_args():
  parser = argparse.ArgumentParser(description="""
      This script is used to convert opus file into wav file.""")
  parser.add_argument('--remove-opus', action='store_true', default='False',
      help="""If true, remove opus files""")
  parser.add_argument('opus_scp', help="""Input opus scp file""")
  parser.add_argument('--start_ix', default=0, type=int,
                    help='Line to start working on', required=False) 
  parser.add_argument('--n_files', default=-1, type=int,
                    help='Files to run per job.', required=False)
 
  args = parser.parse_args()
  return args


def convert_opus2wav(opus_scp, rm_opus, start_ix=0, n_files=-1):
  with open(opus_scp, 'r') as oscp:
    lines = oscp.read().splitlines()

    start = (start_ix-1) * n_files  
    end = start + n_files
    if end > len(lines) or n_files == -1:
      end = -1 
    for line in lines[start:end]:
      utt, opus_path  = line.split('\t')  
      wav_path = opus_path.replace('.opus', '.wav')
      cmd = f'ffmpeg -y -i {opus_path} -ac 1 -ar 16000 {wav_path}'
      try:
        os.system(cmd)
      except:
        sys.exit(f'Failed to run the cmd: {cmd}')
      if rm_opus is True:
        os.remove(opus_path)


def main():
  args = get_args()

  convert_opus2wav(args.opus_scp, args.remove_opus, args.start_ix, args.n_files)


if __name__ == '__main__':
  main()
