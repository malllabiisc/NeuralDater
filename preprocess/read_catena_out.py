import os, sys, pdb, numpy as np, random, argparse, codecs
from pprint import pprint
from collections import defaultdict as ddict
from joblib import Parallel, delayed


def read_catena(fname):
	edge_list = []

	with codecs.open(fname, encoding='utf-8', errors='ignore') as f:
		for line in f:
			if len(line.strip().split('\t')) != 4: continue
			_, src, dest, rtype = line.strip().split('\t')
			src, dest = src.replace('tmx', 't'), dest.replace('tmx', 't')
			edge_list.append({
				'src': 	src,
				'dest': dest,
				'type': rtype
			})

	return edge_list

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Loads Queue with files to process')
	parser.add_argument('-src', 	dest='src', 	default='test_in.tml', 	help='Path of output of CATENA')
	parser.add_argument('-dest', 	dest='dest', 	default='test_out.xml',	help='Destination to dump the extracted temporal graph')
	args = parser.parse_args()

	res = read_catena(args.src)
	open(args.dest, 'w').write(json.dumps(res))