import os, sys, pdb, numpy as np, random, argparse
from pprint import pprint
from collections import defaultdict as ddict
from joblib import Parallel, delayed
from pymongo import MongoClient
from bs4 import BeautifulSoup, NavigableString

"""
The function takes in the path of .xml generated from CAEVO and makes it compatible
for input to CATENA for extracting the temporal graph.
"""

def make_catena_input(src, dest):
	text = open(src).read()
	soup = BeautifulSoup(text, 'xml')
	soup.find('DCT').insert_after(soup.new_tag('TITLE'))
	soup.find('DCT').append(soup.new_tag('TIMEX3', functionInDocument="CREATION_TIME", temporalFunction="false", tid="t0", type="DATE", value=""))

	for e in soup.find_all('event'):
		new_e = soup.new_tag('EVENT', **e.attrs)
		new_e.insert(0, NavigableString(e.get_text()))
		e.replaceWith(new_e)

	[s.extract() for s in soup('TLINK')]

	with open(args.dest + src.split('/')[-1] + '.tml', 'w') as f:
		f.write(str(soup))

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Adjust input for CATENA')
	parser.add_argument('-src', 	dest='src', 	default='test_in.xml')
	parser.add_argument('-dest', 	dest='dest', 	default='test_out.xml')
	args = parser.parse_args()

	make_catena_input(args.src, args.dest)