import argparse, os, sys, pdb, shlex, re, argparse
from parse import parse
from bs4 import BeautifulSoup
from pprint import pprint
from joblib import Parallel, delayed
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

"""
Reads the output of CAEVO (xml files) and extracts teh depdendency tree edges
and the tokenized document. 
"""

def read_xml(src):
	text = open(src).read()
	soup = BeautifulSoup(text, 'html.parser')

	try:
		doc = {
			'_id':       soup.find('file')['name'],
			'sentences': [],
			'links':     []
		}

		for entry in soup.find_all('entry'):
			sent = {
				'tokens':       [],
				'deps':         [],
				'events':       [],
				'times':        []
			}


			for ele in entry.find('tokens').find_all('t'):
				e = {}
				if ele.get_text() == '" " """ ""':
					e['left'], e['text'], e['right'] = ' ', '"', ''

				elif ele.get_text() == '"" """ " "':
					e['left'], e['text'], e['right'] = '', '"', ' '

				elif ele.get_text() == '"" """ ""':
					e['left'], e['text'], e['right'] = '', '"', ''

				elif ele.get_text() == '" " """ " "':
					e['left'], e['text'], e['right'] = ' ', '"', ' '

				elif ele.get_text() == '"  " """ ""':
					e['left'], e['text'], e['right'] = '  ', '"', ''

				elif ele.get_text() == '"" """ "  "':
					e['left'], e['text'], e['right'] = '', '"', '  '

				elif ele.get_text() == '"" """ "  "':
					e['left'], e['text'], e['right'] = '', '"', '  '
				elif ele.get_text() == '"" "\\" ""':
					e['left'], e['text'], e['right'] = '', '\\', ''
				elif ele.get_text().count('"') != 6:
					e['left'], e['text'], e['right'] = '', '"', ''
				else:
					e['left'], e['text'], e['right'] = shlex.split(ele.get_text())
				sent['tokens'].append(e)

			def get_dep(text):
				rel, left = text.split('(')
				src, left = left.split(', ')
				dest      = left.split(')')[0]

				src, src_id   = src[:src.rfind('-')],   src[src.rfind('-')+1:]
				dest, dest_id = dest[:dest.rfind('-')], dest[dest.rfind('-')+1:]

				return rel, src, int(src_id), dest, int(dest_id)

			for ele in entry.find('deps').get_text().split('\n'):
				if ele == '': continue
				e = {}
				if len(list(parse('{}({}-{}, {}-{})', ele))) == 5:
					e['rel'], e['src'], e['src_id'], e['dest'], e['dest_id'] = parse('{}({}-{}, {}-{})', ele)
				elif len(re.findall('\w+', ele)) == 5:
					e['rel'], e['src'], e['src_id'], e['dest'], e['dest_id'] = re.findall('\w+', ele)
				else:
					e['rel'], e['src'], e['src_id'], e['dest'], e['dest_id'] = get_dep(ele)

				sent['deps'].append(e)
			
			for ele in entry.find('events').find_all('event'):
				e = {}
				e['tok_id']     = ele['offset']
				e['eid']        = ele['id']
				e['text']       = ele['string']
				e['tense']      = ele['tense']
				e['class']      = ele['class']
				e['polarity']   = ele['polarity']
				sent['events'].append(e)

			for ele in entry.find('timexes').find_all('timex'):
				e = {}
				e['tok_id']     = ele['offset']
				e['tid']        = ele['tid']
				e['text']       = ele['text']
				e['length']     = ele['length']
				e['type']       = ele['type']
				e['value']      = ele['value']
				sent['times'].append(e)

			doc['sentences'].append(sent)

		for link in soup.find_all('tlink'):
			e = {}
			e['relType']            = link['relation']
			e['src'], e['dest']     = link['event1'].replace('i',''), link['event2'].replace('i','')
			e['type']               = link['type']

			doc['links'].append(e)
	
		return doc

	except Exception as e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		fname = os.src.split(exc_tb.tb_frame.f_code.co_filename)[1]
		print('\nException Type: {}, \nCause: {}, \nfname: {}, \nline_no: {}'.format(exc_type, e.args[0], fname, exc_tb.tb_lineno))
		print('{}|{}'.format(ele))
		return {}

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Adjust input for CATENA')
	parser.add_argument('-src', 	dest='src', 	default='test_in.xml', 	help='Path of output of CAEVO')
	parser.add_argument('-dest', 	dest='dest', 	default='test_out.xml',	help='Destination to dump the tokenized document and dependency graph')
	args = parser.parse_args()

	res = read_xml(args.src)
	open(args.dest, 'w').write(json.dumps(res))