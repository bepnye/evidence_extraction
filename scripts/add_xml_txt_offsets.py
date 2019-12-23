import lxml.etree as etree
from xml.sax.saxutils import escape
from glob import glob
import sys

# we get {http://blah.blah}s instead of ns:s
def format_xmlns(s, el):
  nsmap_inverse = { v: k for k, v in el.nsmap.items() }
  # universal non-specified namespace as per:
  #     https://www.w3.org/TR/xml-names/
  nsmap_inverse['http://www.w3.org/XML/1998/namespace'] = 'xml'
  prefix, suffix = s[1:].split('}', 1)
  ns = nsmap_inverse.get(prefix, '')
  if not ns:
    print('Unrecognized xmlns prefix:')
    print('\t', prefix)
    print('\t', nsmap_invers)
    input()
  return '{}:{}'.format(ns, suffix)

def parse_fname(fname, output_dir = 'parsed_nxmls/'):
  xml_out = ''
  html_out = ''
  txt_out = ''

  newline_tags = ['p', 'sec', 'title', 'td']
  collected_tags = ['abstract', 'body']
  collect_txt = False
  
  for e, el in etree.iterparse(open(fname, 'rb'), ('start', 'end')):

    if e == 'start':

      tag = el.tag
      if el.tag.startswith('{'):
        tag = format_xmlns(tag, el)

      tag_txt = '<{}'.format(tag)

      if tag == 'article':
        for k, v in el.nsmap.items():
          tag_txt += escape(' xmlns:{}="{}"'.format(k, v))

      for k, v in el.attrib.items():
        if k.startswith('{'):
          k = format_xmlns(k, el)
        tag_txt += escape(' {}="{}"'.format(k, v))

      tag_txt += '>'

      if tag in collected_tags:
        collect_txt = True
        txt_out += '<{}>\n'.format(tag.upper())
      xml_out += tag_txt
      html_out += tag_txt

      if el.text:
        xml_i = len(xml_out)
        txt_i = len(txt_out) if collect_txt else -1

        xml_out += escape(el.text)
        if collect_txt:
          txt_out += el.text

        xml_f = len(xml_out)
        txt_f = len(txt_out) if collect_txt else -1

        if collect_txt:
          html_span_txt = '<offsets xml_i="{}" xml_f="{}" txt_i="{}" txt_f="{}">'.format(xml_i, xml_f, txt_i, txt_f)
          html_out += html_span_txt
        html_out += escape(el.text)
        if collect_txt:
          html_out += '</offsets>'

    elif e == 'end':
      tag = el.tag
      if el.tag.startswith('{'):
        tag = format_xmlns(tag, el)

      tag_txt = '</{}>'.format(tag)
      xml_out += tag_txt
      html_out += tag_txt

      if collect_txt and tag in newline_tags:
        txt_out += '\n'

      # wrap tail in a special html tag since the start index is no longer linked to the opening tag
      if el.tail:

        txt_i = len(txt_out) if collect_txt else -1
        xml_i = len(xml_out)

        xml_out += escape(el.tail)
        if collect_txt:
          txt_out += el.tail
        
        txt_f = len(txt_out) if collect_txt else -1
        xml_f = len(xml_out)

        if collect_txt:
          html_tag_txt = '<offsets xml_i="{}" xml_f="{}" txt_i="{}" txt_f="{}">'.format(xml_i, xml_f, txt_i, txt_f)
          html_out += html_tag_txt
        html_out += escape(el.tail)
        if collect_txt:
          html_out += '</offsets>'

      if tag in collected_tags:
        collect_txt = False

  pmid = fname.split('/')[-1].split('.')[0]
  with open('{}/{}.xml'.format(output_dir, pmid), 'w') as fout: fout.write(xml_out)
  with open('{}/{}.txt'.format(output_dir, pmid), 'w') as fout: fout.write(txt_out)
  with open('{}/{}.html'.format(output_dir, pmid), 'w') as fout: fout.write(html_out)

def parse_all(xml_dir):
  fnames = glob('{}/*.nxml'.format(xml_dir))
  for fname in fnames:
    parse_fname(fname)

if __name__ == '__main__':
  parse_all(sys.argv[1])
