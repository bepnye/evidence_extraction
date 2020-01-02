import flask
import json
from collections import defaultdict

import annotator
import article
import config
import reader
import writer
import numpy as np
import pandas as pd

import model_utils

application = flask.Flask(__name__)

anne = annotator.Annotator(reader.get_reader(config.reader)(**config.reader_params),
                           writer.get_writer(config.writer)(**config.writer_params))

valid_users = np.loadtxt('usernames.txt', delimiter = ',', dtype = 'str')
all_anns = pd.read_csv('data/exhaustive_ico.csv');

"""
Display the main page.
"""
@application.route('/', methods=['GET'])
def index():
    return flask.render_template('index.html')

"""
Start the program.
"""
@application.route('/start/<userid>/', defaults={'ann_type':'full'}, methods=['GET', 'POST'])
@application.route('/start/<userid>/<ann_type>/', methods=['GET', 'POST'])
def start(userid, ann_type = 'abst'):
    if not(userid in valid_users):
        return flask.render_template('index_invalid_user.html')
        
    id_ = anne.get_next_file(userid)

    if not id_:
        return flask.redirect(flask.url_for('finish'))
    else:
        return flask.redirect(flask.url_for('annotate_article', 
                                            userid = userid, 
                                            id_ = id_,
                                            ann_type = ann_type))
                
"""
Start the program, but show the error to the user first.
"""
@application.route('/invalid_user/', methods=['GET', 'POST'])
def invalid_user():
    userid = flask.request.form['userid']
    if not(userid in valid_users):
        return flask.render_template('index_invalid_user.html', should_show = "true")
    
    id_ = anne.get_next_file(userid)
    if not id_:
        return flask.redirect(flask.url_for('finish'))
    else:
        return flask.redirect(flask.url_for('annotate_article', 
                                            userid = userid, 
                                            id_ = id_,
                                            ann_type = 'full'))

def get_abst_end(art):
  abst_end = art.abstract
  # recurse in to subsecs until we hit the last string of the last sec
  while type(abst_end) != str:
    abst_end = abst_end[-1]

  tag_idx = abst_end.index('xml_f')
  value_i = abst_end.index('=', tag_idx) + 2 # leading +"
  value_f = abst_end.index(' ', tag_idx) -1 # trailing "
  value = int(abst_end[value_i:value_f])
  return value

def get_ico_anns(art, id_, abst_only = False):
  ICO = ['Intervention', 'Comparator', 'Outcome']
  anns = [a for idx, a in all_anns.iterrows()]
  anns = [a for a in anns if str(a.RowID) == id_.replace('PMC', '')]

  if abst_only:
    abst_f = get_abst_end(art)
    abst_anns = []
    for a in anns:
      try:
        if int(a.xml_offsets.split(':')[1]) <= abst_f:
          abst_anns.append(a)
      except ValueError:
        pass
    anns = abst_anns

  icos = [[a[e] for e in ICO] for a in anns]
  unique_icos = set(map(tuple, icos))
  return [dict(zip(ICO, ico)) for ico in unique_icos]

"""
Grabs a specified article and displays the full text.
"""                             
@application.route('/annotate_article/<ann_type>/<userid>/<id_>/', methods=['GET'])
def annotate_article(userid, id_, ann_type):
  if id_ is None:
      art = anne.get_next_article(userid)
  else:
      art = anne.get_next_article(userid, id_)

  if ann_type == 'abst':
    abst_only = True
    tabs = art.text[0:1]
  else:
    abst_only = False
    tabs = art.text

  anns = get_ico_anns(art, id_, abst_only)

  if not art:
    return flask.redirect(flask.url_for('finish'))

  save_last_path(userid, art.get_extra()['path'])
  return flask.render_template('annotate_article.html',
                               ann_type = ann_type,
                               userid = userid,
                               annotations = anns,
                               id = art.id_,
                               pid = id_,
                               tabs = tabs,
                               xml_file = get_last_path(userid),
                               outcome = art.get_extra()['outcome'],
                               intervention = art.get_extra()['intervention'],
                               comparator = art.get_extra()['comparator'],
                               options = config.options_full)

"""
Grabs a specified article and displays the full text.
"""                             
@application.route('/browse/<userid>/<id_>/', methods=['GET'])
def browse(userid, id_ = None):
    if id_ is None:
        art = anne.get_next_article(userid)
    else:
        art = anne.get_next_article(userid, id_)
    
    if not art:
        return flask.redirect(flask.url_for('finish'))
    else:
        annos = model_annotations['docs'][id_]
        return flask.render_template('browse_article.html',
                                     userid = userid,
                                     id = art.id_,
                                     pid = id_,
                                     tabs = art.text,
                                     spans = annos,
                                     xml_file = get_last_path(userid),
                                     options = config.options_full)

@application.route('/instructions/')
def instructions():
  return flask.render_template('instructions.html')
                                 
"""
Submits the article id with all annotations.
"""
@application.route('/submit/', methods=['POST'])
def submit(): 
    userid = flask.request.form['userid']
    anne.submit_annotation(flask.request.form)

    id_ = anne.get_next_file(userid)
    ann_type = flask.request.form['ann_type']
    if not id_:
        return flask.redirect(flask.url_for('finish'))
    else:
      return flask.redirect(flask.url_for('annotate_article',
                                          userid = userid,
                                          id_ = id_,
                                          ann_type = ann_type))

"""
Only go to this if there are no more articles to be annotated.
"""
@application.route('/finish/', methods=['GET'])
def finish():
    return flask.render_template('finish.html')

"""
Call the get results funciton.
"""
@application.route('/results/', methods=['GET'])
def results():
    return anne.get_results()
    
"""
Get the last path.
"""
def get_user_progress_fname(user):
  return 'data/{}_cur_fname.txt'.format(user)

def get_last_path(user):
  return open(get_user_progress_fname(user)).read()
    
def save_last_path(user, path):
  with open(get_user_progress_fname(user), 'w') as fp:
   fp.write(path)

"""
Run the application.
"""
if __name__ == '__main__':
   #application.run()
   application.run(host = '0.0.0.0', port = 8001, debug = True) 
