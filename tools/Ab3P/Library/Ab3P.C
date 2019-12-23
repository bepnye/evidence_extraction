#include "Ab3P.h"

Ab3P::Ab3P ( void ) :
  buffer(""),
  wrdData( new WordData ) 
{
    
  string sf_grp, sf_nchr, strat;
  double value; 

  char file_name[1000];
  get_pathw( file_name, "Ab3P", "prec", "dat" );
  ifstream fin(file_name);
  if(!fin) {
    cout << "Cannot open Ab3P_prec.dat\n";
    exit(1);
  }
  //get precision of a given #-ch SF's strategy
  while(fin>>sf_grp>>sf_nchr>>strat) {
    fin>>value; //precision 
    stratPrec.insert(pair<string, double>(sf_grp+sf_nchr+strat, value));
    util.push_back_strat(sf_grp+sf_nchr, strat); //set strategy sequence
  }
}

void Ab3P::get_abbrs( char * text, vector<AbbrOut> & abbrs ) {
    abbrs.clear();

    if( ! text[0] ) return; // skip empty line

    ab.Proc(text); //extract potential SF & LF pairs
    
    for(int i=0; i<ab.numa; i++) {
      AbbrOut result;
      
      try_pair( ab.abbs[i], ab.abbl[i], result );
      
      // preserve results
      if ( result.prec > 0 ) {
        abbrs.push_back( result );
      }
    }
    ab.cleara();

  }


void Ab3P::try_pair( char * sf, char * lf, AbbrOut & result ) {
  
  //process i) lf (sf)
  try_strats( sf, lf, false, result );
  
  //process ii) sf (lf)
  ab.token(lf);
  try_strats( ab.lst[ab.num-1], sf, true, result );
}


  /**
     psf -- pointer short form
     plf -- pointer long form
  **/
void Ab3P::try_strats ( char * psf, char * plf, bool swap,
                        AbbrOut & result ) {
      
  string sfg; //SF group eg) Al1, Num2, Spec3
  //false if sf is not ok, sfg will be assigned

  if(!util.group_sf(psf,plf,sfg)) return;
  if (swap) if(!util.exist_upperal(psf)) return;

  char sf[1000], sfl[1000];

  //strategy sequence for a given #-ch SF group
  vector<string> strats = util.get_strats(sfg);
  util.remove_nonAlnum(psf,sf); //sf will be w/o non-alnum

  //go through strategies
  for( int j=0; j<strats.size(); j++) { 
    AbbrStra * strat =
      util.strat_factory(strats[j]); //set a paticular strategy
    strat->wData = wrdData; //set wordset, stopword
    if(strat->strategy(sf,plf)) { //case sensitive
      strat->str_tolower(sf,sfl);

      if( strat->lf_ok(psf,strat->lf) ) {

        map<string, double>::iterator p =
          stratPrec.find(sfg+strats[j]);
        if(p==stratPrec.end()) {
          cout << "No precision assigned" << endl;
          exit(1);
        }

        //add outputs 
        if( p->second>result.prec ) {
          result.sf = psf;
          result.lf = strat->lf;
          result.prec = p->second;
          result.strat = strats[j];
        }

        delete strat;
        return;
      }
    }
    delete strat;
  }

}
