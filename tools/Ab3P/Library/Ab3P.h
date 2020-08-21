/*
Identify sf & lf pairs from free text using multi-stage algorithm
process one line at a time and print out:
line
  sf|lf|P-precision|strategy
*/

#include "AbbrvE.h"
#include "AbbrStra.h"
#include <vector>
#include <map>
#include <string>

using namespace std;
using namespace iret;

namespace iret {

class AbbrOut {
public:
  string sf, lf, strat;
  double prec;

  AbbrOut( void ) : sf(""), lf(""), strat(""), prec(0)
  {}

  void print ( ostream & out ) {
    out << "  "  << sf << "|" << lf << "|" << prec;
  }

};


class Ab3P {
public:
  Ab3P( void );
  ~Ab3P(void) { delete wrdData; }

  /**  Collect text for later abbreviation finding. **/
  void add_text( const string & text ) {
    buffer += text;
  }
  void add_text( char * text ) {
    buffer += text;
  }

  /**  Sets abbrs to the abbreviations found in previous calls to add_text.
       Afterwords, resets the text buffer.  **/
  void get_abbrs( vector<AbbrOut> & abbrs ) {
    get_abbrs( buffer, abbrs );
    buffer = "";
  }

  /**  Sets abbrs to the abbreviations found in text
       Does not interfere with the add_text buffer.  **/
  void get_abbrs( const string & text, vector<AbbrOut> & abbrs ) {
    abbrs.clear();

    if(text.empty()) return; // skip empty line
    // const_cast need so correct get_abbrs get called,
    // otherwise, infinite loop
    get_abbrs( const_cast<char*>(text.c_str()), abbrs );
  }
  void get_abbrs( char * text, vector<AbbrOut> & abbrs );

  /**  Try a potential sf-lf form to find proper lf, strategy used,
       and pseudo-precision of result **/
  void try_pair( char * sf, char * lf, AbbrOut & abbr );

  /**
     psf -- pointer short form
     plf -- pointer long form
  **/
  void try_strats ( char * psf, char * plf, bool swap, AbbrOut & result );

  AbbrvE ab; //default # pairs = 10,000 
  map<string, double> stratPrec;
  StratUtil util;
  WordData *wrdData;            //set data needed for AbbrStra
  string buffer;                // collect text for later use
};

}
