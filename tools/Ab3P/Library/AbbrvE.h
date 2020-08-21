#ifndef ABBRVE_H
#define ABBRVE_H
#include <fstream>
#include <iostream>
#include <runn.h>
#include <MPtok.h>
#include <vector>
using namespace std;
namespace iret {

typedef vector<string> strings;


class Find_Seq {
public:

  Find_Seq( void  );

  // flag the SFs whether part of sequence or not
  void flag_seq( int numa, char* abbs[] );

  // true if good SF, false if part of sequence
  bool rate( int i ) const { my_rate[i]; }

private:
  void find_seq( const vector<string> & seq );
  void create_seq( void );

  // const works with c++0x
  /* const */ strings seq_i;
  /* const */ strings seq_I;
  /* const */ strings seq_a;
  /* const */ strings seq_A;
  
  vector<bool> my_rate;
  int my_numa;
  char ** my_abbs;              // really char *[], but that doesn't work
  
};


class AbbrvE {
   public:
      AbbrvE(long ta=10000,long wrd_spc=10000); //Sets space for extracted
         //potential abbreviations to ta & word_space to wrd_spc
     ~AbbrvE(void);
      void Extract(char *pch); //Extracts possible long-short form
         //pairs, but does not attempt to find the relationship
      void Extract2(const char *pch); //extened version (Jan-9-2008)
      bool Test(const char *str); //Tests a single token and returns true
         //if the token should be a possible first token of a short form
      void Rate(void); //Sets ratings for the proposed pairs. Effort to 
         //remove (a), (b), etc., sequence markers
      void token(const char *str); //Produces a list of tokens in order of
         //of occurrence in the string.
      void token2(const char *str); //extended version (Jan-9-2008)
      void cleara(void); //Clear the abbl & abbs memory of strings
      void clear(void); //Clear the lst memory of words

      //Application functions
      void Proc(char *pch); //Accepts a natural language statement and
         //processes to final results stored in tta, abbs, and abbl
         //Need to call cleara function after each use of this function

      // Internal routines:
      // setup data for Test method
      void setup_Test( void );
      bool prefix_match( const char *str ); // does str begins with a prefix?

      //Data
      long tta; //Total possible abbreviations extracted
         //default 10k
      long numa; //number of abbreviations in current extract
      char **abbl; //Long form space, hold up to 10 tokens
      char **abbs; //Short form space, hold up to 10 tokens
      Find_Seq seq;             // identify sequences to ignore
      int  *nt; //Number of tokens within parentheses
      long word_space; //Space in lst for tokens
         //default 10k
      long num; //Number of tokens
      char **lst; //Holds the tokens

      static const int cnam_size=100000;
      char cnam[cnam_size]; //Work space
      MPtok *pMt; //Pointer at tokenizer class. Used to segment text
         //in Proc function.

      // Test data
      set<string> match;        // bad SF to match exactly
      vector<string> prefix;    // bad SF to match prefix
};
}
#endif
