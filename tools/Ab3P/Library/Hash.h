#ifndef HASH_H
#define HASH_H

#include <iostream>
#include <fstream>
#include <Btree.h>
#include <FBase.h>

namespace iret {

class Hash : public FBase {
public:
  Hash(void);
  Hash(const char *nm);
  Hash(int n,const char *nm); //n gets appended to type if >-1
  ~Hash();

  void create_htable(List &Lst,int excess); //"str" for file of strings, 
      //"ad" for address file, "nm" numbers, 
      //"ha" hash array. Excess is # powers of 2 above size.
  void create_htableM(List &Lst,int excess); //creates in memory ready for use
      //and no need to call gopen or gclose functions
  void create_htable(int mz,List &Lst,int excess); //"str" for file of strings, 
      //Creates a numbered version of above

  void gopen_htable_map(void); //Creates memory maps
  void gopen_htable_map(int mz); //Creates memory maps
  void gclose_htable_map(void); //Destroys memory maps
     //and deletes memory
  void gclose_htable_map(int mz); //Destroys memory maps
     //and deletes memory
  void gopen_htable_copy(Hash *pH); //Copies memory maps

  long find(const char *str); //Return number+1 if present, else 0.
      //Number is not lexical order but hash order and then lexical
      //within collesion groups.

  //Data
  char *strmap; //Holds the bit map.
  long *addr; //Holds the offsets to strmap.
  long nwrds; //Number of words.
  long tnum; //Truncation number, size of har.
  long *harr; //Holds hash array.
  long *farr; //Holds the hash coefficients.
  long *px0;
  long *px1;
  long *px2;
  long *px3;
  long *px4;
  long *px5;
  long *px6;
  long *px7;
  long *px8;
  long *px9;
  long *px10;
  long *px11;
};

class Chash : public Hash {
public:
  Chash(void);
  Chash(const char *nm);
  Chash(int n,const char *nm); //n gets appended to type if >-1
  ~Chash(void);

  void create_ctable(Count &Ct,int excess); //Adds "ct" for counts
     //Calls create_htable and then prodoces the array of counts.
  void create_ctable(int mz,Count &Ct,int excess); //Adds "ct" for counts
     //Creates a numbered version of above
  void create_ctable(List &Lt,int excess); //Adds "ct" for term # 
     //and starts the count at 1 and in lexical order. count() will
     //return 0 if term not in list.
  void create_ctable(int mz,List &Lt,int excess); //Adds "ct" for term # 
     //Creates a numbered version of above
 
  void gopen_ctable_map(void); //Calls gopen_htable_map and also
     //maps "ct" file.
  void gopen_ctable_map(int mz); //Calls gopen_htable_map and also
     //maps "ct" file.
  void gclose_ctable_map(void); //Calls gclose_htable_map and also
     //Unmaps "ct" file.
  void gclose_ctable_map(int mz); //Calls gclose_htable_map and also
     //Unmaps "ct" file.

  long count(const char *str); //Returns count if present, else 0.

  //Data
  long *cnt;
};

}
#endif
