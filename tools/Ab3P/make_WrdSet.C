/*
make a hash set for word (freq>=100, len>=3) 
need a source word list that is formated by word|freq at each line
need a "path" file that designates a directoy to save the hash set
*/

#include <fstream>
#include <Hash.h>

using namespace std;
using namespace iret;

main(int argc, char **argv)
{
  char str[10000], str2[10000];
  long num;
  Count Ct;

  if(argc!=2) {
    cout << "Usuage: make_WrdSet WordFilename" << endl;
    exit(1);
  }

  ifstream fin(argv[1]); 
  if(!fin) {
    cout << "Cannot open " << argv[1] << endl;
    exit(1);
  }

  long cnt;
  while(fin.getline(str,10000,'|')) {
    fin >> num; 
    fin.getline(str2,10000); //remove endl;
    if(strlen(str)<3) continue; // length>=3
    if(num>=100) { Ct.add_count2(str,num); cnt++; } //freq>=100    
  }

  cout << cnt << " words selected" << endl;

  Chash Csh("wrdset3"); 
  Csh.create_ctable(Ct,3);
}
