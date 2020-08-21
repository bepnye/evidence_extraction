#ifndef FBASE_H
#define FBASE_H

#include <iostream>
#include <fstream>

using namespace std;
namespace iret {

typedef char *pChr;

class FBase {
   public:
      FBase(const char *tp,const char *nm); //tp is type name, nm is name 
      FBase(const char *tp,int tn,const char *nm); //tp is type name, if 
         //nonnegative tn is appended to end of tp, nm is name 
      FBase(const char *tp,const char *nm,const char *pt); //tp is type name, nm is name 
         //pt is pointer at a string sss and reads the path from file path_sss in 
         //current directory. But if sss begins with ':' then skips this character 
         //and remaining string is the path string itself.
      ~FBase();
      void set_type_num(int tn); //Sets tpnm and uses if nonnegative: appended
         //to end of type name
      void set_name_num(int nn); //Sets nmnm and uses if nonnegative: appended
         //to end of name
      void change_type(const char *nm); //Allows change of type string for class.
      void change_name(const char *nm); //Allows change of name string for class.
      void set_name(const char *nm); //Allows change of name string for class.
         //Included for compatibility
      void subname(const char *tph,const char *tpl,const char *nm); //Uses the 
         //higher level type tph as type and combines lower level tpl_nm with
         //name to allow one to keep track of file types.
      void set_path_internal(const char *pt); //Path is by default external with
         //eflag=1. But if this function called with nonempty string, then eflag=0
         //and pt stored in path and used for access to data.
      void set_path_name(const char *pa); //path will be extracted from path_pa
         //and eflag=2. Naming conventions for files are unchanged
      void map_down(FBase *pFb); //Maps naming parameters to class instance pFb
      void map_down_sub(FBase *pFb,const char *subtype); //Maps naming parameters to class instance pFb
         //combines subtype with name to make a new name for pFb and type becomes its type

      //Path access functions
      void get_pathx(char *cn,const char *a);
         //Reads the path from a file "path_(*name)" and constructs the
         //file name from as "(*type)_(*name).(*a)". Cats path and file
         //name and returns the full info in cn.
      void get_pathx(char *cn,long n,const char *a);
      char *add_num(const char *ptr,long n,char *buf); //converts long to ascii
         //and cats to end of string and returns pointer to new string
         //that results. Does not change input string. The new string is
         //held in buffer space and this is overwritten at each call.
      char *cat_num(const char *ptr,long n,char *buf); //converts long to ascii
         //and cats to end of ptr string and then cats result to end of
         //whatever is in buffer. Does not change input string. The new string is
         //held in buffer space.

      //Stream object pointers
      ifstream *get_Istr(const char *a,ios::openmode m=ios::in);
         //Opens input file stream by path and name composition.
      ofstream *get_Ostr(const char *a,ios::openmode m=ios::out);
         //Opens output file stream by path and name composition.
      fstream *get_Fstr(const char *a,ios::openmode m=ios::in|ios::out);
         //Opens output file stream by path and name composition.
      ifstream *get_Istr(long n,const char *a,ios::openmode m=ios::in);
      ofstream *get_Ostr(long n,const char *a,ios::openmode m=ios::out);
      fstream *get_Fstr(long n,const char *a,ios::openmode m=ios::in|ios::out);
      void dst_Istr(ifstream *pfin);
      void dst_Ostr(ofstream *pfout);
      void dst_Fstr(fstream *pfstr);

      //Get file size in bytes
      long get_Fsiz(const char *a);
      long get_Fsiz(long n,const char *a);

      //File existence
      int Exists(const char *a); //returns 1 if file exists
      int Exists(long n,const char *a); //returns 1 if file exists

      //Read in array pointers
      char *get_Read(const char *a);
         //Reads in a file into an char array and returns pointer
      char *get_Read(long n,const char *a);

      //Memory map pointers
      char *get_Mmap(const char *a);
         //Memory maps file by path and name composition.
      char *get_Mmap(long n,const char *a);
      char *get_Wmap(const char *a);
         //Memory maps file by path and name composition.
         //Allows to modify contents and is written out when dst_Mmap called
      char *get_Wmap(long n,const char *a);
         //Allows to modify contents and is written out when dst_Mmap called
      void dst_Mmap(const char *a,char *ptr);
         //Removes the memory map for ptr based on path and name composition.
      void dst_Mmap(long n,const char *a,char *ptr);

      //Array of chars and binary write
      void bin_Writ(const char *a,long nm,char *ptr); 
         //Writes out nm bytes binary
      void bin_Writ(long n,const char *a,long nm,char *ptr); 

      //Write and read 1, 2, or 3 long integers to or from a file
      template <typename X>
      void get_Nnum(const char *a,X &m1);
      template <typename X,typename Y>
      void get_Nnum(const char *a,X &m1,Y &m2);
      template <typename X,typename Y,typename Z>
      void get_Nnum(const char *a,X &m1,Y &m2,Z &m3);
      template <typename X>
      void get_Nnum(long n,const char *a,X &m1);
      template <typename X,typename Y>
      void get_Nnum(long n,const char *a,X &m1,Y &m2);
      template <typename X,typename Y,typename Z>
      void get_Nnum(long n,const char *a,X &m1,Y &m2,Z &m3);
      template <typename X>
      void put_Nnum(const char *a,X &m1);
      template <typename X,typename Y>
      void put_Nnum(const char *a,X &m1,Y &m2);
      template <typename X,typename Y,typename Z>
      void put_Nnum(const char *a,X &m1,Y &m2,Z &m3);
      template <typename X>
      void put_Nnum(long n,const char *a,X &m1);
      template <typename X,typename Y>
      void put_Nnum(long n,const char *a,X &m1,Y &m2);
      template <typename X,typename Y,typename Z>
      void put_Nnum(long n,const char *a,X &m1,Y &m2,Z &m3);

      //Logical accounting functions 
      int Gcom(int sflag); //sflag is bit marker such as READ_W, etc.
         //This returns 1 if sflag bit not set in oflag and is in cflag
         //If this is the case then it sets sflag in oflag.
      int Rcom(int sflag);
         //This returns 1 if sflag bit set in oflag and in cflag
         //If this is the case then it turns off sflag in oflag.
      void mark(long,int,const char*);
         //This function prints out string in 3rd argument and count
         //if first argument is multiple of the second

//Data
     int cflag; //Command, what should happen to resources.
     int oflag; //Bit string status of resources, 1 open, 0 closed.
     int open1; //flags to mark whether a resource is open or not
     int open2; //0 means closed, 1 means open
     int open3; //Used for those resources that are either completely
     int open4; //closed or completely open.
     int open5;
     char *type;
     int tpnm; //If nonnegative integer it is appended to end of type
        //in constructing file name
     char *name;
     int nmnm; //If nonnegative integer it is appended to end of name
        //in constructing file name
     int pflag; //Usual print flag, 1 for verbose output, 0 for none
        //Print flag set to 1 by default. 
     int eflag; //Flag set to 1 for external path from path file, 0
        //for internal path
     char *path; //Path stored here if eflag=0.
     char *pnam; //Path extension stored here if eflag=2.
};

//Template functions

template <typename X>
void FBase::get_Nnum(const char *a,X &m1){
   ifstream *pfin=get_Istr(a,ios::in);
   *pfin >> m1;
   dst_Istr(pfin);
}

template <typename X,typename Y>
void FBase::get_Nnum(const char *a,X &m1,Y &m2){
   ifstream *pfin=get_Istr(a,ios::in);
   *pfin >> m1 >> m2;
   dst_Istr(pfin);
}

template <typename X,typename Y,typename Z>
void FBase::get_Nnum(const char *a,X &m1,Y &m2,Z &m3){
   ifstream *pfin=get_Istr(a,ios::in);
   *pfin >> m1 >> m2 >> m3;
   dst_Istr(pfin);
}

template <typename X>
void FBase::get_Nnum(long n,const char *a,X &m1){
   ifstream *pfin=get_Istr(n,a,ios::in);
   *pfin >> m1;
   dst_Istr(pfin);
}  
   
template <typename X,typename Y>
void FBase::get_Nnum(long n,const char *a,X &m1,Y &m2){
   ifstream *pfin=get_Istr(n,a,ios::in);
   *pfin >> m1 >> m2;
   dst_Istr(pfin);
}  
   
template <typename X,typename Y,typename Z>
void FBase::get_Nnum(long n,const char *a,X &m1,Y &m2,Z &m3){
   ifstream *pfin=get_Istr(n,a,ios::in);
   *pfin >> m1 >> m2 >> m3;
   dst_Istr(pfin);
}

template <typename X>
void FBase::put_Nnum(const char *a,X &m1){
   ofstream *pfout=get_Ostr(a,ios::out);
   *pfout << m1 << endl;
   dst_Ostr(pfout);
}

template <typename X,typename Y>
void FBase::put_Nnum(const char *a,X &m1,Y &m2){
   ofstream *pfout=get_Ostr(a,ios::out);
   *pfout << m1 << " " << m2 << endl;
   dst_Ostr(pfout);
}

template <typename X,typename Y,typename Z>
void FBase::put_Nnum(const char *a,X &m1,Y &m2,Z &m3){
   ofstream *pfout=get_Ostr(a,ios::out);
   *pfout << m1 << " " << m2 << " " << m3 << endl;
   dst_Ostr(pfout);
}

template <typename X>
void FBase::put_Nnum(long n,const char *a,X &m1){
   ofstream *pfout=get_Ostr(n,a,ios::out);
   *pfout << m1 << endl;
   dst_Ostr(pfout);
}

template <typename X,typename Y>
void FBase::put_Nnum(long n,const char *a,X &m1,Y &m2){
   ofstream *pfout=get_Ostr(n,a,ios::out);
   *pfout << m1 << " " << m2 << endl;
   dst_Ostr(pfout);
}

template <typename X,typename Y,typename Z>
void FBase::put_Nnum(long n,const char *a,X &m1,Y &m2,Z &m3){
   ofstream *pfout=get_Ostr(n,a,ios::out);
   *pfout << m1 << " " << m2 << " " << m3 << endl;
   dst_Ostr(pfout);
}

}
#endif
