#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "runn.h"
#include "FBase.h"

using namespace std;
namespace iret {

FBase::FBase(const char *typ,const char *nam){
   int lxn=strlen(typ);
   type=new char[lxn+1];
   tpnm=-1;
   nmnm=-1;
   strcpy(type,typ);
   lxn=strlen(nam);
   name=new char[lxn+1];
   strcpy(name,nam);
   cflag=0;
   oflag=0;
   pflag=get_qflag();
   eflag=1;
}

FBase::FBase(const char *typ,int tpn,const char *nam){
   int lxn=strlen(typ);
   type=new char[lxn+1];
   tpnm=tpn;
   nmnm=-1;
   strcpy(type,typ);
   lxn=strlen(nam);
   name=new char[lxn+1];
   strcpy(name,nam);
   cflag=0;
   oflag=0;
   pflag=get_qflag();
   eflag=1;
}

FBase::FBase(const char *typ,const char *nam,const char *pt){
   int lxn=strlen(typ);
   type=new char[lxn+1];
   tpnm=-1;
   nmnm=-1;
   strcpy(type,typ);
   lxn=strlen(nam);
   name=new char[lxn+1];
   strcpy(name,nam);
   cflag=0;
   oflag=0;
   pflag=get_qflag();
   if(*pt!=':')set_path_name(pt);
   else set_path_internal(pt+1);
}

FBase::~FBase(void){
   delete [] type;
   delete [] name;
}

void FBase::set_type_num(int tn){tpnm=tn;}

void FBase::set_name_num(int nn){nmnm=nn;}

void FBase::change_type(const char *typ){
   if(type!=NULL)delete [] type;
   int lxn=strlen(typ);
   type=new char[lxn+1];
   strcpy(type,typ);
}

void FBase::change_name(const char *nam){
   if(name!=NULL)delete [] name;
   int lxn=strlen(nam);
   name=new char[lxn+1];
   strcpy(name,nam);
}

void FBase::set_name(const char *nam){
   if(name!=NULL)delete [] name;
   int lxn=strlen(nam);
   name=new char[lxn+1];
   strcpy(name,nam);
}

void FBase::subname(const char *tph,const char *tpl,const char *nm){
   char cnam[max_str];
   long i=strlen(tpl);
   strcpy(cnam,tpl);
   cnam[i]='_';
   cnam[i+1]='\0';
   strcat(cnam,nm);
   change_type(tph);
   change_name(cnam);
}

void FBase::set_path_internal(const char *pt){
   long len;
   if(pt&&(len=strlen(pt))){
      eflag=0;
      path=new char[len+1];
      strcpy(path,pt);
   }
   else eflag=1;
}

void FBase::set_path_name(const char *pa){
   long len;
   if(pa&&(len=strlen(pa))){
      eflag=2;
      pnam=new char[len+1];
      strcpy(pnam,pa);
   }
   else eflag=1;
}

void FBase::map_down(FBase *pFb){
   pFb->change_type(type);
   pFb->change_name(name);
   pFb->set_type_num(tpnm);
   pFb->set_name_num(nmnm);
   pFb->pflag=pflag;
   if(eflag==2)pFb->set_path_name(pnam);
   else if(!eflag)pFb->set_path_internal(path);
}

void FBase::map_down_sub(FBase *pFb,const char *subtype){
   pFb->subname(type,name,subtype);
   pFb->set_type_num(tpnm);
   pFb->set_name_num(nmnm);
   pFb->pflag=pflag;
   if(eflag==2)pFb->set_path_name(pnam);
   else if(!eflag)pFb->set_path_internal(path);
}

void FBase::get_pathx(char *nam,const char *ch){
   char cnam[256];
   ifstream fin;

   if(eflag==2){
      strcpy(cnam,"path_");
      strcat(cnam,pnam);
      fin.open(cnam,ios::in);
      if(!fin.is_open()){
         fin.clear();
         strcpy(cnam,"path");
         fin.open(cnam,ios::in);
         if(!fin.is_open()){
            cout << "Path file for type " << type << " does not exist!" << endl;
            exit(0);
         }
      }
      fin.getline(nam,256);
      fin.close();
   }
   else if(eflag){
      strcpy(cnam,"path_");
      strcat(cnam,type);
      strcat(cnam,"_");
      strcat(cnam,name);
      strcat(cnam,".");
      strcat(cnam,ch);
      fin.open(cnam,ios::in);
      if(!fin.is_open()){
         fin.clear();
         strcpy(cnam,"path_");
         strcat(cnam,type);
         strcat(cnam,"_");
         strcat(cnam,name);
         fin.open(cnam,ios::in);
         if(!fin.is_open()){
	    fin.clear();
            strcpy(cnam,"path_");
            strcat(cnam,type);
            fin.open(cnam,ios::in);
            if(!fin.is_open()){
               fin.clear();
               strcpy(cnam,"path");
               fin.open(cnam,ios::in);
               if(!fin.is_open()){
                  cout << "Path file for type " << type << " does not exist!" << endl;
                  exit(0);
               }
            }
         }
      }
      fin.getline(nam,256);
      fin.close();
   }
   else {
      strcpy(nam,path);
   }

   if(tpnm<0)strcat(nam,type);
   else cat_num(type,tpnm,nam);
   strcat(nam,"_");
   if(nmnm<0)strcat(nam,name);
   else cat_num(name,nmnm,nam);
   strcat(nam,".");
   strcat(nam,ch);
}

void FBase::get_pathx(char *nam,long n,const char *ch){
   char cnam[256],bnam[256];
   ifstream fin;

   if(eflag==2){
      strcpy(cnam,"path_");
      strcat(cnam,pnam);
      fin.open(cnam,ios::in);
      if(!fin.is_open()){
         fin.clear();
         strcpy(cnam,"path");
         fin.open(cnam,ios::in);
         if(!fin.is_open()){
            cout << "Path file for type " << type << " does not exist!" << endl;
            exit(0);
         }
      }
      fin.getline(nam,256);
      fin.close();
   }
   else if(eflag){
      strcpy(cnam,"path_");
      strcat(cnam,type);
      strcat(cnam,"_");
      strcat(cnam,name);
      strcat(cnam,".");
      strcat(cnam,ch);
      fin.open(cnam,ios::in);
      if(!fin.is_open()){
         fin.clear();
         strcpy(cnam,"path_");
         strcat(cnam,type);
         strcat(cnam,"_");
         strcat(cnam,name);
         fin.open(cnam,ios::in);
         if(!fin.is_open()){
            fin.clear();
            strcpy(cnam,"path_");
            strcat(cnam,type);
            fin.open(cnam,ios::in);
            if(!fin.is_open()){
               fin.clear();
               strcpy(cnam,"path");
               fin.open(cnam,ios::in);
               if(!fin.is_open()){
                  cout << "Path file for type " << type << " does not exist!" << endl;
                  exit(0);
               }
            }
         }
      }
      fin.getline(nam,256);
      fin.close();
   }
   else {
      strcpy(nam,path);
   }

   if(tpnm<0)strcat(nam,type);
   else cat_num(type,tpnm,nam);
   strcat(nam,"_");
   strcat(nam,add_num(name,n,bnam));
   strcat(nam,".");
   strcat(nam,ch);
}

char *FBase::add_num(const char *ptr,long n,char *buf){
   char cnam[100];
   long_str(cnam,n);
   strcpy(buf,ptr);
   strcat(buf,cnam);
   return(buf);
}

char *FBase::cat_num(const char *ptr,long n,char *buf){
   char cnam[100];
   long_str(cnam,n);
   strcat(buf,ptr);
   strcat(buf,cnam);
   return(buf);
}

int FBase::Gcom(int sflag){
   if((cflag&sflag)&&!(oflag&sflag)){
      oflag=oflag|sflag;
      return(1);
   }
   else return(0);
}

int FBase::Rcom(int sflag){
   if((cflag&sflag)&&(oflag&sflag)){
      oflag=oflag&(~sflag);
      return(1);
   }
   else return(0);
}

ifstream *FBase::get_Istr(const char *a,ios::openmode mode){
   char cnam[max_str];
   get_pathx(cnam,a);
   ifstream *pfin=new ifstream(cnam,mode);
   if(pfin->is_open())return(pfin);
   else {
      cout << "Error: " << cnam << " failed to open!" << endl;
      exit(0);
   }
}

ofstream *FBase::get_Ostr(const char *a,ios::openmode mode){
   char cnam[max_str];
   get_pathx(cnam,a);
   ofstream *pfout=new ofstream(cnam,mode);
   if(pfout->is_open())return(pfout);
   else {
      cout << "Error: " << cnam << " failed to open!" << endl;
      exit(0);
   }
}

fstream *FBase::get_Fstr(const char *a,ios::openmode mode){
   char cnam[max_str];
   get_pathx(cnam,a);
   fstream *pfstr=new fstream(cnam,mode);
   if(pfstr->is_open())return(pfstr);
   else {
      cout << "Error: " << cnam << " failed to open!" << endl;
      exit(0);
   }
}

ifstream *FBase::get_Istr(long n,const char *a,ios::openmode mode){
   char cnam[max_str];
   get_pathx(cnam,n,a);
   ifstream *pfin=new ifstream(cnam,mode);
   if(pfin->is_open())return(pfin);
   else {
      cout << "Error: " << cnam << " failed to open!" << endl;
      exit(0);
   }
}

ofstream *FBase::get_Ostr(long n,const char *a,ios::openmode mode){
   char cnam[max_str];
   get_pathx(cnam,n,a);
   ofstream *pfout=new ofstream(cnam,mode);
   if(pfout->is_open())return(pfout);
   else {
      cout << "Error: " << cnam << " failed to open!" << endl;
      exit(0);
   }
}

fstream *FBase::get_Fstr(long n,const char *a,ios::openmode mode){
   char cnam[max_str];
   get_pathx(cnam,n,a);
   fstream *pfstr=new fstream(cnam,mode);
   if(pfstr->is_open())return(pfstr);
   else {
      cout << "Error: " << cnam << " failed to open!" << endl;
      exit(0);
   }
}

void FBase::dst_Istr(ifstream *pfin){
   if(!pfin)return;
   if(!pfin->is_open()){
      cout << "File not open!" << endl;
      exit(0);
   }
   delete pfin;
}

void FBase::dst_Ostr(ofstream *pfout){
   if(!pfout)return;
   if(!pfout->is_open()){
      cout << "File not open!" << endl;
      exit(0);
   }
   delete pfout;
}

void FBase::dst_Fstr(fstream *pfstr){
   if(!pfstr)return;
   if(!pfstr->is_open()){
      cout << "File not open!" << endl;
      exit(0);
   }
   delete pfstr;
}

long FBase::get_Fsiz(const char *a){
   if(!Exists(a))return(0);
   int fld;
   struct stat datf;
   char cnam[max_str];
   get_pathx(cnam,a);
   fld=::open(cnam,O_RDONLY);
   if(fld<=0){cout << cnam << " failed to open" << endl;exit(0);}
   if(fstat(fld,&datf)){cout << cnam << " failed on size \
      determination" << endl;exit(0);}
   ::close(fld);
   return(datf.st_size);
}

long FBase::get_Fsiz(long n,const char *a){
   if(!Exists(n,a))return(0);
   int fld;
   struct stat datf;
   char cnam[max_str];
   get_pathx(cnam,n,a);
   fld=::open(cnam,O_RDONLY);
   if(fld<=0){cout << cnam << " failed to open" << endl;exit(0);}
   if(fstat(fld,&datf)){cout << cnam << " failed on size \
      determination" << endl;exit(0);}
   ::close(fld);
   return(datf.st_size);
}

char *FBase::get_Read(const char *a){
   int fld;
   struct stat datf;
   char cnam[max_str];
   get_pathx(cnam,a);
   fld=::open(cnam,O_RDONLY);
   if(fld<=0){cout << cnam << " failed to open" << endl;exit(0);}
   if(fstat(fld,&datf)){cout << cnam << " failed on size \
      determination" << endl;exit(0);}
   ::close(fld);
   char *ptr=new char[datf.st_size];
   ifstream fin(cnam,ios::in);
   if(!fin.is_open()){
      cout << "Error: " << cnam << " failed to open!" << endl;
      exit(0);
   }
   fin.read(ptr,datf.st_size);
   return(ptr);
}

char *FBase::get_Read(long n,const char *a){
   int fld;
   struct stat datf;
   char cnam[max_str];
   get_pathx(cnam,n,a);
   fld=::open(cnam,O_RDONLY);
   if(fld<=0){cout << cnam << " failed to open" << endl;exit(0);}
   if(fstat(fld,&datf)){cout << cnam << " failed on size \
      determination" << endl;exit(0);}
   ::close(fld);
   char *ptr=new char[datf.st_size];
   ifstream fin(cnam,ios::in);
   if(!fin.is_open()){
      cout << "Error: " << cnam << " failed to open!" << endl;
      exit(0);
   }
   fin.read(ptr,datf.st_size);
   return(ptr);
}

char *FBase::get_Mmap(const char *a){
   int fld;
   struct stat datf;
   char cnam[max_str];
   get_pathx(cnam,a);
   fld=::open(cnam,O_RDONLY);
   if(fld<=0){cout << cnam << " failed to open" << endl;exit(0);}
   if(fstat(fld,&datf)){cout << cnam << " failed on size determination" << endl;exit(0);}
   char *ptr=(char*)mmap(0,datf.st_size,PROT_READ,MAP_PRIVATE|MAP_NORESERVE,fld,0);
   if(ptr==MAP_FAILED){cout << cnam << " failed to map" << endl;exit(0);}
   ::close(fld);
   return(ptr);
}

char *FBase::get_Mmap(long n,const char *a){
   int fld;
   struct stat datf;
   char cnam[max_str];
   get_pathx(cnam,n,a);
   fld=::open(cnam,O_RDONLY);
   if(fld<=0){cout << cnam << " failed to open" << endl;exit(0);}
   if(fstat(fld,&datf)){cout << cnam << " failed on size determination" << endl;exit(0);}
   char *ptr=(char*)mmap(0,datf.st_size,PROT_READ,MAP_PRIVATE|MAP_NORESERVE,fld,0);
   if(ptr==MAP_FAILED){cout << cnam << " failed to map" << endl;exit(0);}
   ::close(fld);
   return(ptr);
}

char *FBase::get_Wmap(const char *a){
   int fld;
   struct stat datf;
   char cnam[max_str];
   get_pathx(cnam,a);
   fld=::open(cnam,O_RDWR);
   if(fld<=0){cout << cnam << " failed to open" << endl;exit(0);}
   if(fstat(fld,&datf)){cout << cnam << " failed on size determination" << endl;exit(0);}
   char *ptr=(char*)mmap(0,datf.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fld,0);
   if(ptr==MAP_FAILED){cout << cnam << " failed to map" << endl;exit(0);}
   ::close(fld);
   return(ptr);
}  
   
char *FBase::get_Wmap(long n,const char *a){
   int fld;
   struct stat datf;
   char cnam[max_str];
   get_pathx(cnam,n,a);
   fld=::open(cnam,O_RDWR);
   if(fld<=0){cout << cnam << " failed to open" << endl;exit(0);}
   if(fstat(fld,&datf)){cout << cnam << " failed on size determination" << endl;exit(0);}
   char *ptr=(char*)mmap(0,datf.st_size,PROT_READ|PROT_WRITE,MAP_SHARED,fld,0);
   if(ptr==MAP_FAILED){cout << cnam << " failed to map" << endl;exit(0);}
   ::close(fld);
   return(ptr);
}

void FBase::dst_Mmap(const char *a,char *ptr){
   struct stat datf;
   char cnam[max_str];
   if(ptr==NULL){cout << "NULL pointer" << endl;return;}
   get_pathx(cnam,a);
   if(stat(cnam,&datf)){cout << cnam << " failed on size determination" << endl;exit(0);}
   if(munmap(ptr,datf.st_size)){cout << cnam << " failed to unmap" << endl;exit(0);}
   ptr=NULL;
}

void FBase::dst_Mmap(long n,const char *a,char *ptr){
   struct stat datf;
   char cnam[max_str];
   if(ptr==NULL){cout << "NULL pointer" << endl;return;}
   get_pathx(cnam,n,a);
   if(stat(cnam,&datf)){cout << cnam << " failed on size determination" << endl;exit(0);}
   if(munmap(ptr,datf.st_size)){cout << cnam << " failed to unmap" << endl;exit(0);}
   ptr=NULL;
}

void FBase::bin_Writ(const char *a,long nm,char *ptr){
   ofstream *pfout=get_Ostr(a,ios::out);
   long k=100000,i=0;
   while(i+k<nm){
      pfout->write((char*)ptr,k);
      i+=k;
      ptr=ptr+k;
   }
   pfout->write((char*)ptr,nm-i);
   pfout->close();
   delete pfout;
}

void FBase::bin_Writ(long n,const char *a,long nm,char *ptr){
   ofstream *pfout=get_Ostr(n,a,ios::out);
   long k=100000,i=0;
   while(i+k<nm){
      pfout->write((char*)ptr,k);
      i+=k;
      ptr=ptr+k;
   }
   pfout->write((char*)ptr,nm-i);
   pfout->close();
   delete pfout;
}

int FBase::Exists(const char *a){
   char cnam[max_str];
   get_pathx(cnam,a);
   ifstream fin(cnam,ios::in);
   if(fin.is_open()){
      fin.close();
      return(1);
   }
   else return(0);
}

int FBase::Exists(long n,const char *a){
   char cnam[max_str];
   get_pathx(cnam,n,a);
   ifstream fin(cnam,ios::in);
   if(fin.is_open()){
      fin.close();
      return(1);
   }
   else return(0);
}

void FBase::mark(long ct, int ivl, const char *what){
   if(pflag&&(ct%ivl==0)){
      cout << what << " count=" << ct << endl;
   }
}
   
}
