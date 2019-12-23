#include <iostream> 
#include <fstream> 
#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cmath>
#include <cstring>
#include <cassert>
#include "runn.h"
#include "Hash.h"

using namespace std;
namespace iret {

Hash::Hash(void) : FBase("hshset","null"){
}

Hash::Hash(const char *nam) : FBase("hshset",nam){
}

Hash::Hash(int n,const char *nam) : FBase("hshset",n,nam){
}

Hash::~Hash(){
}

void Hash::create_htable(List &Lst,int excess){
   char cnam[max_str],*cptr,*uptr;
   int u,len;
   long ct,i,j,k;
   ofstream *pfout;

   nwrds=Lst.cnt_key;
   ct=nwrds;
   tnum=1;
   u=0;
   while(ct=ct/2){tnum*=2;u++;}
   if(u>30){cout << "Error in size, " << u << endl;exit(0);}
   i=0;
   while((u<32)&&(i<excess)){tnum*=2;u++;i++;}
   tnum--;
   harr=new long[tnum+2];
   for(ct=0;ct<tnum+2;ct++)harr[ct]=0;

   farr=new long[1536];
   ct=1;
   for(i=0;i<1536;i++){
      farr[i]=ct=(ct*331)&tnum;
   }
  
   long *pc0=farr,*pc1=farr+128,*pc2=farr+256;
   long *pc3=farr+384,*pc4=farr+512,*pc5=farr+640;
   long *pc6=farr+768,*pc7=farr+896,*pc8=farr+1024;
   long *pc9=farr+1152,*pc10=farr+1280,*pc11=farr+1408;
   
   Lst.node_first();
   while(Lst.node_next()){
      cptr=Lst.show_str();
      ct=0;
      i=0;
      while(u=*(cptr++)){
         switch(i){
            case 0: ct+=*(pc0+u);
                    break;
            case 1: ct+=*(pc1+u);
                    break;
            case 2: ct+=*(pc2+u);
                    break;
            case 3: ct+=*(pc3+u);
                    break;
            case 4: ct+=*(pc4+u);
                    break;
            case 5: ct+=*(pc5+u);
                    break;
            case 6: ct+=*(pc6+u);
                    break;
            case 7: ct+=*(pc7+u);
                    break;
            case 8: ct+=*(pc8+u);
                    break;
            case 9: ct+=*(pc9+u);
                    break;
            case 10: ct+=*(pc10+u);
                    break;
            case 11: ct+=*(pc11+u);
                     i-=12;
                    break;
         }
         i++;
      }
      (harr[ct&tnum])++;
   }

   //Set start points in harr.
   k=0;
   for(i=0;i<tnum+2;i++){
      j=harr[i];
      harr[i]=k;
      k+=j;
   }
   if(k!=nwrds){cout << "Error in summing!" << endl;exit(0);}

   //Write out harr.
   bin_Writ("ha",(tnum+2)*sizeof(long),(char*)harr);

   //Set addresses
   char **addt=new char*[nwrds];
   Lst.node_first();
   while(Lst.node_next()){
      uptr=cptr=Lst.show_str();
      ct=0;
      i=0;
      while(u=*(cptr++)){
         switch(i){
            case 0: ct+=*(pc0+u);
                    break;
            case 1: ct+=*(pc1+u);
                    break;
            case 2: ct+=*(pc2+u);
                    break;
            case 3: ct+=*(pc3+u);
                    break;
            case 4: ct+=*(pc4+u);
                    break;
            case 5: ct+=*(pc5+u);
                    break;
            case 6: ct+=*(pc6+u);
                    break;
            case 7: ct+=*(pc7+u);
                    break;
            case 8: ct+=*(pc8+u);
                    break;
            case 9: ct+=*(pc9+u);
                    break;
            case 10: ct+=*(pc10+u);
                    break;
            case 11: ct+=*(pc11+u);
                     i-=12;
                    break;
         }
         i++;
      }
      k=ct&tnum;
      addt[harr[k]]=uptr;
      (harr[k])++;
   }

   //Write out string file
   pfout=get_Ostr("str");
   k=0;
   for(i=0;i<nwrds;i++){
      *pfout << addt[i] << ends;
      len=strlen((char*)addt[i])+1;
      addt[i]=(char*)k;
      k+=len;
   }
   dst_Ostr(pfout);

   //Write out addr file
   bin_Writ("ad",nwrds*sizeof(long),(char*)addt);
   delete [] addt;

   //Write out counts
   pfout=get_Ostr("nm");
   *pfout << nwrds << " " << tnum << " " << k << endl;
   dst_Ostr(pfout);
   delete [] harr;
   delete [] farr;
}

//In memory model intended for small sets
void Hash::create_htableM(List &Lst,int excess){
   char cnam[max_str],*cptr,*uptr;
   int u,len;
   long ct,i,j,k,*barr;
   ofstream *pfout;

   nwrds=Lst.cnt_key;
   ct=nwrds;
   tnum=1;
   u=0;
   while(ct=ct/2){tnum*=2;u++;}
   if(u>30){cout << "Error in size, " << u << endl;exit(0);}
   i=0;
   while((u<32)&&(i<excess)){tnum*=2;u++;i++;}
   tnum--;
   harr=new long[tnum+2];
   barr=new long[tnum+2];
   for(ct=0;ct<tnum+2;ct++)harr[ct]=0;

   farr=new long[1536];
   ct=1;
   for(i=0;i<1536;i++){
      farr[i]=ct=(ct*331)&tnum;
   }
  
   px0=farr,px1=farr+128,px2=farr+256;
   px3=farr+384,px4=farr+512,px5=farr+640;
   px6=farr+768,px7=farr+896,px8=farr+1024;
   px9=farr+1152,px10=farr+1280,px11=farr+1408;
   
   Lst.node_first();
   while(Lst.node_next()){
      cptr=Lst.show_str();
      ct=0;
      i=0;
      while(u=*(cptr++)){
         switch(i){
            case 0: ct+=*(px0+u);
                    break;
            case 1: ct+=*(px1+u);
                    break;
            case 2: ct+=*(px2+u);
                    break;
            case 3: ct+=*(px3+u);
                    break;
            case 4: ct+=*(px4+u);
                    break;
            case 5: ct+=*(px5+u);
                    break;
            case 6: ct+=*(px6+u);
                    break;
            case 7: ct+=*(px7+u);
                    break;
            case 8: ct+=*(px8+u);
                    break;
            case 9: ct+=*(px9+u);
                    break;
            case 10: ct+=*(px10+u);
                    break;
            case 11: ct+=*(px11+u);
                     i-=12;
                    break;
         }
         i++;
      }
      (harr[ct&tnum])++;
   }

   //Set start points in harr.
   k=0;
   for(i=0;i<tnum+2;i++){
      j=harr[i];
      barr[i]=harr[i]=k;
      k+=j;
   }
   if(k!=nwrds){cout << "Error in summing!" << endl;exit(0);}

   //Set addresses
   len=0;
   char **addt=new char*[nwrds];
   Lst.node_first();
   while(Lst.node_next()){
      uptr=cptr=Lst.show_str();
      len+=strlen(uptr)+1;
      ct=0;
      i=0;
      while(u=*(cptr++)){
         switch(i){
            case 0: ct+=*(px0+u);
                    break;
            case 1: ct+=*(px1+u);
                    break;
            case 2: ct+=*(px2+u);
                    break;
            case 3: ct+=*(px3+u);
                    break;
            case 4: ct+=*(px4+u);
                    break;
            case 5: ct+=*(px5+u);
                    break;
            case 6: ct+=*(px6+u);
                    break;
            case 7: ct+=*(px7+u);
                    break;
            case 8: ct+=*(px8+u);
                    break;
            case 9: ct+=*(px9+u);
                    break;
            case 10: ct+=*(px10+u);
                    break;
            case 11: ct+=*(px11+u);
                     i-=12;
                    break;
         }
         i++;
      }
      k=ct&tnum;
      addt[barr[k]]=uptr;
      (barr[k])++;
   }
   strmap=new char[len];

   //Set up string array
   k=0;
   for(i=0;i<nwrds;i++){
      len=strlen((char*)addt[i])+1;
      strcpy(strmap+k,addt[i]);
      addt[i]=(char*)k;
      k+=len;
   }
   addr=(long*)addt;
   delete [] barr;
}

void Hash::create_htable(int mz,List &Lst,int excess){
   char cnam[max_str],*cptr,*uptr;
   int u,len;
   long ct,i,j,k;
   ofstream *pfout;

   nwrds=Lst.cnt_key;
   ct=nwrds;
   tnum=1;
   u=0;
   while(ct=ct/2){tnum*=2;u++;}
   if(u>30){cout << "Error in size, " << u << endl;exit(0);}
   i=0;
   while((u<32)&&(i<excess)){tnum*=2;u++;i++;}
   tnum--;
   harr=new long[tnum+2];
   for(ct=0;ct<tnum+2;ct++)harr[ct]=0;

   farr=new long[1536];
   ct=1;
   for(i=0;i<1536;i++){
      farr[i]=ct=(ct*331)&tnum;
   }
  
   long *pc0=farr,*pc1=farr+128,*pc2=farr+256;
   long *pc3=farr+384,*pc4=farr+512,*pc5=farr+640;
   long *pc6=farr+768,*pc7=farr+896,*pc8=farr+1024;
   long *pc9=farr+1152,*pc10=farr+1280,*pc11=farr+1408;
   
   Lst.node_first();
   while(Lst.node_next()){
      cptr=Lst.show_str();
      ct=0;
      i=0;
      while(u=*(cptr++)){
         switch(i){
            case 0: ct+=*(pc0+u);
                    break;
            case 1: ct+=*(pc1+u);
                    break;
            case 2: ct+=*(pc2+u);
                    break;
            case 3: ct+=*(pc3+u);
                    break;
            case 4: ct+=*(pc4+u);
                    break;
            case 5: ct+=*(pc5+u);
                    break;
            case 6: ct+=*(pc6+u);
                    break;
            case 7: ct+=*(pc7+u);
                    break;
            case 8: ct+=*(pc8+u);
                    break;
            case 9: ct+=*(pc9+u);
                    break;
            case 10: ct+=*(pc10+u);
                    break;
            case 11: ct+=*(pc11+u);
                     i-=12;
                    break;
         }
         i++;
      }
      (harr[ct&tnum])++;
   }

   //Set start points in harr.
   k=0;
   for(i=0;i<tnum+2;i++){
      j=harr[i];
      harr[i]=k;
      k+=j;
   }
   if(k!=nwrds){cout << "Error in summing!" << endl;exit(0);}

   //Write out harr.
   bin_Writ(mz,"ha",(tnum+2)*sizeof(long),(char*)harr);

   //Set addresses
   char **addt=new char*[nwrds];
   Lst.node_first();
   while(Lst.node_next()){
      uptr=cptr=Lst.show_str();
      ct=0;
      i=0;
      while(u=*(cptr++)){
         switch(i){
            case 0: ct+=*(pc0+u);
                    break;
            case 1: ct+=*(pc1+u);
                    break;
            case 2: ct+=*(pc2+u);
                    break;
            case 3: ct+=*(pc3+u);
                    break;
            case 4: ct+=*(pc4+u);
                    break;
            case 5: ct+=*(pc5+u);
                    break;
            case 6: ct+=*(pc6+u);
                    break;
            case 7: ct+=*(pc7+u);
                    break;
            case 8: ct+=*(pc8+u);
                    break;
            case 9: ct+=*(pc9+u);
                    break;
            case 10: ct+=*(pc10+u);
                    break;
            case 11: ct+=*(pc11+u);
                     i-=12;
                    break;
         }
         i++;
      }
      k=ct&tnum;
      addt[harr[k]]=uptr;
      (harr[k])++;
   }

   //Write out string file
   pfout=get_Ostr(mz,"str");
   k=0;
   for(i=0;i<nwrds;i++){
      *pfout << addt[i] << ends;
      len=strlen((char*)addt[i])+1;
      addt[i]=(char*)k;
      k+=len;
   }
   dst_Ostr(pfout);

   //Write out addr file
   bin_Writ(mz,"ad",nwrds*sizeof(long),(char*)addt);
   delete [] addt;

   //Write out counts
   pfout=get_Ostr(mz,"nm");
   *pfout << nwrds << " " << tnum << " " << k << endl;
   dst_Ostr(pfout);
   delete [] harr;
   delete [] farr;
}

void Hash::gopen_htable_map(void){
   char cnam[max_str],*cptr;
   int fld;
   long ct,asize,i;
   
   ifstream *pfin=get_Istr("nm");
   *pfin >> nwrds >> tnum >> asize;
   dst_Istr(pfin);

   harr=(long*)get_Mmap("ha");
   addr=(long*)get_Mmap("ad");
   strmap=get_Mmap("str");

   farr=new long[1536];
   ct=1;
   for(i=0;i<1536;i++){
      farr[i]=ct=(ct*331)&tnum;
   }

   px0=farr,px1=farr+128,px2=farr+256;
   px3=farr+384,px4=farr+512,px5=farr+640;
   px6=farr+768,px7=farr+896,px8=farr+1024;
   px9=farr+1152,px10=farr+1280,px11=farr+1408;
}

void Hash::gopen_htable_map(int mz){
   char cnam[max_str],*cptr;
   int fld;
   long ct,asize,i;
   
   ifstream *pfin=get_Istr(mz,"nm");
   *pfin >> nwrds >> tnum >> asize;
   dst_Istr(pfin);

   harr=(long*)get_Mmap(mz,"ha");
   addr=(long*)get_Mmap(mz,"ad");
   strmap=get_Mmap(mz,"str");

   farr=new long[1536];
   ct=1;
   for(i=0;i<1536;i++){
      farr[i]=ct=(ct*331)&tnum;
   }

   px0=farr,px1=farr+128,px2=farr+256;
   px3=farr+384,px4=farr+512,px5=farr+640;
   px6=farr+768,px7=farr+896,px8=farr+1024;
   px9=farr+1152,px10=farr+1280,px11=farr+1408;
}

void Hash::gopen_htable_copy(Hash *pH){
   char cnam[max_str],*cptr;
   int fld;
   long ct,asize,i;

   nwrds=pH->nwrds;
   tnum=pH->tnum;

   harr=pH->harr;
   addr=pH->addr;
   strmap=pH->strmap;

   farr=pH->farr;

   px0=farr,px1=farr+128,px2=farr+256;
   px3=farr+384,px4=farr+512,px5=farr+640;
   px6=farr+768,px7=farr+896,px8=farr+1024;
   px9=farr+1152,px10=farr+1280,px11=farr+1408;
}

long Hash::find(const char *str){
   register long ct=0,i=0,k;
   register int ic;
   register const char *utr=str;
   while(ic=*(utr++)){
      switch(i){
         case 0: ct+=*(px0+ic);
                 break;
         case 1: ct+=*(px1+ic);
                 break;
         case 2: ct+=*(px2+ic);
                 break;
         case 3: ct+=*(px3+ic);
                 break;
         case 4: ct+=*(px4+ic);
                 break;
         case 5: ct+=*(px5+ic);
                 break;
         case 6: ct+=*(px6+ic);
                 break;
         case 7: ct+=*(px7+ic);
                 break;
         case 8: ct+=*(px8+ic);
                 break;
         case 9: ct+=*(px9+ic);
                 break;
         case 10: ct+=*(px10+ic);
                 break;
         case 11: ct+=*(px11+ic);
                  i-=12;
                 break;
      }
      i++;
   }
   k=ct&tnum;
   ct=harr[k+1];
   i=harr[k];
//cout << k << " " << i << " " << addr[i] << " " << ct << " " << addr[ct] << endl;
   switch(ct-i){
      case 0: return(0);
              break;
      case 1: if(!strcmp(str,strmap+addr[i]))return(i+1);
              else return(0);
              break;
      case 2: ic=strcmp(str,strmap+addr[i]);
              if(ic>0){
                 if(!strcmp(str,strmap+addr[i+1]))return(i+2);
                 else return(0);
              }
              else if(ic<0)return(0);
              else return(i+1);
              break;
      default: ic=strcmp(str,strmap+addr[i]);
               if(ic<0)return(0);
               else if(!ic)return(i+1);
               ct--;
               ic=strcmp(str,strmap+addr[ct]);
               if(ic>0)return(0);
               else if(!ic)return(ct+1);
               while(ct-i>1){
                  k=(ct+i)/2;
                  ic=strcmp(str,strmap+addr[k]);
                  if(ic>0)i=k;
                  else if(ic<0)ct=k;
                  else return(k+1);
               }
               return(0);
   }
}

void Hash::gclose_htable_map(void){
   dst_Mmap("ha",(char*)harr);
   dst_Mmap("ad",(char*)addr);
   dst_Mmap("str",strmap);
   delete [] farr;
}

void Hash::gclose_htable_map(int mz){
   dst_Mmap(mz,"ha",(char*)harr);
   dst_Mmap(mz,"ad",(char*)addr);
   dst_Mmap(mz,"str",strmap);
   delete [] farr;
}

//Chash code

Chash::Chash() : Hash(){
   change_type("cshset");
}

Chash::Chash(const char *str) : Hash(str){
   change_type("cshset");
}

Chash::Chash(int n,const char *str) : Hash(n,str){
   change_type("cshset");
}

Chash::~Chash(void){}

void Chash::create_ctable(Count &Ct,int excess){
   create_htable(Ct,excess);
   gopen_htable_map();
   long n,i=0;
   long *pct=new long[Ct.cnt_key];
   Ct.node_first();
   while(Ct.node_next()){
      if(n=find(Ct.show_str())){
         pct[n-1]=Ct.count();
      }        
      else {
         cout << "Error in Count tree!" << endl;exit(0);
      }
      mark(++i,10000,"count terms");
   }
   bin_Writ("ct",Ct.cnt_key*sizeof(long),(char*)pct);
   delete [] pct;
   cnt=(long*)get_Mmap("ct");
   gclose_htable_map();
}

void Chash::create_ctable(List &Lt,int excess){
   create_htable(Lt,excess);
   gopen_htable_map();
   long n,i=1;
   long *pct=new long[Lt.cnt_key];
   Lt.node_first();
   while(Lt.node_next()){
      if(n=find(Lt.show_str())){
         pct[n-1]=i;
      }
      else {
         cout << "Error in List tree!" << endl;exit(0);
      }
      mark(++i,10000,"count terms");
   }
   bin_Writ("ct",Lt.cnt_key*sizeof(long),(char*)pct);
   delete [] pct;
   cnt=(long*)get_Mmap("ct");
   gclose_htable_map();
}

void Chash::create_ctable(int mz,Count &Ct,int excess){
   create_htable(mz,Ct,excess);
   gopen_htable_map(mz);
   long n,i=0;
   long *pct=new long[Ct.cnt_key];
   Ct.node_first();
   while(Ct.node_next()){
      if(n=find(Ct.show_str())){
         pct[n-1]=Ct.count();
      }        
      else {
         cout << "Error in Count tree!" << endl;exit(0);
      }
      mark(++i,10000,"count terms");
   }
   bin_Writ(mz,"ct",Ct.cnt_key*sizeof(long),(char*)pct);
   delete [] pct;
   cnt=(long*)get_Mmap(mz,"ct");
   gclose_htable_map(mz);
}

void Chash::create_ctable(int mz,List &Lt,int excess){
   create_htable(mz,Lt,excess);
   gopen_htable_map(mz);
   long n,i=1;
   long *pct=new long[Lt.cnt_key];
   Lt.node_first();
   while(Lt.node_next()){
      if(n=find(Lt.show_str())){
         pct[n-1]=i;
      }
      else {
         cout << "Error in List tree!" << endl;exit(0);
      }
      mark(++i,10000,"count terms");
   }
   bin_Writ(mz,"ct",Lt.cnt_key*sizeof(long),(char*)pct);
   delete [] pct;
   cnt=(long*)get_Mmap(mz,"ct");
   gclose_htable_map(mz);
}

void Chash::gopen_ctable_map(void){
   gopen_htable_map();
   cnt=(long*)get_Mmap("ct");
}   

void Chash::gopen_ctable_map(int mz){
   gopen_htable_map(mz);
   cnt=(long*)get_Mmap(mz,"ct");
}   

void Chash::gclose_ctable_map(void){
   gclose_htable_map();
   dst_Mmap("ct",(char*)cnt);
}   

void Chash::gclose_ctable_map(int mz){
   gclose_htable_map(mz);
   dst_Mmap(mz,"ct",(char*)cnt);
}   

long Chash::count(const char *str){
   long n=find(str);
   if(n)return(cnt[n-1]);
   else return(0);
}

}

