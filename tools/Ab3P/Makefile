LIBPATH=./Library
OS=-g
# OS=-O


.KEEP_STATE:

%:%.o
	g++ $(OS) -o $@ $< -L$(LIBPATH) -lops -lpthread

%.o: %.C
	g++ -c $(OS) -I$(LIBPATH) $< -o $@

all:
	make programs
	make data

programs:
	make library
	make make_wordSet
	make make_wordCountHash
	make identify_abbr

library:
	cd Library; make

data:
	./make_wordSet WordData/stop stop
	./make_wordSet WordData/Lf1chSf Lf1chSf
	./make_wordCountHash WordData/SingTermFreq.dat


test:
	./identify_abbr MED1250_unlabeled | diff identify_abbr-out -

test2:
	./identify_abbr test2.in | diff test2.out -

make_wordSet:		make_wordSet.o
make_wordCountHash:	make_wordCountHash.o
identify_abbr:		identify_abbr.o
