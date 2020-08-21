SRC_DIR=./
TRASHFILES = *.o *~ *.bak core	
LIB_INC=-I./
#.KEEP_STATE:
libops.a: runn.o Btree.o FBase.o Hash.o MPtok.o \
	AbbrStra.o AbbrvE.o Ab3P.o
	ar rus $@ $?
OS=-g
%.o: $(SRC_DIR)/%.C
	g++ -c $(OS) $< -o $@ $(LIB_INC)

clean: rm -f $(TRASHFILES)

