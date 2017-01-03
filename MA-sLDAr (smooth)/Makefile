CC = g++
LDFLAGS = -g -lgsl -lm -lgslcblas


LSOURCE = main.cpp corpus.cpp maslda.cpp utils.cpp spdinverse.cpp stdafx.cpp ap.cpp cholesky.cpp
LHEADER = corpus.h maslda.h utils.h settings.h spdinverse.h stdafx.h ap.h cholesky.h

maslda: $(LSOURCE) $(HEADER)
	  $(CC) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	-rm -f *.o maslda
