CC = g++
LDFLAGS = -lgsl -lm -lgslcblas


LSOURCE = main.cpp corpus.cpp slda.cpp utils.cpp spdinverse.cpp stdafx.cpp ap.cpp cholesky.cpp
LHEADER = corpus.h slda.h utils.h settings.h spdinverse.h stdafx.h ap.h cholesky.h

slda: $(LSOURCE) $(HEADER)
	  $(CC) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	-rm -f *.o slda
