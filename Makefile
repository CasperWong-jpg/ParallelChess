SOURCES := src/*.c
HEADERS := src/*.h

OUTPUTDIR := bin
LICHESSDIR := lichess_bot/engines

COMPILER := gcc
CFLAGS := -std=c17 -fopenmp
DEBUGFLAGS := -Wall -Wextra -Werror -Wno-cast-function-type -Wshadow -std=c99 
DEBUGFLAGS += -pedantic -g -fwrapv

all: playable lichess

# Create shared object file that can be called by Python function
lichess: $(SOURCES) $(HEADERS)
	mkdir -p $(LICHESSDIR)
	$(COMPILER) -o $(LICHESSDIR)/ChessEngine.so -fPIC -shared $(CFLAGS) $(SOURCES)

# Create command line executable to simulate gameplay
playable: $(SOURCES) $(HEADERS)
	mkdir -p $(OUTPUTDIR)
	$(COMPILER) -o $(OUTPUTDIR)/ChessEngine.o $(CFLAGS) $(DEBUGFLAGS) $(SOURCES)

clean:
	rm -rf $(OUTPUTDIR)
	rm -f $(LICHESSDIR)/ChessEngine.so