CC = g++
CPPFLAGS = -O2 -W -Wall
LDLIBS = pfmLib.a -lopencv_core -lopencv_highgui -lopencv_imgproc

BIN = imdiff
all: $(BIN)

clean:
	rm -f $(BIN) *.o core*
