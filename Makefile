CXXFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		fft.o

LIBS = -lfftw3f

TARGET =	fft

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
