#CC=icc
#CCFLAGS =	-fast -Wall -fmessage-length=0
#LIBS = -lfftw3f

CC=mpicc
CCFLAGS =	-O2 -g -Wall -fmessage-length=0
LIBS = -lfftw3f

#CC=cc

OBJS =		fft.o


TARGET =	fft



$(TARGET):	$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
