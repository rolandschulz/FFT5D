CC=icc
CCFLAGS =	-fast -Wall -fmessage-length=0

#CCFLAGS =	-O2 -g -Wall -fmessage-length=0

OBJS =		fft.o

LIBS = -lfftw3f

TARGET =	fft

CC=mpicc

$(TARGET):	$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
