#CC=icc
#CCFLAGS =	-fast -Wall -fmessage-length=0
#LIBS = -lfftw3f

CC=mpicc

#CFLAGS =	 -g -Wall  -I $(HOME)/usr/include/ -DFFT_MPI_TRANSPOSE -DPFFT_SINGLE
#LIBS = -lfftw3f_mpi -lfftw3f -L $(HOME)/usr/lib

CFLAGS =	-O2 -g -Wall  
LIBS = -lfftw3 


#CC=cc

OBJS =		fft.o fftlib.o
#OBJS =		fft.o 

TARGET =	fft



$(TARGET):	$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
