#CC=icc
#CFLAGS =	-fast -Wall -fmessage-length=0
#LIBS = -lfftw3f

CC=mpicc

#CFLAGS =	 -g -Wall  -I $(HOME)/usr/include/ -DFFT_MPI_TRANSPOSE -DFFT5D_SINGLE
#LIBS = -lfftw3f_mpi -lfftw3f -L $(HOME)/usr/lib

CFLAGS =	-O2 -g -Wall 
LIBS = -lfftw3 


CC=cc
CFLAGS = $(FFTW_INCLUDE_OPTS)
LIBS = $(FFTW_POST_LINK_OPTS)

#CFLAGS=-fastsse -I$(HOME)/fftw32/include -DFFT5D_MPI_TRANSPOSE -DFFT5D_SINGLE
#LIBS = -L$(HOME)/fftw32/lib -lfftw3f_mpi -lfftw3f 

OBJS =		testfft5d.o fft5d.o
#OBJS =		fft.o 

TARGET =	fft_fftw_trans



$(TARGET):	$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
