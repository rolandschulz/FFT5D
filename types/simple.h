#ifndef _simple_h
#define _simple_h

/*! \brief Double precision accuracy */
#define GMX_DOUBLE_EPS   1.11022302E-16

/*! \brief Single precision accuracy */
#define GMX_FLOAT_EPS    5.96046448E-08

#ifdef GMX_DOUBLE
#ifndef HAVE_REAL
typedef double          real;
#define HAVE_REAL
#endif
#define GMX_MPI_REAL    MPI_DOUBLE
#define GMX_REAL_EPS    GMX_DOUBLE_EPS
#define GMX_REAL_MIN    GMX_DOUBLE_MIN
#define GMX_REAL_MAX    GMX_DOUBLE_MAX
#else
#ifndef HAVE_REAL
typedef float           real;
#define HAVE_REAL
#endif
#define GMX_MPI_REAL    MPI_FLOAT
#define GMX_REAL_EPS    GMX_FLOAT_EPS
#define GMX_REAL_MIN    GMX_FLOAT_MIN
#define GMX_REAL_MAX    GMX_FLOAT_MAX
#endif

#endif
