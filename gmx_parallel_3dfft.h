/* -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; c-file-style: "stroustrup"; -*-
 *
 * $Id: gmx_parallel_3dfft.h,v 1.1 2009/06/04 11:10:00 rschulz Exp $
 *
 * Gromacs                               Copyright (c) 1991-2005
 * David van der Spoel, Erik Lindahl, University of Groningen.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org
 * 
 * And Hey:
 * Gnomes, ROck Monsters And Chili Sauce
 */

#ifndef _gmx_parallel_3dfft_h_
#define _gmx_parallel_3dfft_h_

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef GMX_MPI

#include "types/simple.h"
#include "gmxcomplex.h"
#include "gmx_fft.h"

/* We NEED MPI here. */
#include <mpi.h>

#include "fft5d.h"

typedef struct gmx_parallel_3dfft  { 
    fft5d_plan p1,p2;    
} * gmx_parallel_3dfft_t;



/*! \brief Initialize parallel MPI-based 3D-FFT.
 *
 *  This routine performs real-to-complex and complex-to-real parallel 3D FFTs,
 *  but not complex-to-complex.
 *
 *  The routine is optimized for small-to-medium size FFTs used for PME and
 *  PPPM algorithms, and do allocate extra workspace whenever it might improve
 *  performance. 
 *
 *  \param pfft_setup     Pointer to parallel 3dfft setup structure, previously
 *                        allocated or with automatic storage.
 *  \param ngrid          Global number of grid cells in each dimension.
 *  \param comm           Major and minor dimension communicator for our node.
 *  \param slab2index     Dimension nnodes+1 in each dimension, where nnodes is
 *                        the size of major/minor communicators. For each node index,
 *                        this array contains the first logical grid cell index 
 *                        belonging to that node.
 *  \param bReproducible  Try to avoid FFT timing optimizations and other stuff
 *                        that could make results differ for two runs with
 *                        identical input (reproducibility for debugging).
 *    
 *  \return 0 or a standard error code.
 */
int
gmx_parallel_3dfft_init   (gmx_parallel_3dfft_t *    pfft_setup,
                           ivec                      ndata,
						   real **                   real_data,
						   t_complex **              complex_data,
                           MPI_Comm                  comm[2],
						   int                      *slab2index[2],
                           bool                      bReproducible);
                           




/*! \brief Get direct space grid index limits
 */
int
gmx_parallel_3dfft_real_limits(gmx_parallel_3dfft_t      pfft_setup,
							   ivec                      local_ndata,
							   ivec                      local_offset,
							   ivec                      local_size);


/*! \brief Get reciprocal space grid index limits
 */
int
gmx_parallel_3dfft_complex_limits(gmx_parallel_3dfft_t      pfft_setup,
								  ivec                      local_ndata,
								  ivec                      local_offset,
								  ivec                      local_size);


int
gmx_parallel_3dfft_execute(gmx_parallel_3dfft_t    pfft_setup,
						   enum gmx_fft_direction  dir,
						   void *                  in_data,
						   void *                  out_data);



/*! \brief Release all data in parallel fft setup
 *
 *  All temporary storage and FFT plans are released. The structure itself
 *  is not released, but the contents is invalid after this call.
 *
 *  \param pfft_setup Parallel 3dfft setup.
 *
 *  \return 0 or a standard error code.
 */
int
gmx_parallel_3dfft_destroy(gmx_parallel_3dfft_t    pfft_setup);

#endif /* GMX_MPI */

#endif /* _gmx_parallel_3dfft_h_ */

