from ndsl.comm.mpi import MPI


if MPI.COMM_WORLD.Get_size() == 1:
    # not run as a parallel test, disable MPI tests
    MPI = None
