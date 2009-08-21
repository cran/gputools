AC_INIT([gputools], 0.01)
AC_CONFIG_AUX_DIR([tools])

AC_ARG_WITH([cuda-home],
	AC_HELP_STRING([--with-cuda-home=cuda_home],
		[specify the root of your cuda install, e.g. /usr/local/cuda]),
	[CUDA_HOME="$withval"])

device_emu=""
AC_ARG_ENABLE([emulation],
	AC_HELP_STRING([--enable-emulation],
		[emulate the gpu because maybe you don't have one]),
	[device_emu=$enableval])

if test "${device_emu}" = yes ; then
	EMULATION="-deviceemu"
	DEVICEOPTS=""
	LIBS="-lcublasemu"
	OBJS13="kendall.o"
	SVMDISABLED="TRUE"
else
	EMULATION=""
	DEVICEOPTS="-arch=sm_13"
	LIBS="-lcublas -lcuda"
	OBJS13="kendall.o classification.o"
	SVMDISABLED="FALSE"
fi

AC_MSG_CHECKING("environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
	AC_MSG_RESULT("CUDA_HOME not set; using default /usr/local/cuda")
	CUDA_HOME="/usr/local/cuda"
else
	AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
fi

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test -z "${HAS_NVCC}"; then
	AC_MSG_ERROR("cannot find nvcc; check CUDA install")
else
	AC_MSG_RESULT("nvcc found")
	(cd R; rm fake.R)
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [BIT64="yes"])
if test -z "${BIT64}"; then
	AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
	CUDA_LIB_DIR="/lib"
else
	AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
	CUDA_LIB_DIR="/lib64"
fi

RPATHFLAG=""
AC_CANONICAL_HOST
case "${host_os}" in
	*darwin*)
		RPATHFLAG="-Wl,-rpath,${CUDA_HOME}/lib"
	;;
esac

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
	AC_MSG_ERROR("could not determine R_HOME")
else
	CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
	R_INCLUDE=`"${R_HOME}/bin/R" CMD config --cppflags`
	R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
fi

AC_SUBST(CUDA_HOME)
AC_SUBST(CUDA_LIB_DIR)
AC_SUBST(CPICFLAGS)
AC_SUBST(EMULATION)
AC_SUBST(DEVICEOPTS)
AC_SUBST(R_INCLUDE)
AC_SUBST(R_LIB)
AC_SUBST(LIBS)
AC_SUBST(OBJS13)
AC_SUBST(SVMDISABLED)
AC_SUBST([RPATHFLAG])
AC_CONFIG_FILES([src/Makefile])
AC_CONFIG_FILES([R/gpuSvm.R])
AC_OUTPUT
