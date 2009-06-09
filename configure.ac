AC_INIT([gputools], 0.01)

AC_ARG_WITH([cuda-home],
	AC_HELP_STRING([--with-cuda-home=cuda_home],
		[specify the root of your cuda install, e.g. /usr/local/cuda]),
	[CUDA_HOME="$withval"])

HAS_CUDA=FALSE
if test -z "${CUDA_HOME}"; then
	AC_MSG_WARN("could not determine CUDA_HOME")
else
	HAS_CUDA=TRUE
fi

AC_SUBST(CUDA_HOME)
AC_SUBST(HAS_CUDA)
AC_CONFIG_FILES([src/Makefile])
AC_CONFIG_FILES([R/gpuCor.R])
AC_CONFIG_FILES([R/gpuGranger.R])
AC_CONFIG_FILES([R/gpuHclust.R])
AC_CONFIG_FILES([R/gpuLr.R])
AC_CONFIG_FILES([R/gpuMatMult.R])
AC_CONFIG_FILES([R/gpuQr.R])
AC_CONFIG_FILES([R/gpuSvm.R])
AC_CONFIG_FILES([R/gpuMi.R])
AC_OUTPUT
