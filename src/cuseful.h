int fatal(const char * msg);
void * xmalloc(size_t nbytes);
float * fMalloc(size_t numElts);
double * dMalloc(size_t numElts);
size_t * stMalloc(size_t n);
char * getTime();
void getRandVect(float * vect, size_t n);
void printVect(int n, const float * vect, const char * msg);
void printMat(int rows, int cols, const float * mat, const char * msg);
void checkCudaError(const char * msg);
float * getMatFromFile(int rows, int cols, const char * fn);
void checkCublasError(const char * msg);
