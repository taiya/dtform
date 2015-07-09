// mex CXXFLAGS='$CXXFLAGS -std=c++11' dtform.cpp;
#include "mex.h"
#include <limits>
#include <cmath>

#define ENABLE_DTFORM_DSTS
typedef double Scalar; //< MATLAB likes doubles

class DistanceTransform{
private:
    int width=0;
    int height=0;
    Scalar* v=NULL;
    Scalar* z=NULL;
    Scalar* DTTps=NULL;
    int* ADTTps=NULL;
    Scalar* realDT=NULL;
    int* realADT=NULL; ///< stores closest point ids
    
public:
    void init(int width, int height){
        this->width = width;
        this->height = height;
        v = new Scalar[width*height];
        z = new Scalar[width*height];
        DTTps = new Scalar[width*height];
        ADTTps = new int[width*height];
        realADT = new int[width*height];
        realDT = new Scalar[width*height];
    }
    void cleanup(){
        delete[] v;
        delete[] z;
        delete[] DTTps;
        delete[] ADTTps;
        delete[] realADT;
        delete[] realDT;        
    }
    
    Scalar* dsts_image_ptr(){ return realDT; }
    int* idxs_image_ptr(){ return realADT; }
    int dst_at(int row, int col){ return realDT[row*width+col]; }
    int idx_at(int row, int col){ return realADT[row*width+col]; }
    
public:
    /// @pars row major uchar binary image. White pixels are "data" and 
    /// label_image[i] > mask_th decides what data is.
    template<class Scalar>
    void exec(Scalar* label_image, Scalar mask_th)
    {
        // #pragma omp parallel
        {
            // #pragma omp for
            for(int i = 0; i < width*height; ++i)
            {
                if(label_image[i]<mask_th)
                    realDT[i] = std::numeric_limits<Scalar>::max();
                else
                    realDT[i] = Scalar(0);
            }

            //return;
            
            /////////////////////////////////////////////////////////////////
            /// DT and ADT
            /////////////////////////////////////////////////////////////////

            //First PASS (rows)
            // #pragma omp for
            for(int row = 0; row<height; ++row)
            {
                unsigned int k = 0;
                unsigned int indexpt1 = row*width;
                v[indexpt1] = 0;
                z[indexpt1] = std::numeric_limits<Scalar>::min();
                z[indexpt1 + 1] = std::numeric_limits<Scalar>::max();
                for(int q = 1; q<width; ++q)
                {
                    Scalar sp1 = Scalar(realDT[(indexpt1 + q)] + (q*q));
                    unsigned int index2 = indexpt1 + k;
                    unsigned int vk = v[index2];
                    Scalar s = (sp1 - Scalar(realDT[(indexpt1 + vk)] + (vk*vk)))/Scalar((q-vk) << 1);
                    while(s <= z[index2] && k > 0)
                    {
                        k--;
                        index2 = indexpt1 + k;
                        vk = v[index2];
                        s = (sp1 - Scalar(realDT[(indexpt1 + vk)] + (vk*vk)))/Scalar((q-vk) << 1);
                    }
                    k++;
                    index2 = indexpt1 + k;
                    v[index2] = q;
                    z[index2] = s;
                    z[index2+1] = std::numeric_limits<Scalar>::max();
                }
                k = 0;
                for(int q = 0; q<width; ++q)
                {
                    while(z[indexpt1 + k+1]<q)
                        k++;
                    unsigned int index2 = indexpt1 + k;
                    unsigned int vk = v[index2];
                    Scalar tp1 =  Scalar(q) - Scalar(vk);
                    DTTps[indexpt1 + q] = tp1*tp1 + Scalar(realDT[(indexpt1 + vk)]);
                    ADTTps[indexpt1 + q] = indexpt1 + vk;
                }
            }

            //--- Second PASS (columns)
            // #pragma omp for
            for(int col = 0; col<width; ++col)
            {
                unsigned int k = 0;
                unsigned int indexpt1 = col*height;
                v[indexpt1] = 0;
                z[indexpt1] = std::numeric_limits<Scalar>::min();
                z[indexpt1 + 1] = std::numeric_limits<Scalar>::max();
                for(int row = 1; row<height; ++row)
                {
                    Scalar sp1 = Scalar(DTTps[col + row*width] + (row*row));
                    unsigned int index2 = indexpt1 + k;
                    unsigned int vk = v[index2];
                    Scalar s = (sp1 - Scalar(DTTps[col + vk*width] + (vk*vk)))/Scalar((row-vk) << 1);
                    while(s <= z[index2] && k > 0)
                    {
                        k--;
                        index2 = indexpt1 + k;
                        vk = v[index2];
                        s = (sp1 - Scalar(DTTps[col + vk*width] + (vk*vk)))/Scalar((row-vk) << 1);
                    }
                    k++;
                    index2 = indexpt1 + k;
                    v[index2] = row;
                    z[index2] = s;
                    z[index2+1] = std::numeric_limits<Scalar>::max();
                }
                k = 0;
                for(int row = 0; row<height; ++row)
                {
                    while(z[indexpt1 + k+1]<row)
                        k++;
                    unsigned int index2 = indexpt1 + k;
                    unsigned int vk = v[index2];
                    #ifdef ENABLE_DTFORM_DSTS
                        /// Also compute the distance value
                        Scalar tp1 =  Scalar(row) - Scalar(vk);
                        realDT[col + row*width] = std::sqrt(tp1*tp1 + DTTps[col + vk*width]);
                    #endif
                    realADT[col + row*width] = ADTTps[col + vk*width];
                }
            }
        } ///< OPENMP
    }
};

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]){
    mexPrintf("welcome to dtform!\n");
	if( nrhs!=1 )
		mexErrMsgTxt("This function requires 1 arguments\n");    
	if( !mxIsNumeric(prhs[0]) )
		mexErrMsgTxt("varargin{0} must be a numeric image\n");
    if( !(mxGetNumberOfDimensions(prhs[0])==2) )
		mexErrMsgTxt("varargin{0} must be an image\n");
    int num_rows = mxGetM(prhs[0]);
    int num_cols = mxGetN(prhs[0]);
    if( !mxIsDouble(prhs[0]) )
		mexErrMsgTxt("varargin{0} must be a _double_ image\n");
        
    mexPrintf("Processing size(I)=[%dx%d]\n", num_rows, num_cols);
    Scalar* image = (Scalar*) mxGetPr(prhs[0]);
	    
    DistanceTransform dt;
    dt.init(num_rows, num_cols);
    dt.exec(image, Scalar(.5));
    
    ///--- allocate & copy output (could be improved
    {
        Scalar* out_dt = dt.dsts_image_ptr();
        plhs[0] = mxCreateDoubleMatrix(num_rows, num_cols, mxREAL);
        Scalar* out = (Scalar*) mxGetPr(plhs[0]);
        for(int i=0; i<(num_rows*num_cols); i++)
            out[i] = out_dt[i];
    }
    
    ///--- Also save corresp (indexes)
    if(nlhs==2){        
        plhs[1] = mxCreateDoubleMatrix(num_rows, num_cols, mxREAL);
        Scalar* out = (Scalar*) mxGetPr(plhs[1]);
        int* out_dti = dt.idxs_image_ptr();
        for(int i=0; i<(num_rows*num_cols); i++)
            out[i] = out_dti[i];
    }
    
    dt.cleanup();    
}