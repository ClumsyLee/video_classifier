/**
 * @file DAISY_.cpp
 * @brief mex interface for cv::xfeatures2d::DAISY
 * @ingroup xfeatures2d
 * @author Amro
 * @date 2015
 */
#include <typeinfo>
#include "mexopencv.hpp"
#include "mexopencv_features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// Persistent objects
namespace {
/// Last object id to allocate
int last_id = 0;
/// Object container
map<int,Ptr<DAISY> > obj_;
}

/**
 * Main entry called from Matlab
 * @param nlhs number of left-hand-side arguments
 * @param plhs pointers to mxArrays in the left-hand-side
 * @param nrhs number of right-hand-side arguments
 * @param prhs pointers to mxArrays in the right-hand-side
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    if (nrhs<2 || nlhs>2)
        mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");

    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);
    int id = rhs[0].toInt();
    string method(rhs[1].toString());

    // Constructor is called. Create a new object from argument
    if (method == "new") {
        if (nrhs<2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        obj_[++last_id] = createDAISY(rhs.begin() + 2, rhs.end());
        plhs[0] = MxArray(last_id);
        return;
    }

    // Big operation switch
    Ptr<DAISY> obj = obj_[id];
    if (method == "delete") {
        if (nrhs!=2 || nlhs!=0)
            mexErrMsgIdAndTxt("mexopencv:error","Output not assigned");
        obj_.erase(id);
    }
    else if (method == "typeid") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(string(typeid(*obj).name()));
    }
    else if (method == "clear") {
        if (nrhs!=2 || nlhs!=0)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        obj->clear();
    }
    else if (method == "load") {
        if (nrhs<3 || (nrhs%2)==0 || nlhs!=0)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        string objname;
        bool loadFromString = false;
        for (int i=3; i<nrhs; i+=2) {
            string key(rhs[i].toString());
            if (key=="ObjName")
                objname = rhs[i+1].toString();
            else if (key=="FromString")
                loadFromString = rhs[i+1].toBool();
            else
                mexErrMsgIdAndTxt("mexopencv:error", "Unrecognized option %s", key.c_str());
        }
        obj_[id] = (loadFromString ?
            Algorithm::loadFromString<DAISY>(rhs[2].toString(), objname) :
            Algorithm::load<DAISY>(rhs[2].toString(), objname));
    }
    else if (method == "save") {
        if (nrhs!=3 || nlhs!=0)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        obj->save(rhs[2].toString());
    }
    else if (method == "empty") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(obj->empty());
    }
    else if (method == "getDefaultName") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(obj->getDefaultName());
    }
    else if (method == "defaultNorm") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(NormTypeInv[obj->defaultNorm()]);
    }
    else if (method == "descriptorSize") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(obj->descriptorSize());
    }
    else if (method == "descriptorType") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(ClassNameInvMap[obj->descriptorType()]);
    }
    else if (method == "GetDescriptor") {
        if (nrhs<5 || (nrhs%2)==0 || nlhs>2)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        double y = rhs[2].toDouble(), x = rhs[3].toDouble();
        int orientation = rhs[4].toInt();
        bool unnormalized = false;
        bool useHomography = false;
        Matx33d H;
        for (int i=5; i<nrhs; i+=2) {
            string key(rhs[i].toString());
            if (key=="Unnormalized")
                unnormalized = rhs[i+1].toBool();
            else if (key=="H") {
                H = rhs[i+1].toMatx<double,3,3>();
                useHomography = true;
            }
            else
                mexErrMsgIdAndTxt("mexopencv:error",
                    "Unrecognized option %s", key.c_str());
        }
        vector<float> descriptor(obj->descriptorSize());
        bool ret = true;
        if (unnormalized) {
            if (useHomography)
                ret = obj->GetUnnormalizedDescriptor(y, x, orientation, &descriptor[0], H.val);
            else
                obj->GetUnnormalizedDescriptor(y, x, orientation, &descriptor[0]);
        }
        else {
            if (useHomography)
                ret = obj->GetDescriptor(y, x, orientation, &descriptor[0], H.val);
            else
                obj->GetDescriptor(y, x, orientation, &descriptor[0]);
        }
        plhs[0] = MxArray(descriptor);
        if (nlhs>1)
            plhs[1] = MxArray(ret);
    }
    else if (method == "compute") {
        if (nrhs!=4 || nlhs>2)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        if (rhs[2].isNumeric()) {  // first variant that accepts an image
            Mat image(rhs[2].toMat()), descriptors;
            vector<KeyPoint> keypoints(rhs[3].toVector<KeyPoint>());
            obj->compute(image, keypoints, descriptors);
            plhs[0] = MxArray(descriptors);
            if (nlhs > 1)
                plhs[1] = MxArray(keypoints);
        }
        else if (rhs[2].isCell()) { // second variant that accepts an image set
            vector<Mat> images(rhs[2].toVector<Mat>()), descriptors;
            vector<vector<KeyPoint> > keypoints(rhs[3].toVector(
                const_mem_fun_ref_t<vector<KeyPoint>, MxArray>(
                &MxArray::toVector<KeyPoint>)));
            obj->compute(images, keypoints, descriptors);
            plhs[0] = MxArray(descriptors);
            if (nlhs > 1)
                plhs[1] = MxArray(keypoints);
        }
        else
            mexErrMsgIdAndTxt("mexopencv:error", "Invalid arguments");
    }
    else if (method == "compute_all") {
        if ((nrhs!=3 && nrhs!=4) || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        Mat image(rhs[2].toMat()), descriptors;
        if (nrhs == 4) {
            Rect roi(rhs[3].toRect());
            obj->compute(image, roi, descriptors);
        }
        else
            obj->compute(image, descriptors);
        plhs[0] = MxArray(descriptors);
    }
    else
        mexErrMsgIdAndTxt("mexopencv:error","Unrecognized operation");
}
