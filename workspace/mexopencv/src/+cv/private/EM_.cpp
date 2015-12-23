/**
 * @file EM_.cpp
 * @brief mex interface for cv::ml::EM
 * @ingroup ml
 * @author Vladimir Eremeev, Amro
 * @date 2012, 2015
 */
#include "mexopencv.hpp"
using namespace std;
using namespace cv;
using namespace cv::ml;

// Persistent objects
namespace {
/// Last object id to allocate
int last_id = 0;
/// Object container
map<int,Ptr<EM> > obj_;

// convenience macro to return results
#define assign_lhs(ll, label, prob)   \
    do {                              \
        plhs[0] = MxArray(ll);        \
        if (nlhs > 1)                 \
            plhs[1] = MxArray(label); \
        if (nlhs > 2)                 \
            plhs[2] = MxArray(prob);  \
    } while(0)

/// Option values for SampleTypes
const ConstMap<std::string, int> SampleTypesMap = ConstMap<std::string, int>
    ("Row", cv::ml::ROW_SAMPLE)   //!< each training sample is a row of samples
    ("Col", cv::ml::COL_SAMPLE);  //!< each training sample occupies a column of samples

/// Option values for TrainData VariableTypes
const ConstMap<std::string, int> VariableTypeMap = ConstMap<std::string, int>
    ("Numerical",   cv::ml::VAR_NUMERICAL)     //!< same as VAR_ORDERED
    ("Ordered",     cv::ml::VAR_ORDERED)       //!< ordered variables
    ("Categorical", cv::ml::VAR_CATEGORICAL)   //!< categorical variables
    ("N",           cv::ml::VAR_NUMERICAL)     //!< shorthand for (N)umerical
    ("O",           cv::ml::VAR_ORDERED)       //!< shorthand for (O)rdered
    ("C",           cv::ml::VAR_CATEGORICAL);  //!< shorthand for (C)ategorical

/// CovMatType map for option processing
const ConstMap<string, int> CovMatType = ConstMap<string, int>
    ("Spherical", EM::COV_MAT_SPHERICAL)
    ("Diagonal",  EM::COV_MAT_DIAGONAL)
    ("Generic",   EM::COV_MAT_GENERIC)
    ("Default",   EM::COV_MAT_DEFAULT);

/// CovMatTypeInv map for option processing
const ConstMap<int, string> CovMatTypeInv = ConstMap<int, string>
    (EM::COV_MAT_SPHERICAL, "Spherical")
    (EM::COV_MAT_DIAGONAL,  "Diagonal")
    (EM::COV_MAT_GENERIC,   "Generic")
    (EM::COV_MAT_DEFAULT,   "Default");
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
    // Check the number of arguments
    if (nrhs<2 || nlhs>3)
        mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");

    // Argument vector
    vector<MxArray> rhs(prhs,prhs+nrhs);
    int id = rhs[0].toInt();
    string method(rhs[1].toString());

    // Constructor is called. Create a new object from argument
    if (method == "new") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        obj_[++last_id] = EM::create();
        plhs[0] = MxArray(last_id);
        return;
    }

    // Big operation switch
    Ptr<EM> obj = obj_[id];
    if (method == "delete") {
        if (nrhs!=2 || nlhs!=0)
            mexErrMsgIdAndTxt("mexopencv:error","Output not assigned");
        obj_.erase(id);
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
            Algorithm::loadFromString<EM>(rhs[2].toString(), objname) :
            Algorithm::load<EM>(rhs[2].toString(), objname));
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
    else if (method == "getVarCount") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(obj->getVarCount());
    }
    else if (method == "isClassifier") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(obj->isClassifier());
    }
    else if (method == "isTrained") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(obj->isTrained());
    }
    else if (method == "train") {
        if (nrhs<4 || (nrhs%2)==1 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        int layout = cv::ml::ROW_SAMPLE;
        for (int i=4; i<nrhs; i+=2) {
            string key(rhs[i].toString());
            if (key=="Layout")
                layout = SampleTypesMap[rhs[i+1].toString()];
            else
                mexErrMsgIdAndTxt("mexopencv:error", "Unrecognized option %s", key.c_str());
        }
        Mat samples(rhs[2].toMat(CV_32F));
        Mat responses(rhs[3].toMat(rhs[3].isInt32() ? CV_32S : CV_32F));
        bool b = obj->train(samples, layout, responses);
        plhs[0] = MxArray(b);
    }
    else if (method == "train_") {
        if (nrhs<4 || (nrhs%2)==1 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        int flags = 0;
        int layout = cv::ml::ROW_SAMPLE;
        Mat varIdx, sampleIdx, sampleWeights, varType;
        for (int i=4; i<nrhs; i+=2) {
            string key(rhs[i].toString());
            if (key=="Flags")
                flags = rhs[i+1].toInt();
            else if (key=="UpdateModel")
                UPDATE_FLAG(flags, rhs[i+1].toBool(), StatModel::UPDATE_MODEL);
            else if (key=="RawOuput")
                UPDATE_FLAG(flags, rhs[i+1].toBool(), StatModel::RAW_OUTPUT);
            else if (key=="CompressedInput")
                UPDATE_FLAG(flags, rhs[i+1].toBool(), StatModel::COMPRESSED_INPUT);
            else if (key=="PreprocessedInput")
                UPDATE_FLAG(flags, rhs[i+1].toBool(), StatModel::PREPROCESSED_INPUT);
            else if (key=="Layout")
                layout = SampleTypesMap[rhs[i+1].toString()];
            else if (key=="VarIdx")
                varIdx = rhs[i+1].toMat((rhs[i+1].isUint8() || rhs[i+1].isLogical()) ? CV_8U : CV_32S);
            else if (key=="SampleIdx")
                sampleIdx = rhs[i+1].toMat((rhs[i+1].isUint8() || rhs[i+1].isLogical()) ? CV_8U : CV_32S);
            else if (key=="SampleWeights")
                sampleWeights = rhs[i+1].toMat(CV_32F);
            else if (key=="VarType") {
                if (rhs[i+1].isCell()) {
                    vector<string> vtypes(rhs[i+1].toVector<string>());
                    varType.create(1, vtypes.size(), CV_8U);
                    for (size_t idx = 0; idx < vtypes.size(); idx++)
                        varType.at<uchar>(idx) = VariableTypeMap[vtypes[idx]];
                }
                else if (rhs[i+1].isNumeric())
                    varType = rhs[i+1].toMat(CV_8U);
                else
                    mexErrMsgIdAndTxt("mexopencv:error", "Invalid VarType value");
            }
            else
                mexErrMsgIdAndTxt("mexopencv:error", "Unrecognized option %s", key.c_str());
        }
        Mat samples(rhs[2].toMat(CV_32F));
        Mat responses(rhs[3].toMat(rhs[3].isInt32() ? CV_32S : CV_32F));
        Ptr<TrainData> trainData = TrainData::create(samples, layout, responses,
            varIdx, sampleIdx, sampleWeights, varType);
        bool b = obj->train(trainData, flags);
        plhs[0] = MxArray(b);
    }
    else if (method == "calcError") {
        if (nrhs<5 || (nrhs%2)==0 || nlhs>2)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        int layout = cv::ml::ROW_SAMPLE;
        Mat varIdx, sampleIdx, sampleWeights, varType;
        for (int i=5; i<nrhs; i+=2) {
            string key(rhs[i].toString());
            if (key=="Layout")
                layout = SampleTypesMap[rhs[i+1].toString()];
            else if (key=="VarIdx")
                varIdx = rhs[i+1].toMat((rhs[i+1].isUint8() || rhs[i+1].isLogical()) ? CV_8U : CV_32S);
            else if (key=="SampleIdx")
                sampleIdx = rhs[i+1].toMat((rhs[i+1].isUint8() || rhs[i+1].isLogical()) ? CV_8U : CV_32S);
            else if (key=="SampleWeights")
                sampleWeights = rhs[i+1].toMat(CV_32F);
            else if (key=="VarType") {
                if (rhs[i+1].isCell()) {
                    vector<string> vtypes(rhs[i+1].toVector<string>());
                    varType.create(1, vtypes.size(), CV_8U);
                    for (size_t idx = 0; idx < vtypes.size(); idx++)
                        varType.at<uchar>(idx) = VariableTypeMap[vtypes[idx]];
                }
                else if (rhs[i+1].isNumeric())
                    varType = rhs[i+1].toMat(CV_8U);
                else
                    mexErrMsgIdAndTxt("mexopencv:error", "Invalid VarType value");
            }
            else
                mexErrMsgIdAndTxt("mexopencv:error", "Unrecognized option %s", key.c_str());
        }
        Mat samples(rhs[2].toMat(CV_32F));
        Mat responses(rhs[3].toMat(rhs[3].isInt32() ? CV_32S : CV_32F));
        bool test = rhs[4].toBool();
        Ptr<TrainData> data = TrainData::create(samples, layout, responses,
            varIdx, sampleIdx, sampleWeights, varType);
        Mat resp;
        float err = obj->calcError(data, test, resp);
        plhs[0] = MxArray(err);
        if (nlhs>1)
            plhs[1] = MxArray(resp);
    }
    else if (method == "predict") {
        if (nrhs<3 || (nrhs%2)==0 || nlhs>2)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        int flags = 0;
        for (int i=3; i<nrhs; i+=2) {
            string key(rhs[i].toString());
            if (key=="Flags")
                flags = rhs[i+1].toInt();
            else if (key=="UpdateModel")
                UPDATE_FLAG(flags, rhs[i+1].toBool(), StatModel::UPDATE_MODEL);
            else if (key=="RawOuput")
                UPDATE_FLAG(flags, rhs[i+1].toBool(), StatModel::RAW_OUTPUT);
            else if (key=="CompressedInput")
                UPDATE_FLAG(flags, rhs[i+1].toBool(), StatModel::COMPRESSED_INPUT);
            else if (key=="PreprocessedInput")
                UPDATE_FLAG(flags, rhs[i+1].toBool(), StatModel::PREPROCESSED_INPUT);
            else
                mexErrMsgIdAndTxt("mexopencv:error", "Unrecognized option %s", key.c_str());
        }
        Mat samples(rhs[2].toMat(CV_32F)), results;
        float f = obj->predict(samples, results, flags);
        plhs[0] = MxArray(results);
        if (nlhs>1)
            plhs[1] = MxArray(f);
    }
    else if (method == "trainE") {
        if (nrhs < 4 || (nrhs % 2) != 0 || nlhs>3)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        vector<Mat> covs0;
        Mat weights0;
        for(int i = 4; i < nrhs; i += 2) {
            string key = rhs[i].toString();
            if (key == "Covs0")
                covs0 = rhs[i+1].toVector<Mat>();
            else if (key == "Weights0")
                weights0 = rhs[i+1].toMat();
            else
                mexErrMsgIdAndTxt("mexopencv:error", "Unrecognized option %s", key.c_str());
        }
        Mat samples(rhs[2].toMat());
        Mat means0(rhs[3].toMat());  // rows=obj->getClustersNumber(), cols=samples.cols
        Mat logLikelihoods, labels, probs;
        bool b = obj->trainE(samples, means0, covs0, weights0, logLikelihoods, labels, probs);
        assign_lhs(logLikelihoods, labels, probs);
        if (nlhs>3) plhs[3] = MxArray(b);
    }
    else if (method == "trainM") {
        if (nrhs!=4 || nlhs>3)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        Mat samples = rhs[2].toMat();
        Mat probs0  = rhs[3].toMat(); // rows=obj->getClustersNumber(), cols=samples.rows
        Mat logLikelihoods, labels, probs;
        bool b = obj->trainM(samples, probs0, logLikelihoods, labels, probs);
        assign_lhs(logLikelihoods, labels, probs);
        if (nlhs>3) plhs[3] = MxArray(b);
    }
    else if (method == "trainEM") {
        if (nrhs!=3 || nlhs>3)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        Mat samples = rhs[2].toMat();
        Mat logLikelihoods, labels, probs;
        bool b = obj->trainEM(samples, logLikelihoods, labels, probs);
        assign_lhs(logLikelihoods, labels, probs);
        if (nlhs>3) plhs[3] = MxArray(b);
    }
    else if (method == "predict2") {
        if (nrhs!=3 || nlhs>3)
            mexErrMsgIdAndTxt("mexopencv:error","Wrong number of arguments");
        Mat sample(rhs[2].toMat()), probs;
        Vec2d result = obj->predict2(sample, probs);
        assign_lhs(result[0], result[1], probs);
    }
    else if (method == "getCovs") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        vector<Mat> covs;
        obj->getCovs(covs);
        plhs[0] = MxArray(covs);
    }
    else if (method == "getMeans") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(obj->getMeans());
    }
    else if (method == "getWeights") {
        if (nrhs!=2 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        plhs[0] = MxArray(obj->getWeights());
    }
    else if (method == "get") {
        if (nrhs!=3 || nlhs>1)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        string prop(rhs[2].toString());
        if (prop == "ClustersNumber")
            plhs[0] = MxArray(obj->getClustersNumber());
        else if (prop == "CovarianceMatrixType")
            plhs[0] = MxArray(CovMatTypeInv[obj->getCovarianceMatrixType()]);
        else if (prop == "TermCriteria")
            plhs[0] = MxArray(obj->getTermCriteria());
        else
            mexErrMsgIdAndTxt("mexopencv:error", "Unrecognized property %s", prop.c_str());
    }
    else if (method == "set") {
        if (nrhs!=4 || nlhs!=0)
            mexErrMsgIdAndTxt("mexopencv:error", "Wrong number of arguments");
        string prop(rhs[2].toString());
        if (prop == "ClustersNumber")
            obj->setClustersNumber(rhs[3].toInt());
        else if (prop == "CovarianceMatrixType")
            obj->setCovarianceMatrixType(CovMatType[rhs[3].toString()]);
        else if (prop == "TermCriteria")
            obj->setTermCriteria(rhs[3].toTermCriteria());
        else
            mexErrMsgIdAndTxt("mexopencv:error", "Unrecognized property %s", prop.c_str());
    }
    else
        mexErrMsgIdAndTxt("mexopencv:error","Unrecognized operation");
}
