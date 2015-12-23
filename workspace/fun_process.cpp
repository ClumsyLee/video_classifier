#include    <mex.h>
#include    <stdint.h>
#include    <math.h>
#include    <stdio.h>
#include    <opencv2/opencv.hpp>

const int32_t   MAX_FILENAME_LENGTH =   1024;

void    mexFunction(int32_t nou, mxArray * vou[], int32_t nin, const mxArray * vin[])
{
    double      dat_vid_rate    =   0.0;
    double      dat_vid_nrFramesTotal   =   0.0;
    double      dat_aud_nrChannels  =   0.0;
    double      dat_aud_rate    =   0.0;
    double *    dat_aud_data    =   NULL;
    int32_t     dat_aud_dataM   =   0;
    int32_t     dat_aud_dataN   =   0;
    char        img_dir[MAX_FILENAME_LENGTH]    =   "";
    int32_t     img_dir_len     =   0;
    char        filename[MAX_FILENAME_LENGTH]   =   "";
    
    double      group_rst       =   1006;
    
    // check input arguments
    switch (nin)
    {
        case 4:
            dat_vid_rate        =   *mxGetPr(mxGetField(vin[0], 0, "rate"));
            dat_vid_nrFramesTotal   =   *mxGetPr(mxGetField(vin[0], 0, "nrFramesTotal"));
            dat_aud_nrChannels  =   *mxGetPr(mxGetField(vin[1], 0, "nrChannels"));
            dat_aud_rate        =   *mxGetPr(mxGetField(vin[1], 0, "rate"));
            dat_aud_data        =   (double *)mxGetData(mxGetField(vin[1], 0, "data"));
            dat_aud_dataM       =   (int32_t)mxGetM(mxGetField(vin[1], 0, "data"));
            dat_aud_dataN       =   (int32_t)mxGetN(mxGetField(vin[1], 0, "data"));
            img_dir_len         =   (int32_t)mxGetNumberOfElements(vin[2]);
            if (0 != mxGetString(vin[2], img_dir, img_dir_len + 1))
            {
                mexErrMsgTxt("Failed to get img_dir.");
            }
            break;
        default:
            mexErrMsgTxt("Illegal input argument amount.");
            break;
    }
    
    // process
    
    // display values
    printf("dat_vid:\nRate\t:%f\nnrFramesTotal\t:%f\n", dat_vid_rate, dat_vid_nrFramesTotal);
    printf("dat_aud:\nnrChannels\t:%f\nRate\t:%f\nData size\t:%d x %d\n", dat_aud_nrChannels, dat_aud_rate, dat_aud_dataM, dat_aud_dataN);
    printf("img_dir:\n%s\n", img_dir);
    
    // display frames, only one frame of a second is displayed
    cv::Mat     img;
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    for (int i = 1; i <= dat_vid_nrFramesTotal; i += int(dat_vid_rate))
    {
        sprintf(filename, "%s%06d.jpg\0", img_dir, i);
        img     =   cv::imread(filename, CV_LOAD_IMAGE_COLOR);
        if (!img.data)
        {
            mexErrMsgTxt("Could not open image file.");
        }
        cv::imshow("Display window", img);
        cv::waitKey(30);
    }
    
    // display audio data sample
    cv::Mat     fig(600, 800, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::namedWindow("Audio window (Press any key to continue)", cv::WINDOW_AUTOSIZE);
    for (int i = 0; i < dat_aud_dataN; i++)
    {
        cv::Scalar  color(0, 0, 0);
        color.val[i]    =   255;
        for (int j = 1; j < dat_aud_dataM; j++)
        {
            cv::line(fig, cv::Point(double(j - 1) / dat_aud_dataM * 800, dat_aud_data[j - 1 + i * dat_aud_dataM] * 300 + 300), 
                    cv::Point(double(j) / dat_aud_dataM * 800, dat_aud_data[j + i * dat_aud_dataM] * 300 + 300), 
                    color);
        }
    }
    cv::imshow("Audio window (Press any key to continue)", fig);
    
    // wait for continue
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    // check output arguments
    switch (nou)
    {
        case 1:
            vou[0]              =   mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
            *mxGetPr(vou[0])    =   group_rst;
            break;
        default:
            mexErrMsgTxt("Illegal output argument amount.");
            break;
    }
    
    return;
}