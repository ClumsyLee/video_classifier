clear all;
close all;
clc;

dataset_root = 'E:/dataset';

%% setup environment

addpath('./mexopencv/');

flag_install_mexopencv   =  true;
if (flag_install_mexopencv)
    if (isunix())
        error('Failed to install mexopencv automatically. Please install mexopencv using make.');
    else
        mexopencv.make('opencv_path', 'D:\Libraries\opencv\build');
    end
end

flag_compile_mexprocess     =   false;
if (flag_compile_mexprocess)
    if (exist('fun_process.cpp', 'file'))
        mex     -largeArrayDims -I'C:\Users\field\Desktop\opencv\build\include' -L'C:\Users\liuxj\Desktop\opencv\build\x64\vc11\lib'  -lopencv_ts300 -lopencv_world300  './fun_process.cpp' -output './fun_process';
    else
        error('Failed to find file fun_process.cpp.');
    end
end

%% input data

N_traindata  =   200;
N_testdata   =   25;
dir1 = dir([dataset_root '/1014/*.ogv']);
dir2 = dir([dataset_root '/1019/*.ogv']);
dir3 = dir([dataset_root '/1016/*.ogv']);
dir4 = dir([dataset_root '/1017/*.ogv']);
dir5 = dir([dataset_root '/1020/*.ogv']);
filenames   =   cell(N_traindata, 1);
testnames   =   cell(N_testdata, 1);
for times = 1:N_traindata/5
    filenames{times} = [dataset_root '/1014/' dir1(times).name];
    filenames{times+N_traindata/5} = [dataset_root '/1019/' dir2(times).name];
    filenames{times+N_traindata*2/5} = [dataset_root '/1016/' dir3(times).name];
    filenames{times+N_traindata*3/5} = [dataset_root '/1017/' dir4(times).name];
    filenames{times+N_traindata*4/5} = [dataset_root '/1020/' dir5(times).name];
end
for times = 1:5
    testnames{times} = [dataset_root '/1014/' dir1(40+times).name];
    testnames{times+5} = [dataset_root '/1019/' dir2(40+times).name];
    testnames{times+10} = [dataset_root '/1016/' dir3(40+times).name];
    testnames{times+15} = [dataset_root '/1017/' dir4(40+times).name];
    testnames{times+20} = [dataset_root '/1020/' dir5(40+times).name];
end
HLLLL  =   zeros(N_traindata, 1);
%{
trainingData = zeros(N_traindata,38);
%}
testData = zeros(N_testdata,38);
resultData = zeros(N_testdata,10);

load('feature.mat');
%% process results

groups_rst  =   zeros(N_traindata, 1);
time_used   =   zeros(N_traindata, 1);

prev_rst    =   -1;
global  img_dir;

for i_testdata = 40 : N_traindata
    disp(['training -- i_testdata' int2str(i_testdata)]);
    filename    =   filenames{i_testdata};
    group_std   =   groups_std(i_testdata);

    % load sample video data and full audio data
    [dat_vid, ~]    =   mmread(filename, [1 : 10], [], false, true);
    [~, dat_aud]    =   mmread(filename, [], [], true, false);

    % convert video to images
    img_dir     =   [filename, '_img/'];
    %{
    if (~exist(img_dir, 'dir'))
        mkdir(img_dir);
        mmread(filename, [1:min(5000,dat_vid.nrFramesTotal)], [], false, false, 'saveFrame');
    end
    %}
    dat_vid.nrFramesTotal   =   numel(dir([img_dir, '*.jpg']));
    dat_vid.filename        =   filename;

    % call function process
    tic;
    [group_rst,output]   =   fun_process(dat_vid, dat_aud, img_dir, prev_rst);
    toc
    groups_rst(i_testdata)  =   group_rst;
    trainingData(i_testdata,:) = output;
    % update previous group result
    prev_rst    =   group_std;
    if mod(i_testdata, 3) == 0
        save 'feature.mat' trainingData
    end
end
save 'feature.mat' trainingData
groups_std  =   zeros(N_testdata, 1);
groups_std(1:5)   =   1014;
groups_std(6:10)   =   1019;
groups_std(11:15)   =   1016;
result_record = zeros(15,1);

label1 = [ones(1,2*N_traindata/5) -ones(1,3*N_traindata/5)]'; %1014/1019,1014yes
label2 = [ones(1,N_traindata/5) -ones(1,1*N_traindata/5) ones(1,1*N_traindata/5) -ones(1,2*N_traindata/5)]'; %1014/1016 1014yes
label3 = [ones(1,N_traindata/5) -ones(1,2*N_traindata/5) ones(1,1*N_traindata/5) -ones(1,1*N_traindata/5)]'; %1019/1016 1019yes
label4 = [ones(1,1*N_traindata/5) -ones(1,3*N_traindata/5) ones(1,1*N_traindata/5)]';
label5 = [-ones(1,1*N_traindata/5) ones(1,2*N_traindata/5) -ones(1,2*N_traindata/5)]';
label6 = [-ones(1,1*N_traindata/5) ones(1,1*N_traindata/5) -ones(1,1*N_traindata/5) ones(1,1*N_traindata/5) -ones(1,1*N_traindata/5)];
label7 = [-ones(1,1*N_traindata/5) ones(1,1*N_traindata/5) -ones(1,2*N_traindata/5) ones(1,1*N_traindata/5)];
label8 = [-ones(1,2*N_traindata/5) ones(1,2*N_traindata/5) -ones(1,1*N_traindata/5)];
label9 = [-ones(1,2*N_traindata/5) ones(1,1*N_traindata/5) -ones(1,1*N_traindata/5) ones(1,1*N_traindata/5)];
label10 = [-ones(1,3*N_traindata/5) ones(1,2*N_traindata/5)];
mysvm1 = svmtrain(double(trainingData), int32(label1), 'kernel_function', 'polynomial');
mysvm2 = svmtrain(double(trainingData), int32(label2), 'kernel_function', 'polynomial');
mysvm3 = svmtrain(double(trainingData), int32(label3), 'kernel_function', 'polynomial');
mysvm4 = svmtrain(double(trainingData), int32(label4), 'kernel_function', 'polynomial');
mysvm5 = svmtrain(double(trainingData), int32(label5), 'kernel_function', 'polynomial');
mysvm6 = svmtrain(double(trainingData), int32(label6), 'kernel_function', 'polynomial');
mysvm7 = svmtrain(double(trainingData), int32(label7), 'kernel_function', 'polynomial');
mysvm8 = svmtrain(double(trainingData), int32(label8), 'kernel_function', 'polynomial');
mysvm9 = svmtrain(double(trainingData), int32(label9), 'kernel_function', 'polynomial');
mysvm10 = svmtrain(double(trainingData), int32(label10), 'kernel_function', 'polynomial');
for i_testdata = 1 : N_testdata
    disp(['predicting -- i_testdata' int2str(i_testdata)]);
    filename    =   testnames{i_testdata};
    group_std   =   groups_std(i_testdata);

    % load sample video data and full audio data
    [dat_vid, ~]    =   mmread(filename, [1 : 10], [], false, true);
    [~, dat_aud]    =   mmread(filename, [], [], true, false);

    % convert video to images
    img_dir     =   [filename, '_img/'];
    % if (~exist(img_dir, 'dir'))
    %     mkdir(img_dir);
    %     mmread(filename, [1:min(5000,dat_vid.nrFramesTotal)], [], false, false, 'saveFrame');
    % end
    dat_vid.nrFramesTotal   =   numel(dir([img_dir, '*.jpg']));
    dat_vid.filename        =   filename;

    % call function process
    tic;
    [group_rst,output]   =   fun_process(dat_vid, dat_aud, img_dir, prev_rst);
    toc;
    groups_rst(i_testdata)  =   group_rst;

    resultData(i_testdata,1) = svmclassify(mysvm1, double(output));
    resultData(i_testdata,2) = svmclassify(mysvm2, double(output));
    resultData(i_testdata,3) = svmclassify(mysvm3, double(output));
    resultData(i_testdata,4) = svmclassify(mysvm4, double(output));
    resultData(i_testdata,5) = svmclassify(mysvm5, double(output));
    resultData(i_testdata,6) = svmclassify(mysvm6, double(output));
    resultData(i_testdata,7) = svmclassify(mysvm7, double(output));
    resultData(i_testdata,8) = svmclassify(mysvm8, double(output));
    resultData(i_testdata,9) = svmclassify(mysvm9, double(output));
    resultData(i_testdata,10) = svmclassify(mysvm10, double(output));
    
    testData(i_testdata,:) = output;
    %result_record(i_testdata) = result;
    % update previous group result
    prev_rst    =   group_std;
    if mod(i_testdata, 3) == 0
        save 'test.mat' testData resultData
    end
end
%% performance evaluation
save 'test.mat' testData resultData
%precision   =   mean(groups_std == groups_rst);
%disp(['Processing precision:     ', num2str(precision, '%f')]);
%disp(['Relative processing time: ', num2str(mean(time_used), '%f')]);

%% post process
if (flag_compile_mexprocess)
    delete('fun_process.mex*');
end
