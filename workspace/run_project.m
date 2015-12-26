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
        mexopencv.make('opencv_path', 'F:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\include\opencv\build');
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

N_traindata  =   75;
N_testdata   =   25;
dir1 = dir([dataset_root '/1014/*.ogv']);
dir2 = dir([dataset_root '/1019/*.ogv']);
dir3 = dir([dataset_root '/1016/*.ogv']);
dir4 = dir([dataset_root '/1017/*.ogv']);
dir5 = dir([dataset_root '/1020/*.ogv']);
filenames   =   cell(N_traindata, 1);
testnames   =   cell(N_testdata, 1);
for times = 1:15
    filenames{times} = [dataset_root '/1014/' dir1(2+times).name];
    filenames{times+15} = [dataset_root '/1019/' dir2(27+times).name];
    filenames{times+30} = [dataset_root '/1016/' dir3(4+times).name];
    filenames{times+45} = [dataset_root '/1017/' dir4(times).name];
    filenames{times+60} = [dataset_root '/1020/' dir5(times).name];
end
for times = 1:5
    testnames{times} = [dataset_root '/1014/' dir1(20+times).name];
    testnames{times+5} = [dataset_root '/1019/' dir2(40+times).name];
    testnames{times+10} = [dataset_root '/1016/' dir3(20+times).name];
    testnames{times+15} = [dataset_root '/1017/' dir4(40+times).name];
    testnames{times+20} = [dataset_root '/1020/' dir5(15+times).name];
end

groups_std  =   zeros(N_traindata, 1);
groups_std(1:15)   =   1014;
groups_std(16:30)   =   1019;
groups_std(31:45)   =   1016;
groups_std(46:60)   =   1017;
groups_std(61:75)   =   1020;

trainingData = zeros(75,38);
testData = zeros(25,38);
resultData = zeros(25,10);

load('feature.mat');
load('test.mat')
%% process results

groups_rst  =   zeros(N_traindata, 1);
time_used   =   zeros(N_traindata, 1);

prev_rst    =   -1;
global  img_dir;

for i_testdata = 1 : 0 %N_traindata
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

label1 = [ones(1,30) -ones(1,45)]'; %1014/1019,1014yes
label2 = [ones(1,15) -ones(1,15) ones(1,15) -ones(1,30)]'; %1014/1016 1014yes
label3 = [ones(1,15) -ones(1,30) ones(1,15) -ones(1,15)]'; %1019/1016 1019yes
label4 = [ones(1,15) -ones(1,45) ones(1,15)]';
label5 = [-ones(1,15) ones(1,30) -ones(1,30)]';
label6 = [-ones(1,15) ones(1,15) -ones(1,15) ones(1,15) -ones(1,15)];
label7 = [-ones(1,15) ones(1,15) -ones(1,30) ones(1,15)];
label8 = [-ones(1,30) ones(1,30) -ones(1,15)];
label9 = [-ones(1,30) ones(1,15) -ones(1,15) ones(1,15)];
label10 = [-ones(1,45) ones(1,30)];
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
for i_testdata = 13 : N_testdata
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
    time_used(i_testdata)   =   toc / dat_vid.totalDuration;
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

precision   =   mean(groups_std == groups_rst);
disp(['Processing precision:     ', num2str(precision, '%f')]);
disp(['Relative processing time: ', num2str(mean(time_used), '%f')]);

%% post process
if (flag_compile_mexprocess)
    delete('fun_process.mex*');
end
