% process a video and return its genre using audio-visual information
%
% INPUT
%   dat_vid Video structure returned by mmread, additional field filename added for direct video file location
%   dat_aud Audio structure returned by mmread
%   img_dir Directory of images converted from the input video in time order
%   prev_rst    Standard genre result of the last call of this function
%
% OUTPUT
%   group_rst   A single integer indicating the group id.
%

%% edit your own code in this file, leave the function interface unmodified

function    [group_rst, output]   =   fun_process(dat_vid, dat_aud, img_dir, prev_rst)
    totalFrames = round(dat_vid.rate * dat_vid.totalDuration);
    totalLength = length(1:5:totalFrames);
    gap = 2001;
    onceFrames = min(gap, totalFrames);
    %camera and move feature
    feature = zeros(5,1); 
    [framesPool,~] = mmread(dat_vid.filename, [1:5:onceFrames], [], false, true);
    beginFrame = 1;
    bia = 0;
    lastFrame = length(framesPool.frames);
    readbegin = onceFrames;
    % set group result
    group_rst   =   1006;
    
    % use mexopencv to process image
    %% camera feature

    gray_delta = zeros(totalLength-1,1);
    v_var = zeros(totalLength, 1);
    v_mean_total = zeros(totalLength, 1);
    rgb_diff = zeros(totalLength-1,1);
    last_rgb = framesPool.frames(1).cdata;
    %last_rgb = imread([img_dir, num2str(1, '%06d.jpg')]);
    last_img = rgb2gray(last_rgb);
    hsv = reshape(cv.cvtColor(last_rgb, 'RGB2HSV'),[dat_vid.height*dat_vid.width 3]);
    last_rgb = reshape(last_rgb, [dat_vid.height*dat_vid.width 3]);
    v_mean_total(1) = mean(hsv(:,3));
    v_var(1) = var(double(hsv(:,3)));
    while true
        for turn = beginFrame:lastFrame
            if turn == 1
                continue
            end
            %img = imread([img_dir, num2str(turn, '%06d.jpg')]);
            img = framesPool.frames(turn - bia).cdata;
            hsv = reshape(cv.cvtColor(img, 'RGB2HSV'),[dat_vid.height*dat_vid.width 3]);
            gray_img = rgb2gray(img);
            img = reshape(img, [dat_vid.height*dat_vid.width 3]);
            v_mean_total(turn) = mean(hsv(:,3));
            v_var(turn) = var(double(hsv(:,3)));
            rgb_diff(turn-1) = sqrt(sum(sum((img - last_rgb).^2)))/(dat_vid.height*dat_vid.width);
            last_rgb = img;
            gray_delta(turn-1) = sum(sum(abs(gray_img-last_img)));
            last_img = gray_img;
        end
        if totalFrames - readbegin <= 5
            break;
        end
        nextFrames = min(gap, totalFrames - readbegin - 4);
        [framesPool,~] = mmread(dat_vid.filename, [readbegin+5:5:readbegin+4+nextFrames], [], false, true);
        bia = lastFrame;
        beginFrame = lastFrame + 1;
        lastFrame = lastFrame + length(framesPool.frames);
        readbegin = readbegin+4+nextFrames;
    end
    over_th = (gray_delta > 2*mean(gray_delta));
    over_th_last = 0;
    cut_transform = 0;
    gradual_transform = 0;
    for turn = length(over_th):-1:2
        if over_th(turn-1) && over_th(turn)
            over_th(turn) = 0;
            over_th_last = over_th_last + 1;
        elseif ~over_th(turn)
            if over_th_last == 1
                cut_transform = cut_transform + 1;
            elseif over_th_last > 1
                gradual_transform = gradual_transform + 1;
            end
            over_th_last = 0;
        end
    end
    feature(1) = sum(over_th)/dat_vid.totalDuration;
    if cut_transform+gradual_transform > 0
        feature(2) = cut_transform/(cut_transform+gradual_transform);
    else
        feature(2) = 0;
    end
    transform_point = [1 find(over_th == 1)' totalLength];
    if length(transform_point) >= 4
        key_frame = [floor((transform_point(2) + transform_point(1))/2), ...
            floor((transform_point(floor(length(transform_point)/2)+1) + transform_point(floor(length(transform_point)/2)))/2), ...
            floor((transform_point(end) + transform_point(end-1))/2)];
    elseif length(transform_point) == 3
        key_frame = [floor((transform_point(2) + transform_point(1))/2), ...
            floor((transform_point(3) + transform_point(1))/2), ...
            floor((transform_point(3) + transform_point(2))/2)];
    else
        key_frame = [floor(transform_point(2)/4),floor(transform_point(2)/2),floor(transform_point(2)*3/4)];
    end
    %% color and material feature
    [framesPool,~] = mmread(dat_vid.filename, key_frame, [], false, true);
    % static feature
    % =========================================
    color_hist_max = zeros(length(key_frame),1);
    color_hist_var = zeros(length(key_frame),1);
    v_mean = zeros(length(key_frame),1);
    s_mean = zeros(length(key_frame),1);
    v_over = zeros(length(key_frame),1);
    s_over = zeros(length(key_frame),1);
    contrast = zeros(length(key_frame),1);
    similarity = zeros(length(key_frame),1);
    energy = zeros(length(key_frame),1);
    entropy = zeros(length(key_frame),1);
    correlation = zeros(length(key_frame),1);
    % =========================================
    for index = 1:length(key_frame);
        %img = imread([img_dir, num2str(frame, '%06d.jpg')]);
        img = framesPool.frames(index).cdata;
        hsv = cv.cvtColor(img, 'RGB2HSV');
        f_hsv = hsv(:,:,1)*16+hsv(:,:,2)*4+hsv(:,:,3);
        edges = {linspace(0,8191,1024)};
        H = cv.calcHist(f_hsv, edges);
        [~,color_hist_max(index)] = max(H);
        color_hist_var(index) = var(H);
        hsv = reshape(hsv, [dat_vid.height*dat_vid.width 3]);
        s_mean(index) = mean(hsv(:,2));
        v_mean(index) = mean(hsv(:,3));
        s_over(index) = sum(hsv(:,2)>s_mean(index)*1.5);
        v_over(index) = sum(hsv(:,3)>v_mean(index)*1.5);
        
        gray_mat = zeros(256,256);
        gray_img = rgb2gray(img);
        [gray_img_1, gray_img_2] = size(gray_img);
        for index_i = 1:gray_img_1
            for index_j = 1:gray_img_2-1
                gray_mat(gray_img(index_i, index_j)+1, gray_img(index_i, index_j+1)+1) = gray_mat(gray_img(index_i, index_j)+1, gray_img(index_i, index_j+1)+1) + 1;
                gray_mat(gray_img(index_i, index_j+1)+1, gray_img(index_i, index_j)+1) = gray_mat(gray_img(index_i, index_j+1)+1, gray_img(index_i, index_j)+1) + 1;
            end
        end

        miu_i = sum(sum(linspace(1,256,256)'*ones(1,256).*gray_mat));
        miu_j = miu_i;
        sigma_i = sum(sum((linspace(1,256,256)'*ones(1,256)-miu_i).^2.*gray_mat));
        sigma_j = sigma_i;

        contrast(index) = sum(sum((2 - linspace(1,256,256)'*ones(1,256)).^2.*gray_mat));
        similarity(index) = sum(sum(gray_mat./(1+(2-linspace(1,256,256)'*ones(1,256)).^2)));
        energy(index) = sum(sum(gray_mat.^2));
        entropy(index) = sum(sum(gray_mat(gray_mat>0).*log2(gray_mat(gray_mat>0))));
        correlation(index) = sum(sum(gray_mat.*(linspace(1,256,256)'*ones(1,256)-miu_i).*(ones(256,1)*linspace(1,256,256)-miu_j)./sqrt(sigma_i*sigma_j)));
    end

    %% move feature

    omiga_mean = abs(v_mean_total(1:end-1) - v_mean_total(2:end));
    omiga_var = abs(v_var(1:end-1) - v_var(2:end));
    feature(3) = sum(omiga_mean > mean(omiga_mean)*1.2)/totalFrames;
    feature(4) = sum((omiga_var < mean(omiga_var)*0.8) & (omiga_mean < mean(omiga_mean)*0.8))/totalFrames;
    feature(5) = sum(rgb_diff)/totalFrames;

    output = [feature;color_hist_max/32;color_hist_var/100000000; ...
    v_mean/100;s_mean/100;v_over/100000;s_over/100000;contrast/10000000000;similarity/100; ...
    energy/100000000;entropy/100000000;correlation]';
end
