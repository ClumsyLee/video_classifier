function saveFrame(data,width,height,frameNr,time)

persistent warned;

try
    scanline = ceil(width*3/4)*4; % the scanline size must be a multiple of 4.

    % some times the amount of data doesn't match exactly what we expect...
    if (numel(data) ~= scanline*height)
        if (numel(data) > 3*width*height)
            if (isemtpy(warned))
                warning('mmread:general','dimensions do not match data size. Guessing badly...');
                warned = true;
            end
            scanline = width*3;
            data = data(1:3*width*height);
        else
            error('dimensions do not match data size. Too little data.');
        end
    end

    % if there is any extra scanline data, remove it
    data = reshape(data,scanline,height);
    data = data(1:3*width,:);

    % the data ordering is wrong for matlab images, so permute it
    data = permute(reshape(data, 3, width, height),[3 2 1]);

    % now do something with the data...
    global  img_dir;
    imwrite(data, num2str(frameNr, [img_dir, '%06d.jpg']));

catch
    % if we don't catch the error here we will loss the stack trace.
    err = lasterror;
    disp([err.identifier ': ' err.message]);
    for i=1:length(err.stack)
        disp([err.stack(i).file ' (' err.stack(i).name ') Line: ' num2str(err.stack(i).line)]);
    end
    rethrow(err);
end
