function mfccs = my_mfcc(audio)
    addpath('mfcc/mfcc');

    Tw = 20;           % analysis frame duration (ms)
    Ts = 10;           % analysis frame shift (ms)
    alpha = 0.97;      % preemphasis coefficient
    R = [ 300 3700 ];  % frequency range to consider
    M = 20;            % number of filterbank channels
    C = 13;            % number of cepstral coefficients
    L = 22;            % cepstral sine lifter parameter

    % hamming window (see Eq. (5.2) on p.73 of [1])
    hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));

    % Feature extraction (feature vectors as columns)
    mfccs = [];
    for k = 1:size(audio.data, 2)
        mfccs = [mfccs, mfcc(audio.data(:, k), audio.rate, Tw, Ts, alpha, hamming, R, M, C, L)];
    end

    % Remove NAN cols.
    mfccs = mfccs(:, ~any(isnan(mfccs)));
