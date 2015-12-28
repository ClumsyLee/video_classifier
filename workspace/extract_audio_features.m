function features = extract_audio_features(audio)
    data = audio.data;
    len = length(data);

    % Energy.
    energy = sum(data.^2) / len;

    % % Energy entropy.
    % non_zero_energies = (data).^2;
    % non_zero_energies = non_zero_energies(non_zero_energies > 1e-6);
    % entropy = -sum(non_zero_energies .* log2(non_zero_energies));

    % ZCR.
    pos = data > 0;
    neg = data < 0;
    zcr = sum(pos(1:end-1) & neg(2:end) | ...
              neg(1:end-1) & pos(2:end)) / (len - 1);

    spectrum = abs(spectrogram(data));
    spectrum_var = std2(spectrum)^2;
    spectrum_max = max(max(spectrum));

    features = [energy * 100; zcr * 100; spectrum_var / 1000; spectrum_max / 1000];
