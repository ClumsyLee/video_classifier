function summary = collect_votes(classify_output, kinds)
    rows = size(classify_output, 1);
    summary = zeros(rows, kinds);
    for row = 1:rows
        col = 1;
        for kind1 = 1:kinds-1
            for kind2 = kind1+1:kinds
                if classify_output(row, col) == 1
                    summary(row, kind1) = summary(row, kind1) + 1;
                    summary(row, kind2) = summary(row, kind2) + 1;
                end
                col = col + 1;
            end
        end
    end
