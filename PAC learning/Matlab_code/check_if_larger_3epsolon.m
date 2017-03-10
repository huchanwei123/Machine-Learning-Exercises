% function used to check if P(C) >= 2*eposlon.
function concept = check_if_larger_3epsolon(r, epsolon)
    max_x = max(r(:, 1));
    min_x = min(r(:, 1));
    max_y = max(r(:, 2));
    min_y = min(r(:, 2));
    lower_left = min_x + [max_x - min_x, max_y - min_y] * rand(1);
    %lower_left = [-0.5 2.5];
    upper_right = lower_left + [max_x - lower_left(1), max_y - lower_left(2)] * rand(1);
    %upper_right = [3 8];
    concept = [lower_left; upper_right];
    for i = 1:inf
        in_C = 0;
        for point = 1:size(r,1)
            if(r(point, 1) > concept(1,1) & r(point, 1) < concept(2,1) & r(point, 2) > concept(1,2) & r(point, 2) < concept(2,2))
                in_C = in_C + 1;
            end
        end
        if (in_C / size(r, 1) > 3*epsolon)
            disp('The selected empirical probability of concept c has P_hat >= 3*epsolon !!!');
            break;
        else
            lower_left = min_x + [max_x - min_x, max_y - min_y] * rand(1);
            upper_right = lower_left + [max_x - lower_left(1), max_y - lower_left(2)] * rand(1);
            concept = [lower_left; upper_right];
        end
    end
end