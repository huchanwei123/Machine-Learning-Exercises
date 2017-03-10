% function used to find hs.
function estimated_hs = find_hs(concept, mu, sigma, m)
    r = get_bivariate_normal_distribution(mu, sigma, m);
    in_C = [];
    for point = 1:size(r,1)
        if(r(point, 1) > concept(1,1) & r(point, 1) < concept(2,1) & r(point, 2) > concept(1,2) & r(point, 2) < concept(2,2))
            in_C = [in_C; r(point, 1), r(point, 2)];
        end
    end
    max_x = max(in_C(:, 1));
    min_x = min(in_C(:, 1));
    max_y = max(in_C(:, 2));
    min_y = min(in_C(:, 2));
    estimated_hs = [min_x, min_y; max_x, max_y];
end