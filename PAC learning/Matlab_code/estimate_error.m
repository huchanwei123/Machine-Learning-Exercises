% function used to estimate error of hs.
function error = estimate_error(concept, hs, r)
    %in_C = [];
    error_point = 0;
    for point = 1:size(r,1)
        if(r(point, 1) > concept(1,1) & r(point, 1) < concept(2,1) & r(point, 2) > concept(1,2) & r(point, 2) < concept(2,2))
            %in_C = [in_C; r(point, 1), r(point, 2)];
            if(r(point, 1) < hs(1,1) | r(point, 1) > hs(2,1) | r(point, 2) < hs(1,2) | r(point, 2) > hs(2,2))
                error_point = error_point +1;
            end
        end
    end
    error = error_point/size(r,1);
end