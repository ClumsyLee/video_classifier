classdef TestSqrBoxFilter
    %TestSqrBoxFilter
    properties (Constant)
        img = imread(fullfile(mexopencv.root(),'test','fruits.jpg'));
    end

    methods (Static)
        function test_1
            out = cv.sqrBoxFilter(TestBoxFilter.img);
            validateattributes(out, {'single'}, ...
                {'size',size(TestBoxFilter.img)});
        end

        function test_2
            out = cv.sqrBoxFilter(TestBoxFilter.img, 'DDepth','double', ...
                'KSize',[5 5], 'Anchor',[-1 -1], 'BorderType','Default');
            validateattributes(out, {'double'}, ...
                {'size',size(TestBoxFilter.img)});
        end

        function test_3
            out = cv.sqrBoxFilter(TestBoxFilter.img, 'Normalize',false);
            validateattributes(out, {'single'}, ...
                {'size',size(TestBoxFilter.img)});
        end

        function test_error_1
            try
                cv.sqrBoxFilter();
                throw('UnitTest:Fail');
            catch e
                assert(strcmp(e.identifier,'mexopencv:error'));
            end
        end
    end
end
