classdef InterpolationAlgorithms
    % InterpolationAlgorithms Enumeration class defining CFA interpolation
    % algorithms used to reconstruct full-color images from raw Bayer array data.
    %
    % Each algorithm re-interpolates an input image to compute the CFA percent
    % error feature (F1) described in:
    %
    % Usage:
    %   classifier = CFA_FeatureGenerator(InterpolationAlgorithms.Bicubic, Bayer.GRBG);
    %   err = percentError(classifier, img);
    %
    % Algorithms:
    %   Bilinear  - 3x3 kernel convolution (fast, lower accuracy)
    %   Bicubic   - 7x7 kernel convolution (slower, higher accuracy)
    %   SmoothHue - Hue-ratio based interpolation (assumes locally constant hue)
    %   Gradient  - MATLAB's built-in gradient-corrected linear demosaicing

    enumeration
        Bilinear, Bicubic, SmoothHue, Gradient
    end

    properties (Constant)
        % Bilinear interpolation kernels
        bilinear_g = [[0,    0.25, 0   ];
                      [0.25, 1.0,  0.25];
                      [0,    0.25, 0   ]];

        bilinear_rb = [[0.25, 0.5, 0.25];
                       [0.5,  1.0, 0.5 ];
                       [0.25, 0.5, 0.25]];

        % Bicubic interpolation kernels
        bicubic_g = [[0,  0,  0,   1,  0,  0, 0];
                     [0,  0, -9,   0, -9,  0, 0];
                     [0, -9,  0,  81,  0, -9, 0];
                     [1,  0, 81, 256, 81,  0, 1];
                     [0, -9,  0,  81,  0, -9, 0];
                     [0,  0, -9,   0, -9,  0, 0];
                     [0,  0,  0,   1,  0,  0, 0]] * (1/256);

        bicubic_rb = [[1,   0,   -9,  -16,  -9,  0,   1];
                      [0,   0,    0,    0,    0,  0,   0];
                      [-9,  0,   81,  144,   81,  0,  -9];
                      [-16, 0,  144,  256,  144,  0, -16];
                      [-9,  0,   81,  144,   81,  0,  -9];
                      [0,   0,    0,    0,    0,  0,   0];
                      [1,   0,   -9,  -16,   -9,  0,   1]] * (1/256);
    end

    methods (Static)

        function img = bilinearInterpolation(img)
            % bilinearInterpolation Reconstructs a full-color image from a
            % raw Bayer image using bilinear interpolation (equations 4-6).
            %
            % Each channel is convolved separately: red and blue share the
            % KRB kernel, green uses KG.
            %
            % Input:
            %   img - n x m x 3 uint8 raw (single-channel-per-pixel) image
            % Output:
            %   img - n x m x 3 uint8 full-color interpolated image

            img(:,:,1) = conv2(img(:,:,1), InterpolationAlgorithms.bilinear_rb, "same");
            img(:,:,2) = conv2(img(:,:,2), InterpolationAlgorithms.bilinear_g,  "same");
            img(:,:,3) = conv2(img(:,:,3), InterpolationAlgorithms.bilinear_rb, "same");
        end


        function img = bicubicInterpolation(img)
            % bicubicInterpolation Reconstructs a full-color image from a
            % raw Bayer image using bicubic interpolation (equations 7-8).
            %
            % Uses larger 7x7 kernels (FRB and FG) for smoother reconstruction
            % compared to bilinear. Each channel is convolved separately.
            %
            % Input:
            %   img - n x m x 3 uint8 raw (single-channel-per-pixel) image
            % Output:
            %   img - n x m x 3 uint8 full-color interpolated image

            img(:,:,1) = conv2(img(:,:,1), InterpolationAlgorithms.bicubic_rb, "same");
            img(:,:,2) = conv2(img(:,:,2), InterpolationAlgorithms.bicubic_g,  "same");
            img(:,:,3) = conv2(img(:,:,3), InterpolationAlgorithms.bicubic_rb, "same");
        end


        function img = smoothHueInterpolation(img, bayer)
            % smoothHueInterpolation Reconstructs a full-color image using
            % the smooth hue interpolation algorithm (equations 9-10).
            %
            % Assumes color hue is locally constant between neighboring pixels.
            % Interpolates the green channel first via bilinear, then estimates
            % red and blue from their ratio to the green channel.
            %
            % Input:
            %   img   - n x m x 3 uint8 raw (single-channel-per-pixel) image
            %   bayer - Bayer enum specifying the CFA pattern (e.g. Bayer.GRBG)
            % Output:
            %   img   - n x m x 3 uint8 full-color interpolated image

            img = double(img);

            % Step 1: interpolate green channel with bilinear
            img(:,:,2) = conv2(img(:,:,2), InterpolationAlgorithms.bilinear_g, "same");

            % Step 2: compute hue ratios for red and blue relative to green,
            % then smooth with a 3x3 averaging kernel
            img(:,:,1) = img(:,:,1) ./ img(:,:,2);
            img(:,:,1) = conv2(img(:,:,1), ones(3), "same");

            img(:,:,3) = img(:,:,3) ./ img(:,:,2);
            img(:,:,3) = conv2(img(:,:,3), ones(3), "same");

            % Step 3: define Bayer-pattern-specific weighting matrices
            % that count non-zero values in a 3x3 window per channel
            switch bayer
                case Bayer.BGGR
                    red  = [4 2; 2 1];
                    blue = [1 2; 2 4];
                case Bayer.RGGB
                    red  = [1 2; 2 4];
                    blue = [4 2; 2 1];
                case Bayer.GBRG
                    red  = [2 4; 1 2];
                    blue = [2 1; 4 2];
                case Bayer.GRBG
                    red  = [2 1; 4 2];
                    blue = [2 4; 1 2];
            end

            % Step 4: tile the 2x2 weighting matrices to full image size
            [w, h, ~] = size(img);
            rep = ceil(max(w, h) / 2);
            redFactor  = repmat(red,  rep);
            blueFactor = repmat(blue, rep);

            % Step 5: reconstruct red and blue from hue ratios and green
            img(:,:,1) = img(:,:,2) ./ redFactor(1:w,  1:h) .* img(:,:,1);
            img(:,:,3) = img(:,:,2) ./ blueFactor(1:w, 1:h) .* img(:,:,3);

            img = uint8(img);
        end


        function img = gradientInterpolation(img, bayer)
            % gradientInterpolation Reconstructs a full-color image using
            % MATLAB's built-in gradient-corrected linear demosaicing.
            %
            % Collapses the 3-channel raw image to a single-channel grayscale
            % (by summing channels) then passes it to MATLAB's demosaic().
            %
            % Input:
            %   img   - n x m x 3 uint8 raw (single-channel-per-pixel) image
            %   bayer - Bayer enum specifying the CFA pattern (e.g. Bayer.GRBG)
            % Output:
            %   img   - n x m x 3 uint8 full-color interpolated image

            % Map Bayer enum to the string format required by demosaic()
            switch bayer
                case Bayer.BGGR,  bayerStr = "bggr";
                case Bayer.RGGB,  bayerStr = "rggb";
                case Bayer.GBRG,  bayerStr = "gbrg";
                case Bayer.GRBG,  bayerStr = "grbg";
            end

            % Sum channels to collapse to single-channel raw image,
            % then demosaic using MATLAB's gradient-corrected implementation
            img = img(:,:,1) + img(:,:,2) + img(:,:,3);
            img = demosaic(img, bayerStr);
        end

    end
end