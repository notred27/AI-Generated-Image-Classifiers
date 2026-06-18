classdef CFA_FeatureGenerator
    % CFA_FeatureGenerator Generates CFA interpolation-based features for
    % distinguishing AI-generated images from genuine photographs.
    %
    % The core idea: genuine camera images are produced by interpolating raw
    % Bayer array data, leaving behind low-level pixel correlation artifacts.
    % AI-generated images (GANs, LDMs) are never raw-interpolated, so they
    % lack these artifacts. This class exploits that difference by
    % re-interpolating an input image and measuring the percent error between
    % the original and re-interpolated versions.
    %
    % Two features are produced (as described in the paper):
    %   F1 - scalar sum of per-channel percent errors
    %   F2 - 3x1 vector of per-channel (R, G, B) percent errors
    %
    % Reference:
    %   "Investigating the Effectiveness of Deep Learning and CFA
    %    Interpolation Based Classifiers on Identifying AIGC" (IEEE BigData 2023)
    %   Reidy, Mallon, Luo — DOI: 10.1109/BigData59044.2023.10386096
    %
    % Usage:
    %   classifier = CFA_FeatureGenerator(InterpolationAlgorithms.Bicubic, Bayer.GRBG);
    %   img = imread('image.jpg');
    %   pe = percentError(classifier, img);  % Returns 3x1 F2 feature vector
    %   f1 = sum(pe);                        % Scalar F1 feature

    properties (Access = private)
        % InterpolationAlgorithms enum — algorithm used during re-interpolation
        selectedAlgorithm

        % Bayer enum — CFA pattern used to approximate the raw image
        bayerArrangement
    end

    methods

        function obj = CFA_FeatureGenerator(selectedAlgo, selectedBayer)
            % CFA_FeatureGenerator Constructor.
            %
            % Input:
            %   selectedAlgo  - InterpolationAlgorithms enum value
            %                   (Bilinear, Bicubic, SmoothHue, or Gradient)
            %   selectedBayer - Bayer enum value
            %                   (GRBG, GBRG, BGGR, or RGGB)

            obj.selectedAlgorithm = selectedAlgo;
            obj.bayerArrangement  = selectedBayer;
        end


        function image = imageToRaw(obj, image)
            % imageToRaw Approximates a raw Bayer image from a full-color
            % RGB image by zeroing out pixels not captured by each channel's
            % color filter, according to the selected Bayer arrangement.
            %
            % Each color filter in a Bayer array only captures one channel
            % per pixel. This method simulates that by masking out the
            % non-captured channels at each pixel position.
            %
            % Input:
            %   image - n x m x 3 uint8 full-color RGB image
            % Output:
            %   image - n x m x 3 uint8 raw (single-channel-per-pixel) image

            switch obj.bayerArrangement

                case Bayer.GRBG
                    image(2:2:end, :,       1) = 0;  % Red: keep odd rows, odd cols
                    image(:,       1:2:end, 1) = 0;
                    image(2:2:end, 1:2:end, 2) = 0;  % Green: keep checkerboard
                    image(1:2:end, 2:2:end, 2) = 0;
                    image(1:2:end, :,       3) = 0;  % Blue: keep even rows, even cols
                    image(:,       2:2:end, 3) = 0;

                case Bayer.GBRG
                    image(1:2:end, :,       1) = 0;  % Red
                    image(:,       2:2:end, 1) = 0;
                    image(2:2:end, 1:2:end, 2) = 0;  % Green
                    image(1:2:end, 2:2:end, 2) = 0;
                    image(2:2:end, :,       3) = 0;  % Blue
                    image(:,       1:2:end, 3) = 0;

                case Bayer.BGGR
                    image(1:2:end, :,       1) = 0;  % Red
                    image(:,       1:2:end, 1) = 0;
                    image(1:2:end, 1:2:end, 2) = 0;  % Green
                    image(2:2:end, 2:2:end, 2) = 0;
                    image(2:2:end, :,       3) = 0;  % Blue
                    image(:,       2:2:end, 3) = 0;

                case Bayer.RGGB
                    image(2:2:end, :,       1) = 0;  % Red
                    image(:,       2:2:end, 1) = 0;
                    image(1:2:end, 1:2:end, 2) = 0;  % Green
                    image(2:2:end, 2:2:end, 2) = 0;
                    image(1:2:end, :,       3) = 0;  % Blue
                    image(:,       1:2:end, 3) = 0;

                otherwise
                    error("Invalid Bayer arrangement. Use a Bayer enum value: GRBG, GBRG, BGGR, or RGGB.");
            end
        end


        function img = interpolateImage(obj, img)
            % interpolateImage Converts a full-color image to an approximate
            % raw image and re-interpolates it using the selected algorithm.
            %
            % This is the core re-interpolation step: the output should match
            % the input closely for genuine images (low percent error) and
            % differ significantly for AI-generated images (high percent error).
            %
            % Input:
            %   img - n x m x 3 uint8 full-color RGB image
            % Output:
            %   img - n x m x 3 uint8 re-interpolated image

            % Step 1: approximate the raw Bayer image
            img = imageToRaw(obj, img);

            % Step 2: re-interpolate using the selected algorithm
            switch obj.selectedAlgorithm
                case InterpolationAlgorithms.Bilinear
                    img = InterpolationAlgorithms.bilinearInterpolation(img);

                case InterpolationAlgorithms.Bicubic
                    img = InterpolationAlgorithms.bicubicInterpolation(img);

                case InterpolationAlgorithms.SmoothHue
                    img = InterpolationAlgorithms.smoothHueInterpolation(img, obj.bayerArrangement);

                case InterpolationAlgorithms.Gradient
                    img = InterpolationAlgorithms.gradientInterpolation(img, obj.bayerArrangement);

                otherwise
                    error("Invalid algorithm. Use an InterpolationAlgorithms enum value: Bilinear, Bicubic, SmoothHue, or Gradient.");
            end
        end


        function pe = percentError(obj, img)
            % percentError Computes the per-channel percent error between
            % the original image and its re-interpolated version (equation 11).
            %
            % This produces the F2 feature: a 3x1 vector [Re, Ge, Be]
            % representing the mean absolute difference per channel, normalized
            % by image dimensions. Summing the result gives the scalar F1.
            %
            % Input:
            %   img - n x m x 3 uint8 full-color RGB image
            % Output:
            %   pe  - 3x1 double vector of per-channel percent errors [R; G; B]
            %         (F2 feature from the paper)

            reinterpolated = interpolateImage(obj, img);

            % Mean absolute difference per channel (equation 11)
            diff = abs(double(img) - double(reinterpolated));
            [w, h, ~] = size(img);

            pe = [sum(diff(:,:,1), "all") / (w * h);
                  sum(diff(:,:,2), "all") / (w * h);
                  sum(diff(:,:,3), "all") / (w * h)];
        end

    end
end