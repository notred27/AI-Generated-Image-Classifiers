classdef Bayer
    % Bayer Enumeration of common Color Filter Array (CFA) Bayer arrangements.
    %
    % A Bayer array defines which color channel (R, G, or B) is captured at
    % each pixel position in a digital camera sensor. The arrangement is named
    % by reading the top-left 2x2 grid of filters left-to-right, top-to-bottom.
    %
    % For example, GRBG looks like:
    %   G R
    %   B G
    %
    % The chosen arrangement must match the Bayer pattern of the source images.
    % In our experiments, GRBG produced the best results on the sparse dataset
    % (76.42% accuracy, 0.696 F1 with bicubic interpolation).
    %
    % Usage:
    %   classifier = CFA_FeatureGenerator(InterpolationAlgorithms.Bicubic, Bayer.GRBG);

    enumeration
        BGGR,   % Blue-Green / Green-Red
        RGGB,   % Red-Green / Green-Blue
        GRBG,   % Green-Red / Blue-Green (best performing in paper)
        GBRG    % Green-Blue / Red-Green
    end
end