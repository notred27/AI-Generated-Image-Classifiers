% CFA Percent Error Classifier Demo

% Computes F1 features for a single image using four interpolation algorithms
% and classifies it as genuine or AI-generated based on the threshold from:
addpath(fullfile(fileparts(mfilename('fullpath')), 'src'));



% Change these to your desired inputs
IMAGE_PATH = './sample_images/test.jpg';
THRESHOLD = 1.33;

% Load image
img = imread(IMAGE_PATH);

% Initialize classifiers for each interpolation algorithm
classifier_bilinear  = CFA_FeatureGenerator(InterpolationAlgorithms.Bilinear,  Bayer.GRBG);
classifier_bicubic   = CFA_FeatureGenerator(InterpolationAlgorithms.Bicubic,   Bayer.GRBG);
classifier_smoothhue = CFA_FeatureGenerator(InterpolationAlgorithms.SmoothHue, Bayer.GBRG);
classifier_gradient  = CFA_FeatureGenerator(InterpolationAlgorithms.Gradient,  Bayer.GBRG);

% Compute F1 percent error feature for each algorithm
err_bilinear  = percentError(classifier_bilinear,  img);
err_bicubic   = percentError(classifier_bicubic,   img);
err_smoothhue = percentError(classifier_smoothhue, img);
err_gradient  = percentError(classifier_gradient,  img);

% Display results
fprintf('\n--- CFA Percent Error (F1) ---\n');
fprintf('Bilinear:   %.4f\n', err_bilinear);
fprintf('Bicubic:    %.4f\n', err_bicubic);
fprintf('Smooth Hue: %.4f\n', err_smoothhue);
fprintf('Gradient:   %.4f\n', err_gradient);

% Use minimum F1 across algorithms for classification (per paper)
min_err = min([err_bilinear, err_bicubic, err_smoothhue, err_gradient]);
fprintf('\nMinimum F1: %.4f (threshold: %.2f)\n', min_err, THRESHOLD);

% Classify
if min_err > THRESHOLD
    fprintf('Classification: AI-generated\n');
else
    fprintf('Classification: Genuine\n');
end