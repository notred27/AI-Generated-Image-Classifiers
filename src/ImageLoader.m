classdef ImageLoader
    % ImageLoader Loads and organizes image file paths from a structured
    % dataset directory into a struct array for CFA feature extraction.
    %
    % Usage:
    %   loader = ImageLoader('./dataset', 'testing');
    %   loader = loadData(loader, 'real', 'fake');
    %   img = imread(loader.data(1).path);

    properties
        % Path to the root dataset folder
        pathToDataset

        % Subset to load — typically 'training' or 'testing'
        datasetType

        % Struct array of loaded image metadata.
        % Each entry contains:
        %   path      - absolute path to the image file
        %   label     - class label ('real' or 'fake')
        %   dataset   - dataset split ('training' or 'testing')
        %   predicted - predicted label, populated after classification (initially empty)
        data
    end

    methods

        function obj = ImageLoader(path, datasetType)
            % ImageLoader Constructor.
            %
            % Input:
            %   path        - path to the root dataset folder
            %   datasetType - subfolder to load, e.g. 'testing' or 'training'

            obj.pathToDataset = path;
            obj.datasetType = datasetType;
        end


        function obj = loadData(obj, positiveLabel, negativeLabel)
            % loadData Scans the dataset directory and loads image metadata
            % into a struct array, combining positive and negative classes.
            %
            % Input:
            %   obj           - ImageLoader instance
            %   positiveLabel - folder name for the positive class (e.g. 'real')
            %   negativeLabel - folder name for the negative class (e.g. 'fake')
            %
            % Output:
            %   obj.data - struct array with fields: path, label, dataset, predicted

            positiveData = ImageLoader.loadClass(obj.pathToDataset, obj.datasetType, positiveLabel);
            negativeData = ImageLoader.loadClass(obj.pathToDataset, obj.datasetType, negativeLabel);

            obj.data = [positiveData, negativeData];
        end

    end


    methods (Static, Access = private)

        function data = loadClass(rootPath, datasetType, label)
            % loadClass Scans a single class folder and returns a struct array
            % of image metadata.
            %
            % dir() returns 2 non-file entries ('.' and '..') which are
            % skipped by offsetting the index by 2.

            fileData = dir(fullfile(rootPath, datasetType, label));
            numFiles = length(fileData) - 2;

            % Pre-allocate struct array
            data(numFiles) = struct( ...
                "path",      "", ...
                "label",     label, ...
                "dataset",   datasetType, ...
                "predicted", []);

            for i = 1:numFiles
                entry = fileData(i + 2);
                % Use fullfile for cross-platform path separators
                data(i).path = fullfile(entry.folder, entry.name);
            end
        end

    end
end