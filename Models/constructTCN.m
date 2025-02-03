function dlnet = constructTCN(parameters)
    % arguments
    %     numChannels     = 1
    %     numFilters      = 16
    %     filterSize      = 21 
    %     numBlocks       = 2
    %     dropoutFactor   = 0.1
    %     numReponse      = 4
    % end

    layer  = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)
              cbt2sscbLayer(Name="t2s")]; % Normalization removed 2023.07
    lgraph = layerGraph(layer);
    
    outputName = layer(end).Name;
    
    for k = 1:parameters.numBlocks
        dilationFactor = 2^(k-1);
        if(k == 1)
        layers = [
            convolution2dLayer([parameters.filterSize, 1], parameters.numFilters, DilationFactor=dilationFactor, Padding="same", Name="conv_1_" + k)
            layerNormalizationLayer
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
            groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same")
            convolution2dLayer(1, parameters.numFilters)
            layerNormalizationLayer
            swishLayer %reluLayer
            spatialDropoutLayer(parameters.dropoutFactor)
            additionLayer(2,Name="add_" + k)];
        else
            layers = [
            groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same", Name="conv_1_" + k)
            convolution2dLayer(1, parameters.numFilters, Name="conv_2_" + k)
            layerNormalizationLayer
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
            groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same")
            convolution2dLayer(1, parameters.numFilters)
            layerNormalizationLayer
            swishLayer %reluLayer
            spatialDropoutLayer(parameters.dropoutFactor)
            additionLayer(2,Name="add_"+k)];        
        end

        % Add and connect layers.
        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"conv_1_"+k);

        % layers = [
        %     convolution1dLayer(parameters.filterSize, parameters.numFilters, DilationFactor=dilationFactor, Padding="same", Name="conv1_" + k)
        %     layerNormalizationLayer
        %     spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
        %     convolution1dLayer(parameters.filterSize, parameters.numFilters, DilationFactor=dilationFactor, Padding="same")
        %     layerNormalizationLayer
        %     swishLayer %reluLayer
        %     spatialDropoutLayer(parameters.dropoutFactor)
        %     additionLayer(2,Name="add_"+k)];
        % 
        % % Add and connect layers.
        % lgraph = addLayers(lgraph,layers);
        % lgraph = connectLayers(lgraph,outputName,"conv1_"+k);
        % 
        % Skip connection.
        if true
            % Include convolution in first skip connection.
            layer = convolution2dLayer(1, parameters.numFilters,Name="convSkip_" + k);
    
            lgraph = addLayers(lgraph,layer);
            lgraph = connectLayers(lgraph, outputName,"convSkip_" + k);
            lgraph = connectLayers(lgraph, "convSkip_" + k,"add_" + k + "/in2");
        else
            lgraph = connectLayers(lgraph, outputName,"add_" + k + "/in2");
        end
        
        % Update layer output name.
        outputName = "add_" + k;
    end
    
    layers = [
        sscb2cbtLayer(Name="s2t")
        flattenLayer(Name="fl")
        fullyConnectedLayer(parameters.numLabels, Name="fc")
        softmaxLayer
        ];
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph, outputName, "s2t");
    dlnet = dlnetwork(lgraph);
end