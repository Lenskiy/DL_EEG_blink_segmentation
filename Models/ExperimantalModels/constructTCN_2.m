function lgraph = constructTCN(parameters)
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
    layers  = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)
              functionLayer(@(x) repelem(x, parameters.numFilters, 1, 1), Name="repeatInputLayer")]; % Normalization removed 2023.07
    lgraph = layerGraph(layers);
    
    outputName = layers(end).Name;
    
    for k = 1:parameters.numBlocks
        dilationFactor = 2^(k-1);

        lgraph = addLayers(lgraph, depthConcatenationLayer(parameters.numFilters, Name="dc_" + k));
        
        for l=1:parameters.numFilters
            layers = [
                functionLayer(@(z) z((l-1)*parameters.numChannels+1:l*parameters.numChannels,:,:),'Name',"channels_" + k + "_" + l)
                cbt2sscbLayer(Name="t2s_" + k + "_" + l)
                groupedConvolution2dLayer([1 parameters.filterSize], 1, "channel-wise", DilationFactor=dilationFactor, Padding="same", Name="conv_1_" + k + "_" + l)
                sscb2cbtLayer(Name="s2t_" + k + "_" + l)
                layerNormalizationLayer(Name="ln_" + k + "_" + l)% spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k + "_" + l)
                convolution1dLayer(1,parameters.numChannels, DilationFactor=dilationFactor, Padding="same", Name="conv_2_" + k + "_" + l)];
                
                lgraph = addLayers(lgraph, layers);
                lgraph = connectLayers(lgraph, outputName, "channels_" + k + "_" + l);
                lgraph = connectLayers(lgraph, "conv_2_" + k + "_" + l,  "dc_" + k + "/in" + l);
        end

            layers = [
                layerNormalizationLayer(Name="ln_" + k)
                swishLayer(Name="sl_" + k) %reluLayer
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                additionLayer(2, Name="add_" + k)];

        

        % Add and connect layers.
        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, "dc_" + k, "ln_" + k);
        %lgraph = connectLayers(lgraph, outputName,"conv_1_" + k);
    
        % Skip connection.
        if k == 1
            % Include convolution in first skip connection.
            layer = convolution1dLayer(1, parameters.numChannels*parameters.numFilters,Name="convSkip");
    
            lgraph = addLayers(lgraph,layer);
            lgraph = connectLayers(lgraph, outputName,"convSkip");
            lgraph = connectLayers(lgraph, "convSkip","add_" + k + "/in2");
        else
            lgraph = connectLayers(lgraph, outputName,"add_" + k + "/in2");
        end
        
        % Update layer output name.
        outputName = "add_" + k;
    end
    
    layers = [
        fullyConnectedLayer(parameters.numLabels, Name="fc")
        softmaxLayer
        classificationLayer("Name",'classificationLayer')
        ];
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph, outputName, "fc");
    %analyzeNetwork(lgraph)
end