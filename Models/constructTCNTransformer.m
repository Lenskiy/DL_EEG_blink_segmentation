function dlnet = constructTCNTransformer(parameters)
    assert(mod(parameters.embedDim, parameters.numHeads) == 0, ...
        'embedDim (%d) must be divisible by numHeads (%d)', ...
        parameters.embedDim, parameters.numHeads);
    
    keyDim = parameters.embedDim / parameters.numHeads;
    
    % Initial layers
    layers = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)
             cbt2sscbLayer(Name="t2s")];
    lgraph = layerGraph(layers);
    outputName = layers(end).Name;
    
    % TCN blocks with dilated convolutions
    for k = 1:parameters.numTCNBlocks
        dilationFactor = 2^(k-1);
        if(k == 1)
            layers = [
                convolution2dLayer([parameters.filterSize, 1], parameters.embedDim, DilationFactor=dilationFactor, Padding="same", Name="conv_1_" + k)
                layerNormalizationLayer
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same")
                convolution2dLayer(1, parameters.embedDim)
                layerNormalizationLayer
                swishLayer
                spatialDropoutLayer(parameters.dropoutFactor)
                additionLayer(2,Name="add_" + k)];
        else
            layers = [
                groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same", Name="conv_1_" + k)
                convolution2dLayer(1, parameters.embedDim, Name="conv_2_" + k)
                layerNormalizationLayer
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same")
                convolution2dLayer(1, parameters.embedDim)
                layerNormalizationLayer
                swishLayer
                spatialDropoutLayer(parameters.dropoutFactor)
                additionLayer(2,Name="add_"+k)];        
        end

        % Add and connect layers
        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"conv_1_"+k);

        % Skip connection
        layer = convolution2dLayer(1, parameters.embedDim,Name="convSkip_" + k);
        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph, outputName,"convSkip_" + k);
        lgraph = connectLayers(lgraph, "convSkip_" + k,"add_" + k + "/in2");
        
        outputName = "add_" + k;
    end
    
    % Convert to format for transformer
    layers = [
        sscb2cbtLayer(Name="s2t")
        flattenLayer(Name="fl")
    ];
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph, outputName, "s2t");
    outputName = "fl";
    
    % Add transformer blocks
    for k = 1:parameters.numTransformerBlocks
        attentionBlock = [
            layerNormalizationLayer(Name="preAttnNorm_" + k)
            selfAttentionLayer(parameters.numHeads, keyDim, ...
                'Name', "attention_" + k, ...
                'NumValueChannels', parameters.embedDim, ...
                'OutputSize', parameters.embedDim, ...
                'DropoutProbability', parameters.dropoutFactor)
            additionLayer(2, Name="attnAdd_" + k)
            
            layerNormalizationLayer(Name="preFFNorm_" + k)
            fullyConnectedLayer(4*parameters.embedDim, Name="ffn1_" + k)
            swishLayer(Name="swish1_" + k)
            fullyConnectedLayer(parameters.embedDim, Name="ffn2_" + k)
            dropoutLayer(parameters.dropoutFactor, Name="ffnDrop_" + k)
            additionLayer(2, Name="ffnAdd_" + k)
        ];
        
        lgraph = addLayers(lgraph, attentionBlock);
        
        lgraph = connectLayers(lgraph, outputName, "attnAdd_" + k + "/in2");
        lgraph = connectLayers(lgraph, outputName, "preAttnNorm_" + k);
        lgraph = connectLayers(lgraph, "attnAdd_" + k, "ffnAdd_" + k + "/in2");
        
        outputName = "ffnAdd_" + k;
    end
    
    % Output layers
    outputLayers = [
        layerNormalizationLayer(Name="outputNorm")
        dropoutLayer(parameters.dropoutFactor, Name="outputDrop")
        fullyConnectedLayer(parameters.numLabels, Name="fc")
        softmaxLayer(Name="softmax")
    ];
    
    lgraph = addLayers(lgraph, outputLayers);
    lgraph = connectLayers(lgraph, outputName, "outputNorm");
    
    dlnet = dlnetwork(lgraph);
end