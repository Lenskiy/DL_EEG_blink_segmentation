function dlnet = constructCNNTransformer(parameters)
    assert(mod(parameters.embedDim, parameters.numHeads) == 0, ...
        'embedDim (%d) must be divisible by numHeads (%d)', ...
        parameters.embedDim, parameters.numHeads);
    
    keyDim = parameters.embedDim / parameters.numHeads;
    
    layers  = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)
               cbt2sscbLayer(Name="t2s")]; 
    lgraph = layerGraph(layers);
    
    outputName = layers(end).Name;
    
    for k = 1:parameters.numCNNBlocks
        layers = [
            convolution2dLayer([parameters.filterSize 1], parameters.embedDim, Padding="same", Name="conv_" + k)
            layerNormalizationLayer(Name="ln_" + k)
            swishLayer(Name="swish_" + k)
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
            ];

        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, outputName, "conv_" + k);
        outputName = "dropout_" + k;
    end
    
    layers = [
        sscb2cbtLayer(Name="s2t")
        flattenLayer(Name="fl")
        ];
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph, outputName, "s2t");
    outputName = "fl";

    
    % Add transformer blocks (no format conversion needed)
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