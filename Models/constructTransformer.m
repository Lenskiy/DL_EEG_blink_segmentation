function dlnet = constructTransformer(parameters)
    % Ensure embedDim is divisible by numHeads
    assert(mod(parameters.embedDim, parameters.numHeads) == 0, ...
        'embedDim (%d) must be divisible by numHeads (%d)', ...
        parameters.embedDim, parameters.numHeads);
    
    % Calculate key dimension
    keyDim = parameters.embedDim / parameters.numHeads;
    
    layers = [
        sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)
        fullyConnectedLayer(parameters.embedDim, Name="embedding")
        positionEmbeddingLayer(parameters.embedDim, 384, Name="posEmb") % 384 needs to be updated for different size of segment
    ];
    
    lgraph = layerGraph(layers);
    outputName = "posEmb";
    
    % Add transformer blocks
    for k = 1:parameters.numBlocks
        attentionBlock = [
            layerNormalizationLayer(Name="attnNorm_" + k)
            selfAttentionLayer(parameters.numHeads, keyDim, ...
                'Name', "attention_" + k, ...
                'NumValueChannels', parameters.embedDim, ...
                'OutputSize', parameters.embedDim, ...
                'DropoutProbability', parameters.dropoutFactor)
            additionLayer(2, Name="attnAdd_" + k)
            
            layerNormalizationLayer(Name="ffnNorm_" + k)
            fullyConnectedLayer(4*parameters.embedDim, Name="ffn1_" + k)
            swishLayer(Name="swish1_" + k)
            dropoutLayer(parameters.dropoutFactor, Name="ffnDrop1_" + k)
            fullyConnectedLayer(parameters.embedDim, Name="ffn2_" + k)
            dropoutLayer(parameters.dropoutFactor, Name="ffnDrop2_" + k)
            additionLayer(2, Name="ffnAdd_" + k)
        ];
        
        lgraph = addLayers(lgraph, attentionBlock);
        
        lgraph = connectLayers(lgraph, outputName, "attnAdd_" + k + "/in2");
        lgraph = connectLayers(lgraph, outputName, "attnNorm_" + k);
        lgraph = connectLayers(lgraph, "attnAdd_" + k, "ffnAdd_" + k + "/in2");
        
        outputName = "ffnAdd_" + k;
    end
    
    outputLayers = [
        fullyConnectedLayer(parameters.numLabels, Name="fc")
        softmaxLayer(Name="softmax")
    ];
    
    lgraph = addLayers(lgraph, outputLayers);
    lgraph = connectLayers(lgraph, outputName, "fc");
    
    dlnet = dlnetwork(lgraph);
end