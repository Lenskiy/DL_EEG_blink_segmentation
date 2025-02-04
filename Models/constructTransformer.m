function dlnet = constructTransformer(parameters)
    assert(mod(parameters.embedDim, parameters.numHeads) == 0, ...
        'embedDim (%d) must be divisible by numHeads (%d)', ...
        parameters.embedDim, parameters.numHeads);
    
    keyDim = parameters.embedDim / parameters.numHeads;
    
    % Input and embedding layers
    layers = [
        sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)
        layerNormalizationLayer(Name="inputNorm")
        fullyConnectedLayer(parameters.embedDim, Name="embedding", ...
            WeightsInitializer='glorot', ...
            BiasInitializer='zeros')
        dropoutLayer(parameters.dropoutFactor/2, Name="embedDrop")
        positionEmbeddingLayer(parameters.embedDim, 384, Name="posEmb")
        layerNormalizationLayer(Name="posNorm")
    ];
    
    lgraph = layerGraph(layers);
    outputName = "posNorm";
    
    % Add transformer blocks
    for k = 1:parameters.numBlocks
        % Create complete transformer block
        attentionBlock = [
            % Self-attention
            layerNormalizationLayer(Name="preAttnNorm_" + k)
            selfAttentionLayer(parameters.numHeads, keyDim, ...
                'Name', "attention_" + k, ...
                'NumValueChannels', parameters.embedDim, ...
                'OutputSize', parameters.embedDim, ...
                'DropoutProbability', parameters.dropoutFactor, ...
                'WeightsInitializer', 'glorot', ...
                'BiasInitializer', 'zeros')
            dropoutLayer(parameters.dropoutFactor, Name="attnDrop_" + k)
            additionLayer(2, Name="attnAdd_" + k)
            
            % Feed-forward network
            layerNormalizationLayer(Name="preFFNorm_" + k)
            fullyConnectedLayer(4*parameters.embedDim, Name="ffn1_" + k, ...
                WeightsInitializer='glorot', ...
                BiasInitializer='zeros')
            swishLayer(Name="swish1_" + k)
            dropoutLayer(parameters.dropoutFactor, Name="ffnDrop1_" + k)
            fullyConnectedLayer(parameters.embedDim, Name="ffn2_" + k, ...
                WeightsInitializer='glorot', ...
                BiasInitializer='zeros')
            dropoutLayer(parameters.dropoutFactor, Name="ffnDrop2_" + k)
            additionLayer(2, Name="ffnAdd_" + k)
        ];
        
        % Add block to graph
        lgraph = addLayers(lgraph, attentionBlock);
        
        % Connect residual paths only
        lgraph = connectLayers(lgraph, outputName, "attnAdd_" + k + "/in2");
        lgraph = connectLayers(lgraph, outputName, "preAttnNorm_" + k);
        lgraph = connectLayers(lgraph, "attnAdd_" + k, "ffnAdd_" + k + "/in2");
        
        outputName = "ffnAdd_" + k;
    end
    
    % Output layers
    outputLayers = [
        layerNormalizationLayer(Name="outputNorm")
        dropoutLayer(parameters.dropoutFactor, Name="outputDrop")
        fullyConnectedLayer(parameters.numLabels, Name="fc", ...
                WeightsInitializer='glorot', ...
                BiasInitializer='zeros')
        softmaxLayer(Name="softmax")
    ];
    
    lgraph = addLayers(lgraph, outputLayers);
    lgraph = connectLayers(lgraph, outputName, "outputNorm");
    
    dlnet = dlnetwork(lgraph);
end