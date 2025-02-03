function lgraph = bigruLayerGraph(numHiddenUnits, NameValueArgs)
    arguments
        numHiddenUnits
        NameValueArgs.Name = ''
        NameValueArgs.OutputMode = 'sequence'
    end

    layers = [functionLayer(@(x) x, Name=NameValueArgs.Name + "_input")
                          gruLayer(numHiddenUnits, OutputMode=NameValueArgs.OutputMode, Name=NameValueArgs.Name + "_gru_1")
                          concatenationLayer(1, 2, Name=NameValueArgs.Name)
                          ];
                
    lgraph = layerGraph(layers);

    layers = [FlipLayer(Name=NameValueArgs.Name + "_flip_1")
              gruLayer(numHiddenUnits, OutputMode=NameValueArgs.OutputMode, Name=NameValueArgs.Name + "_gru_2")
              FlipLayer(Name=NameValueArgs.Name + "_flip_2")];

    lgraph = addLayers(lgraph, layers);

    lgraph = connectLayers(lgraph, NameValueArgs.Name + "_input", NameValueArgs.Name + "_flip_1");
    lgraph = connectLayers(lgraph, NameValueArgs.Name + "_flip_2", NameValueArgs.Name + "/in2");
end
