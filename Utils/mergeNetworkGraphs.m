function lgraph = mergeNetworkGraphs(lgraph1, lgraph2)  

    layers = [lgraph1.Layers; lgraph2.Layers];
    
    lgraph = layerGraph(layers);
    defaultConnections = lgraph.Connections;
    connections = [lgraph1.Connections; lgraph2.Connections];

    numOfDefaultConnections = height(defaultConnections);
    for k = 1:numOfDefaultConnections
        lgraph = disconnectLayers(lgraph,...
            string(defaultConnections(k,"Source").Variables),...
            string(defaultConnections(k,"Destination").Variables));
    end

    numOfConnections = height(connections);
    for k = 1:numOfConnections
        lgraph = connectLayers(lgraph,...
            string(connections(k,"Source").Variables),...
            string(connections(k,"Destination").Variables));
    end
end